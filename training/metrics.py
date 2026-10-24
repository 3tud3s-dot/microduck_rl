"""Training-local episode accounting and compact PPO metrics; no env writes."""

import math

import torch


class EpisodeTracker:
    """Accumulate complete episodes across rollout horizons, one row per env."""

    def __init__(self, env):
        self.n, self.device = env.num_envs, env.device
        self.dt = getattr(env, "step_dt", None)
        self.returns = torch.zeros(self.n, device=self.device, dtype=torch.float64)
        self.lengths = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.indices = torch.zeros_like(self.lengths)
        self.vector_steps = 0
        self.end_term_names = tuple(env.termination_manager.active_terms)
        manager = getattr(env, "reward_manager", None)
        self.term_names = tuple(manager.active_terms) if manager is not None else ()
        if self.term_names and (self.dt is None or not hasattr(manager, "_step_reward")
                                or not hasattr(manager, "_scale_by_dt")):
            raise ValueError("unsupported reward manager accounting layout")
        self.term_returns = torch.zeros((self.n, len(self.term_names)), device=self.device, dtype=torch.float64)
        self.begin_iteration()

    def begin_iteration(self):
        # Keep unfinished episode accumulators; clear only this round's reports.
        self.completed = []
        self.term_sum = torch.zeros(len(self.term_names), device=self.device, dtype=torch.float64)
        self.sample_count = 0
        self.weights = {}

    @torch.no_grad()
    def observe(self, env, record):
        """Called after step and BEFORE reset can clear termination/reward buffers."""
        rewards = record["rewards"]
        if rewards.shape != (self.n,) or not torch.isfinite(rewards).all():
            raise ValueError("episode accounting requires finite [env] rewards")
        self.vector_steps += 1
        self.sample_count += self.n
        self.returns += rewards.double()
        self.lengths += 1
        if self.term_names:
            manager = env.reward_manager
            # mjlab stores weighted RATE in _step_reward, whereas step() returns
            # dt-scaled reward by default. Convert once to actual contribution.
            scale = self.dt if manager._scale_by_dt else 1.0
            terms = manager._step_reward.detach().double().clone() * scale
            if terms.shape != self.term_returns.shape or not torch.isfinite(terms).all():
                raise ValueError("invalid reward term batch")
            torch.testing.assert_close(terms.sum(-1), rewards.double(), rtol=1e-5, atol=1e-6)
            self.term_returns += terms
            self.term_sum += terms.sum(0)
            for name in self.term_names:
                weight = float(manager.get_term_cfg(name).weight)
                if not math.isfinite(weight):
                    raise ValueError("non-finite reward weight")
                old = self.weights.get(name, (weight, weight))
                self.weights[name] = (min(old[0], weight), max(old[1], weight))
        ended = record["terminated"] | record["truncated"]
        for index in ended.nonzero().flatten().tolist():
            self.indices[index] += 1
            reasons = [name for name in env.termination_manager.active_terms
                       if env.termination_manager.get_term(name)[index].item()]
            steps = self.lengths[index].item()
            self.completed.append({"env_index": index, "episode_index": self.indices[index].item(),
                                   "end_vector_step": self.vector_steps, "end_env_steps": self.vector_steps * self.n,
                                   "return": self.returns[index].item(), "steps": steps,
                                   "duration_s": steps * self.dt if self.dt is not None else None,
                                   "terminated": record["terminated"][index].item(),
                                   "truncated": record["truncated"][index].item(), "end_terms": reasons,
                                   "reward_terms": {name: self.term_returns[index, i].item()
                                                    for i, name in enumerate(self.term_names)}})
            self.returns[index] = 0
            self.lengths[index] = 0
            self.term_returns[index] = 0

    def summary(self):
        count = len(self.completed)
        mean = lambda key: sum(e[key] for e in self.completed) / count if count else None
        terminations = {}
        for episode in self.completed:
            for name in episode["end_terms"]:
                terminations[name] = terminations.get(name, 0) + 1
        return {"completed_count": count, "mean_return": mean("return"), "mean_steps": mean("steps"),
                "mean_duration_s": mean("duration_s") if self.dt is not None else None,
                "terminated_fraction": mean("terminated"), "truncated_fraction": mean("truncated"),
                "both_flags_count": sum(e["terminated"] and e["truncated"] for e in self.completed),
                "end_term_counts": terminations,
                "fell_over_fraction": terminations.get("fell_over", 0) / count
                if count and "fell_over" in self.end_term_names else None,
                "unfinished": [{"env_index": i, "steps": self.lengths[i].item(),
                                "partial_return": self.returns[i].item()} for i in range(self.n)],
                "unfinished_count": (self.lengths > 0).sum().item(),
                "unfinished_mean_steps": self.lengths.double().mean().item(),
                "unfinished_mean_partial_return": self.returns.mean().item(),
                "reward_terms_available": bool(self.term_names),
                "completed_reward_terms_mean": {
                    name: sum(e["reward_terms"][name] for e in self.completed) / count if count else None
                    for name in self.term_names},
                "rollout_reward_terms": {name: {
                    "mean_step_contribution": self.term_sum[i].item() / self.sample_count if self.sample_count else None,
                    "weight_min": self.weights.get(name, (None, None))[0],
                    "weight_max": self.weights.get(name, (None, None))[1]}
                    for i, name in enumerate(self.term_names)}, "records": list(self.completed)}


def optimization_metrics(report, batch, targets, optimizer):
    returns, values = targets["returns"].double(), batch["values"].double()
    variance = returns.var(correction=0).item()
    actions, advantages = batch["actions"].double(), targets["advantages"].double()
    return {**report["after"], "pre_update_explained_variance":
            1 - (returns - values).var(correction=0).item() / variance if variance > 1e-12 else None,
            "raw_advantage_mean": advantages.mean().item(), "raw_advantage_std": advantages.std(correction=0).item(),
            "action_mean": actions.mean().item(), "action_std": actions.std(correction=0).item(),
            "action_max_abs": actions.abs().max().item(),
            "grad_norm_before": report["total_grad_norm_before"], "grad_norm_after": report["total_grad_norm_after"],
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}
