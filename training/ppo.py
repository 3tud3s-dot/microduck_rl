"""PPO epochs and shuffled minibatches with fixed rollout targets and persistent Adam."""

import argparse
import hashlib
import math
from pathlib import Path

import torch

from training.advantages import compute_gae
from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer


def ppo_loss(log_prob, old_log_prob, values, returns, advantages, entropy,
             *, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
    """Flat [sample] tensors, clipped policy objective and unclipped value MSE.

    Old probabilities, advantages and returns are fixed targets. Entropy is the
    joint differential entropy over action dimensions and can be negative.
    """
    if not math.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
        raise ValueError("clip_ratio must be finite and in (0, 1)")
    for name, coefficient in (("value_coef", value_coef), ("entropy_coef", entropy_coef)):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(log_prob, torch.Tensor) or log_prob.ndim != 1 or not log_prob.numel():
        raise ValueError("log_prob must be a nonempty floating [sample] tensor")
    for name, x in (("log_prob", log_prob), ("old_log_prob", old_log_prob),
                    ("values", values), ("returns", returns),
                    ("advantages", advantages), ("entropy", entropy)):
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point()
                or x.shape != log_prob.shape or x.dtype != log_prob.dtype
                or x.device != log_prob.device):
            raise ValueError(f"{name}: shape/device/dtype must match [sample]; no broadcasting")
        if not torch.isfinite(x).all().item():
            raise ValueError(f"{name}: contains NaN or Inf")
    log_ratio = log_prob - old_log_prob.detach()
    ratio = log_ratio.exp()
    fixed_advantages = advantages.detach()
    surrogate = ratio * fixed_advantages
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * fixed_advantages
    policy_loss = -torch.minimum(surrogate, clipped).mean()
    value_loss = (values - returns.detach()).square().mean()
    mean_entropy = entropy.mean()
    result = {"loss": policy_loss + value_coef * value_loss - entropy_coef * mean_entropy,
              "policy_loss": policy_loss, "value_loss": value_loss, "entropy": mean_entropy,
              "approx_kl": ((ratio - 1) - log_ratio).mean(),
              "clip_fraction": ((ratio - 1).abs() > clip_ratio).float().mean(),
              "ratio_min": ratio.min(), "ratio_max": ratio.max()}
    if not all(torch.isfinite(x).all().item() for x in result.values()):
        raise ValueError("PPO loss or probability ratio overflowed")
    return result


def _norm(tensors):
    return torch.stack([x.detach().double().square().sum() for x in tensors]).sum().sqrt().item()


def create_optimizer(actor, critic, learning_rate=1e-5):
    """Create once outside the training loop; reuse for every newly collected batch."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=learning_rate)


def optimizer_status(optimizer):
    """Check all Adam states and report their shared cumulative step and moment norms."""
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("expected an Adam optimizer")
    steps, first, second = [], [], []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if not state:
                steps.append(0)
                continue
            step = state["step"].item()
            if not math.isfinite(step) or step < 0 or not step.is_integer():
                raise RuntimeError("Adam step must be a finite nonnegative integer")
            steps.append(int(step))
            for key, moments in (("exp_avg", first), ("exp_avg_sq", second)):
                value = state[key]
                if value.shape != parameter.shape or not torch.isfinite(value).all().item():
                    raise RuntimeError(f"invalid Adam {key}")
                if key == "exp_avg_sq" and (value < 0).any().item():
                    raise RuntimeError("negative Adam second moment")
                moments.append(value)
    if not steps or len(set(steps)) != 1:
        raise RuntimeError("all optimized parameters must have the same Adam step")
    return {"step": steps[0], "exp_avg_norm": _norm(first) if first else 0.0,
            "exp_avg_sq_norm": _norm(second) if second else 0.0}


def update_ppo(actor, critic, normalizer, batch, targets, *, optimizer=None, learning_rate=None,
               epochs=3, minibatch_size=32, shuffle=True, generator=None,
               clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
    """Reuse one rollout across epochs; each minibatch performs one Adam step.

    Normalize advantages across the full batch (population std + 1e-8), only for
    the policy loss, ONCE before all epochs. Clone old log_prob, actions, advantages
    and returns; never refresh them from the changing network or recompute GAE.
    Check the sampling policy once before the first update. Every epoch visits all
    T*N samples once, including a short final minibatch. generator, if supplied,
    must match the data device. minibatch_size=None means the full batch.
    Pass the same optimizer on subsequent calls to retain Adam moments and steps.
    With a supplied optimizer, its param_groups own the learning rate.
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if minibatch_size is not None and (not isinstance(minibatch_size, int)
            or isinstance(minibatch_size, bool) or minibatch_size < 1):
        raise ValueError("minibatch_size must be a positive integer or None")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if optimizer is not None and learning_rate is not None:
        raise ValueError("set learning_rate on the supplied optimizer's param_groups")
    if normalizer.training or any(group.training for group in normalizer.stats.values()):
        raise ValueError("normalizer must already be frozen with eval()")
    leading = batch["log_prob"].shape
    if len(leading) != 2 or 0 in leading:
        raise ValueError("rollout must have nonempty [time, env] sample axes")
    for name, x in (("values", batch["values"]), ("advantages", targets["advantages"]),
                    ("returns", targets["returns"])):
        if x.shape != leading:
            raise ValueError(f"{name}: expected [time, env] matching log_prob")
    for name, x in batch["observations"].items():
        if x.ndim != 3 or x.shape[:2] != leading:
            raise ValueError(f"observations/{name}: expected [time, env, feature]")
    if batch["actions"].shape != (*leading, actor.config["action_dim"]):
        raise ValueError("actions must match [time, env, action_dim]")
    frozen = normalizer.checkpoint()
    with torch.no_grad():
        normalized = normalizer(batch["observations"])
        for name, x in normalized.items():
            torch.testing.assert_close(x, batch["normalized_observations"][name],
                                       rtol=1e-6, atol=1e-6)
    obs = {name: x.flatten(0, 1).detach() for name, x in normalized.items()}
    actions = batch["actions"].flatten(0, 1).detach().clone()
    old_log_prob = batch["log_prob"].flatten().detach().clone()
    old_values = batch["values"].flatten().detach().clone()
    returns = targets["returns"].flatten().detach().clone()
    raw_advantages = targets["advantages"].flatten().detach().clone()
    advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std(correction=0) + 1e-8)
    sample_count = actions.shape[0]
    batch_size = sample_count if minibatch_size is None else min(minibatch_size, sample_count)

    def check_fixed_targets():
        for live, fixed in ((batch["actions"].flatten(0, 1), actions),
                            (batch["log_prob"].flatten(), old_log_prob),
                            (batch["values"].flatten(), old_values),
                            (targets["returns"].flatten(), returns),
                            (targets["advantages"].flatten(), raw_advantages)):
            if not torch.equal(live, fixed):
                raise RuntimeError("rollout actions, old log_prob or GAE targets changed")
    groups = {"actor": list(actor.parameters()), "critic": list(critic.parameters())}
    snapshots = {name: [p.detach().clone() for p in parameters]
                 for name, parameters in groups.items()}
    parameters = groups["actor"] + groups["critic"]
    if optimizer is None:
        optimizer = create_optimizer(actor, critic, 1e-5 if learning_rate is None else learning_rate)
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if sorted(map(id, optimized)) != sorted(map(id, parameters)):
        raise ValueError("optimizer must contain exactly this actor and critic's parameters")
    optimizer_before = optimizer_status(optimizer)
    actor.train()
    critic.train()

    def evaluate(indices=None):
        def select(x):
            return x if indices is None else x[indices]

        log_prob, entropy = actor.evaluate_actions(select(obs["actor"]), select(actions))
        values = critic(select(obs["critic"]))
        losses = ppo_loss(log_prob, select(old_log_prob), values, select(returns), select(advantages), entropy,
                          clip_ratio=clip_ratio, value_coef=value_coef, entropy_coef=entropy_coef)
        return losses, log_prob, values

    with torch.no_grad():
        losses, log_prob, values = evaluate()
        # Compare only at the start: later epochs are expected to change density/value.
        torch.testing.assert_close(log_prob, old_log_prob, rtol=0, atol=1e-4)
        torch.testing.assert_close(values, old_values, rtol=1e-5, atol=1e-6)
        before = {name: x.item() for name, x in losses.items()}
    epoch_reports, minibatch_reports = [], []
    gradients = {name: {"before_clip_norm": 0.0, "after_clip_norm": 0.0} for name in groups}
    for epoch in range(1, epochs + 1):
        order = (torch.randperm(sample_count, device=actions.device, generator=generator)
                 if shuffle else torch.arange(sample_count, device=actions.device))
        seen = torch.zeros(sample_count, dtype=torch.long, device=actions.device)
        sizes, weighted_loss = [], 0.0
        for start in range(0, sample_count, batch_size):
            indices = order[start:start + batch_size]
            seen.index_add_(0, indices, torch.ones_like(indices))
            optimizer.zero_grad(set_to_none=True)
            losses, _, _ = evaluate(indices)
            loss_value = losses["loss"].detach().item()
            losses["loss"].backward()
            current_gradients = {}
            for name, params in groups.items():
                if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in params):
                    raise RuntimeError(f"{name}: missing or non-finite gradient; optimizer did not step")
                current_gradients[name] = {"before_clip_norm": _norm([p.grad for p in params])}
            raw_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
            for name, params in groups.items():
                current_gradients[name]["after_clip_norm"] = _norm([p.grad for p in params])
                for key, value in current_gradients[name].items():
                    gradients[name][key] = max(gradients[name][key], value)
            clipped_norm = _norm([p.grad for p in parameters])
            if clipped_norm > max_grad_norm * (1 + 1e-5):
                raise RuntimeError("gradient clipping exceeded max_grad_norm")
            optimizer.step()  # One step per minibatch; moments persist across epochs.
            optimizer_after = optimizer_status(optimizer)
            if optimizer_after["step"] != optimizer_before["step"] + len(minibatch_reports) + 1:
                raise RuntimeError("Adam must advance once per minibatch")
            if any(not torch.isfinite(p).all().item() for p in parameters):
                raise RuntimeError("non-finite parameter after minibatch update")
            sizes.append(indices.numel())
            weighted_loss += loss_value * indices.numel()
            minibatch_reports.append({"epoch": epoch, "samples": indices.numel(),
                                      "optimizer_step": optimizer_after["step"], "loss": loss_value,
                                      "total_grad_norm_before": raw_norm.item(),
                                      "total_grad_norm_after": clipped_norm,
                                      "gradients": current_gradients})
        if not torch.equal(seen, torch.ones_like(seen)):
            raise RuntimeError("every epoch must visit each sample exactly once")
        check_fixed_targets()
        with torch.no_grad():
            epoch_after = {name: x.item() for name, x in evaluate()[0].items()}
        epoch_reports.append({"epoch": epoch, "minibatch_sizes": sizes,
                              "sample_order": order.cpu().tolist(), "coverage_verified": True,
                              "samples_seen": sum(sizes), "weighted_mean_loss": weighted_loss / sample_count,
                              "after": epoch_after})

    changes = {}
    for name, params in groups.items():
        if any(not torch.isfinite(p).all().item() for p in params):
            raise RuntimeError(f"{name}: non-finite parameter after update")
        deltas = [p.detach().double() - old.double() for p, old in zip(params, snapshots[name])]
        changes[name] = {"l2": _norm(deltas), "max_abs": max(x.abs().max().item() for x in deltas),
                         "changed_elements": sum(torch.count_nonzero(x).item() for x in deltas)}
    after = epoch_reports[-1]["after"]
    restored = normalizer.checkpoint()
    for key in ("dims", "eps", "training", "group_training"):
        if restored[key] != frozen[key]:
            raise RuntimeError(f"normalizer {key} changed")
    for key, value in frozen["state_dict"].items():
        if not torch.equal(value, restored["state_dict"][key]):
            raise RuntimeError(f"normalizer buffer {key} changed")
    report = {"before": before, "after": after, "gradients": gradients,
              "total_grad_norm_before": max(x["total_grad_norm_before"] for x in minibatch_reports),
              "total_grad_norm_after": max(x["total_grad_norm_after"] for x in minibatch_reports),
              "gradient_summary": "maximum per minibatch",
              "parameter_changes": changes, "samples": sample_count,
              "optimizer_steps": len(minibatch_reports), "epochs": epochs, "minibatch_size": batch_size,
              "epoch_reports": epoch_reports, "minibatch_reports": minibatch_reports,
              "fixed_targets_verified": True,
              "optimizer_before": optimizer_before, "optimizer_after": optimizer_after,
              "normalizer_frozen": True,
              "normalizer_counts": {name: group.count.item() for name, group in normalizer.stats.items()}}
    return report, optimizer.state_dict()


def update_once(actor, critic, normalizer, batch, targets, *, optimizer=None, learning_rate=None,
                clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
    """Backward-compatible single full-batch update without shuffling."""
    return update_ppo(actor, critic, normalizer, batch, targets,
                      optimizer=optimizer, learning_rate=learning_rate,
                      epochs=1, minibatch_size=None, shuffle=False,
                      clip_ratio=clip_ratio, value_coef=value_coef,
                      entropy_coef=entropy_coef, max_grad_norm=max_grad_norm)


def load_inputs(rollout_path, advantages_path):
    """Reject stale/mismatched GAE before creating an optimizer or changing models."""
    source = torch.load(rollout_path, map_location="cpu", weights_only=True)
    targets = torch.load(advantages_path, map_location="cpu", weights_only=True)
    metadata = targets["metadata"]
    if Path(metadata["source_rollout"]).resolve() != Path(rollout_path).resolve():
        raise ValueError("GAE source_rollout does not match this rollout")
    if metadata["advantage_normalization"]:
        raise ValueError("expected original, unnormalized GAE targets")
    batch = source["rollout"]
    reference = compute_gae(*(batch[k] for k in (
        "rewards", "values", "bootstrap_values", "terminated", "truncated")),
        gamma=metadata["gamma"], gae_lambda=metadata["gae_lambda"])
    for name, x in reference.items():
        torch.testing.assert_close(targets[name], x, rtol=0, atol=0)
    return source, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--advantages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {args.rollout.resolve(), args.advantages.resolve()}:
        parser.error("--output must be new and must not overwrite rollout or GAE inputs")
    source, targets = load_inputs(args.rollout, args.advantages)
    policy = source["policy"]  # Use the actual sampling weights, not another checkpoint.
    actor, critic = GaussianActor(**policy["actor_config"]), Critic(**policy["critic_config"])
    actor.load_state_dict(policy["actor_state_dict"], strict=True)
    critic.load_state_dict(policy["critic_state_dict"], strict=True)
    normalizer = ObservationNormalizer.from_checkpoint(policy["normalizer"])
    report, optimizer_state = update_once(actor, critic, normalizer, source["rollout"], targets,
                                           learning_rate=args.lr)
    for name in ("actor", "critic"):
        if (report["gradients"][name]["before_clip_norm"] <= 0
                or report["parameter_changes"][name]["changed_elements"] == 0):
            raise RuntimeError(f"{name}: expected a nonzero gradient and parameter change")
    actor.eval()
    critic.eval()
    payload = {"actor_config": actor.config, "critic_config": critic.config,
               "actor_state_dict": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
               "critic_state_dict": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
               "actor_training": False, "critic_training": False,
               "normalizer": normalizer.checkpoint(), "optimizer_state_dict": optimizer_state,
               "report": report,
               "metadata": {"device": "cpu", "learning_rate": args.lr,
                            "clip_ratio": 0.2, "value_coef": 0.5, "entropy_coef": 0.01,
                            "max_grad_norm": 0.5, "advantage_normalization": "population std + 1e-8",
                            "source_rollout": str(args.rollout.resolve()),
                            "source_advantages": str(args.advantages.resolve()),
                            "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in (args.rollout, args.advantages)}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    from training.rollout import load_policy
    restored_actor, restored_critic, restored_norm = load_policy(args.output, "cpu")
    with torch.no_grad():
        normalized = normalizer(source["rollout"]["observations"])
        recovered = restored_norm(source["rollout"]["observations"])
        for name in normalized:
            torch.testing.assert_close(recovered[name], normalized[name], rtol=0, atol=0)
        actions = source["rollout"]["actions"]
        torch.testing.assert_close(restored_actor.evaluate_actions(recovered["actor"], actions)[0],
                                   actor.evaluate_actions(normalized["actor"], actions)[0], rtol=0, atol=0)
        torch.testing.assert_close(restored_critic(recovered["critic"]), critic(normalized["critic"]),
                                   rtol=0, atol=0)
    for phase in ("before", "after"):
        print(f"[ppo] {phase}={report[phase]}")
    print(f"[ppo] gradients={report['gradients']} "
          f"total_norm={report['total_grad_norm_before']:.6f} -> {report['total_grad_norm_after']:.6f}")
    print(f"[ppo] parameter_changes={report['parameter_changes']}")
    print(f"[ppo] samples={report['samples']} optimizer_steps=1 lr={args.lr} "
          f"normalizer_counts={report['normalizer_counts']} frozen=PASS reload=PASS")
    print(f"[ppo] saved={args.output}")
    print("[ppo] PASS: one CPU update only; policy behavior has not been evaluated")


if __name__ == "__main__":
    main()
