"""Collect [time, env, ...] snapshots with zero actions or a frozen actor/critic."""

import argparse
from pathlib import Path
from typing import Callable

import torch

from training.vector_reset import reset_done_envs


def _check_policy_result(name, tensor, shape, device):
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
            or not tensor.is_floating_point() or tensor.device != torch.device(device)):
        raise ValueError(f"{name}: expected floating tensor {shape} on {device}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name}: contains NaN or Inf")


@torch.no_grad()
def collect_rollout(env, steps: int, action_fn: Callable | None = None,
                    *, actor=None, critic=None, normalizer=None, bootstrap=False,
                    initial_observations=None, on_transition=None) -> dict:
    """Collect steps, resetting initially unless continuation observations are given.

    Supplying actor, critic and a frozen normalizer records normalized inputs,
    log_prob(a_t | obs_t) and V(obs_t) BEFORE executing that same action a_t.
    next_observations are the values returned by step(): with auto-reset, done
    rows contain reset observations, NOT the previous episode's terminal state.
    Rewards and actions retain the original environment's units and scaling.

    bootstrap=True requires a policy and auto_reset=False. Capture the true
    transition_next_observations and bootstrap_values BEFORE resetting done envs.
    next_observations then holds the inputs for the next policy decision.
    Partial resets preserve continuing rows' observation/motor delay and history
    through the training-local mjlab reset adapter. Output axes stay [time, env].
    To continue without a reset at the rollout horizon, pass the previous batch's
    next_observations[-1] as initial_observations; the caller must keep the same env.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    policy_mode = any(module is not None for module in (actor, critic, normalizer))
    if bootstrap and (not policy_mode or env.cfg.auto_reset):
        raise ValueError("bootstrap requires a policy and auto_reset=False")
    if not bootstrap and not env.cfg.auto_reset:
        raise ValueError("This collector requires auto_reset=True unless bootstrap=True")
    if policy_mode:
        if any(module is None for module in (actor, critic, normalizer)):
            raise ValueError("provide actor, critic and normalizer together")
        if action_fn is not None:
            raise ValueError("action_fn and actor/critic collection are mutually exclusive")
        if normalizer.training or any(stats.training for stats in normalizer.stats.values()):
            raise ValueError("normalizer must be frozen with eval()")
    if initial_observations is None:
        observations, _ = env.reset()
    else:
        observations = {name: value.detach().clone()
                        for name, value in initial_observations.items()}
    groups = tuple(observations)
    records = []
    for _ in range(steps):
        record = {"observations": {
            name: value.detach().clone() for name, value in observations.items()
        }}
        if policy_mode:
            normalized = normalizer(observations)  # Exactly once, using frozen statistics.
            record["normalized_observations"] = {
                name: value.detach().clone() for name, value in normalized.items()
            }
            action, log_prob = actor.sample(normalized["actor"])
            value = critic(normalized["critic"])
            for name, tensor in (("actions", action), ("log_prob", log_prob), ("values", value)):
                shape = ((env.num_envs, env.action_manager.total_action_dim)
                         if name == "actions" else (env.num_envs,))
                _check_policy_result(name, tensor, shape, env.device)
            record["log_prob"] = log_prob.detach().clone()
            record["values"] = value.detach().clone()
        else:
            action = (action_fn(observations) if action_fn is not None else
                      torch.zeros((env.num_envs, env.action_manager.total_action_dim),
                                  device=env.device))
        record["actions"] = action.detach().clone()  # Before step can touch it.
        observations, reward, terminated, truncated, _ = env.step(action)
        for name, value in (("rewards", reward), ("terminated", terminated),
                            ("truncated", truncated)):
            record[name] = value.detach().clone()  # Flags are reused by mjlab.
        if "nan_state" in env.termination_manager.active_terms:
            if env.termination_manager.get_term("nan_state").any().item():
                raise RuntimeError("nan_state termination during collection")
        if on_transition is not None:
            on_transition(env, record)  # Read metrics before any reset clears managers.
        if bootstrap:
            # Clone flags/rewards first: reset() may overwrite all shared buffers.
            terminal_next = {name: x.detach().clone() for name, x in observations.items()}
            next_value = critic(normalizer.stats["critic"](terminal_next["critic"]))
            _check_policy_result("bootstrap_values", next_value, (env.num_envs,), env.device)
            record["bootstrap_values"] = torch.where(
                record["terminated"], torch.zeros_like(next_value), next_value).detach().clone()
            record["transition_next_observations"] = terminal_next
            done_ids = (record["terminated"] | record["truncated"]).nonzero().flatten()
            if done_ids.numel():
                observations = reset_done_envs(env, done_ids, terminal_next)
            else:
                observations = terminal_next
        record["next_observations"] = {
            name: value.detach().clone() for name, value in observations.items()
        }
        records.append(record)

    tensor_fields = ("actions", "rewards", "terminated", "truncated")
    observation_fields = ("observations", "next_observations")
    if policy_mode:
        tensor_fields += ("log_prob", "values")
        observation_fields += ("normalized_observations",)
    if bootstrap:
        tensor_fields += ("bootstrap_values",)
        observation_fields += ("transition_next_observations",)
    batch = {name: torch.stack([record[name] for record in records]) for name in tensor_fields}
    for field in observation_fields:
        batch[field] = {name: torch.stack([record[field][name] for record in records])
                        for name in groups}
    return batch


def load_policy(path: Path, device):
    from training.models import Critic, GaussianActor
    from training.normalization import ObservationNormalizer

    state = torch.load(path, map_location="cpu", weights_only=True)
    actor = GaussianActor(**state["actor_config"]).to(device)
    critic = Critic(**state["critic_config"]).to(device)
    actor.load_state_dict(state["actor_state_dict"], strict=True)
    critic.load_state_dict(state["critic_state_dict"], strict=True)
    normalizer = ObservationNormalizer.from_checkpoint(state["normalizer"]).to(device)
    return actor.eval(), critic.eval(), normalizer.eval()


@torch.no_grad()
def verify_policy_rollout(batch, actor, critic, normalizer):
    # Recompute one step at a time, matching the sampling batch size on CUDA.
    # A large batched GEMM can otherwise use different TF32 kernels/rounding.
    for step in range(batch["actions"].shape[0]):
        raw = {name: x[step] for name, x in batch["observations"].items()}
        normalized = normalizer(raw)
        for name, x in normalized.items():
            torch.testing.assert_close(x, batch["normalized_observations"][name][step],
                                       rtol=0, atol=0)
        log_prob, _ = actor.evaluate_actions(normalized["actor"], batch["actions"][step])
        torch.testing.assert_close(log_prob, batch["log_prob"][step], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(critic(normalized["critic"]), batch["values"][step],
                                   rtol=1e-5, atol=1e-6)
        if "bootstrap_values" in batch:
            next_critic = batch["transition_next_observations"]["critic"][step]
            next_value = critic(normalizer.stats["critic"](next_critic))
            expected = torch.where(batch["terminated"][step], 0.0, next_value)
            torch.testing.assert_close(expected, batch["bootstrap_values"][step],
                                       rtol=1e-5, atol=1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    from training.paths import INITIAL_CHECKPOINT
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--agent", choices=("zero", "policy"), default="zero")
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy_mode = args.agent == "policy"
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("--output must not overwrite the model checkpoint")
    if args.output.exists():
        parser.error("--output already exists; choose a new file")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from bam.mjlab import BamActuator
    from training.inspect_env import ACTION_DIM, OBS_DIMS, TASK_ID, check_tensor

    configure_torch_backends(allow_tf32=False)
    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if policy_mode:
        cfg.auto_reset = False  # Public manual-reset API preserves terminal observations.
    print(f"[rollout] task={TASK_ID} mode=train envs={args.num_envs} device=cuda:0 "
          f"seed={args.seed} steps={args.steps} agent={args.agent}", flush=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        actuators = env.scene["robot"].actuators
        if not actuators or not all(isinstance(a, BamActuator) for a in actuators):
            raise RuntimeError("Expected original BAM actuators")
        if any((a.cfg.motor_name, a.cfg.model) != ("xl330", "m6") for a in actuators):
            raise RuntimeError("Expected XL330 M6")
        print(f"[rollout] BAM=XL330 M6 step_dt={env.step_dt} auto_reset={cfg.auto_reset}")
        modules, before = {}, {}
        if policy_mode:
            actor, critic, normalizer = load_policy(args.checkpoint, env.device)
            modules = {"actor": actor, "critic": critic, "normalizer": normalizer}
            before = {name: {key: value.clone() for key, value in module.state_dict().items()}
                      for name, module in modules.items()}
            batch = collect_rollout(env, args.steps, actor=actor, critic=critic,
                                    normalizer=normalizer, bootstrap=True)
            verify_policy_rollout(batch, actor, critic, normalizer)
            for name, module in modules.items():
                for key, value in module.state_dict().items():
                    if not torch.equal(value, before[name][key]):
                        raise RuntimeError(f"collection changed {name}/{key}")
            counts = {name: stats.count.item() for name, stats in normalizer.stats.items()}
            print(f"[rollout] same_action_log_prob=PASS pre_action_value=PASS "
                  f"normalizer_counts={counts} frozen=PASS model_parameters=unchanged")
        else:
            batch = collect_rollout(env, args.steps)
        observation_fields = ("observations", "next_observations")
        if policy_mode:
            observation_fields += ("normalized_observations", "transition_next_observations")
        for field in observation_fields:
            for name, width in OBS_DIMS.items():
                check_tensor(f"{field}/{name}", batch[field][name],
                             (args.steps, args.num_envs, width))
                print(f"[rollout] {field}/{name}={tuple(batch[field][name].shape)}")
        tensor_fields = ("actions", "rewards", "terminated", "truncated")
        if policy_mode:
            tensor_fields += ("log_prob", "values", "bootstrap_values")
        for field in tensor_fields:
            shape = ((args.steps, args.num_envs, ACTION_DIM) if field == "actions"
                     else (args.steps, args.num_envs))
            check_tensor(field, batch[field], shape)
            print(f"[rollout] {field}={tuple(batch[field].shape)}")
        for field in ("terminated", "truncated"):
            if batch[field].dtype != torch.bool:
                raise RuntimeError(f"{field} must stay boolean")
        if not policy_mode and torch.count_nonzero(batch["actions"]).item():
            raise RuntimeError("Expected only zero actions")
        for name in OBS_DIMS:
            if not torch.equal(batch["observations"][name][1:],
                               batch["next_observations"][name][:-1]):
                raise RuntimeError(f"{name}: adjacent transition observations do not align")

        metadata = {"task_id": TASK_ID, "mode": "train", "seed": cfg.seed,
                    "num_envs": args.num_envs, "steps": args.steps, "step_dt": env.step_dt,
                    "auto_reset": cfg.auto_reset, "agent": args.agent,
                    "actions": "sampled HOME offsets" if policy_mode else "zero HOME offsets",
                    "next_observations": "next policy input; done rows are reset observations"}
        cpu_batch = {key: ({name: tensor.cpu() for name, tensor in value.items()}
                           if isinstance(value, dict) else value.cpu())
                     for key, value in batch.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": metadata, "rollout": cpu_batch}
        if policy_mode:
            metadata.update({"source_checkpoint": str(args.checkpoint),
                             "normalization_frozen": True,
                             "log_prob": "log pi(a_t | normalized actor obs_t)",
                             "values": "V(normalized critic obs_t), before env.step",
                             "transition_next_observations": "true step return before reset",
                             "bootstrap_values": "zero if terminated, otherwise V(true next critic obs)"})
            payload["policy"] = {"actor_config": actor.config, "critic_config": critic.config,
                                 "actor_state_dict": {k: v.detach().cpu().clone()
                                                      for k, v in actor.state_dict().items()},
                                 "critic_state_dict": {k: v.detach().cpu().clone()
                                                       for k, v in critic.state_dict().items()},
                                 "normalizer": normalizer.checkpoint()}
        torch.save(payload, args.output)
        term_count = batch["terminated"].sum().item()
        trunc_count = batch["truncated"].sum().item()
        ended = batch["terminated"] | batch["truncated"]
        print(f"[rollout] terminated={term_count} truncated={trunc_count} "
              f"ended={ended.sum().item()} simulated={args.steps * env.step_dt:.3f}s")
        if ended.any().item():
            print(f"[rollout] end indices [time, env]={ended.nonzero().cpu().tolist()}")
        print(f"[rollout] saved={args.output}")
        print(f"[rollout] action_range=({batch['actions'].min().item():.5f}, "
              f"{batch['actions'].max().item():.5f})")
        print(f"[rollout] PASS: {args.agent} actions, shapes, finite values and alignment checked")
        print("[rollout] The collection horizon does not mark an episode as ended.")
        if policy_mode:
            print("[rollout] Actions were executed in simulation; no policy/value parameter updates.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
