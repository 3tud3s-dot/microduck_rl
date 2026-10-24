"""Paired headless episodes using actor means, fresh seeded envs and frozen stats."""

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from training.rollout import _check_policy_result, load_policy
from training.train import assert_same_state, cpu_snapshot
from training.paths import INITIAL_CHECKPOINT, LATEST_CHECKPOINT


def _check_observations(observations, normalizer, device):
    if set(observations) != set(normalizer.stats):
        raise ValueError("observation groups must match the normalizer")
    for name, group in normalizer.stats.items():
        _check_policy_result(f"observations/{name}", observations[name], (1, group.dim), device)


@torch.no_grad()
def evaluate_episode(env, actor, critic, normalizer, *, seed, max_steps=128,
                     capture_conditions=None, expected_conditions=None):
    """Reset once, then stop at the first done or at an explicit evaluation cutoff.

    Returns are sums of original environment rewards (already dt-scaled). Duration
    is simulated time. Reaching max_steps does NOT invent a truncated flag or a
    completed episode. Critic is only checked for immutability; actions use actor
    forward() means, never sample(). No optimizer or GAE is used.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if env.num_envs != 1 or env.cfg.auto_reset:
        raise ValueError("evaluation requires one env with auto_reset=False")
    if not math.isfinite(env.step_dt) or env.step_dt <= 0:
        raise ValueError("step_dt must be finite and positive")
    if actor.training or critic.training:
        raise ValueError("actor and critic must be in eval mode")
    if normalizer.training or any(group.training for group in normalizer.stats.values()):
        raise ValueError("normalizer must already be frozen")
    policy_before = {"actor": cpu_snapshot(actor.state_dict()),
                     "critic": cpu_snapshot(critic.state_dict()), "normalizer": normalizer.checkpoint()}
    observations, _ = env.reset(seed=seed)
    _check_observations(observations, normalizer, env.device)
    initial_conditions = {"observations": cpu_snapshot(observations)}
    if capture_conditions is not None:
        initial_conditions["physics"] = cpu_snapshot(capture_conditions(env))
    if expected_conditions is not None:
        assert_same_state(initial_conditions, expected_conditions)
    records, episode_return = [], 0.0
    for _ in range(max_steps):
        raw = {name: x.detach().clone() for name, x in observations.items()}
        normalized = normalizer(observations)
        action = actor(normalized["actor"])  # Mean of the Gaussian; no sampling or clipping.
        _check_policy_result("mean actions", action, (1, env.action_manager.total_action_dim), env.device)
        record = {"observations": raw, "actions": action.detach().clone()}
        observations, reward, terminated, truncated, _ = env.step(action)
        _check_observations(observations, normalizer, env.device)
        _check_policy_result("reward", reward, (1,), env.device)
        for name, flag in (("terminated", terminated), ("truncated", truncated)):
            if (not isinstance(flag, torch.Tensor) or flag.shape != (1,)
                    or flag.dtype != torch.bool or flag.device != torch.device(env.device)):
                raise ValueError(f"{name} must be a boolean [1] tensor on the env device")
        if "nan_state" in env.termination_manager.active_terms:
            if env.termination_manager.get_term("nan_state").any().item():
                raise RuntimeError("nan_state termination during evaluation")
        record.update({"next_observations": {name: x.detach().clone() for name, x in observations.items()},
                       "rewards": reward.detach().clone(), "terminated": terminated.detach().clone(),
                       "truncated": truncated.detach().clone()})
        records.append(record)
        episode_return += reward.item()
        if terminated.item() or truncated.item():
            break
    final = records[-1]
    terminated, truncated = final["terminated"].item(), final["truncated"].item()
    completed = terminated or truncated
    end_terms = [name for name in env.termination_manager.active_terms
                 if env.termination_manager.get_term(name).any().item()]
    assert_same_state(actor.state_dict(), policy_before["actor"])
    assert_same_state(critic.state_dict(), policy_before["critic"])
    assert_same_state(normalizer.checkpoint(), policy_before["normalizer"])
    if actor.training or critic.training:
        raise RuntimeError("model mode changed during evaluation")
    trace = {name: torch.stack([record[name] for record in records])
             for name in ("actions", "rewards", "terminated", "truncated")}
    for field in ("observations", "next_observations"):
        trace[field] = {name: torch.stack([record[field][name] for record in records])
                        for name in normalizer.stats}
    trace["initial_conditions"] = initial_conditions
    result = {"seed": seed, "steps": len(records), "duration_s": len(records) * env.step_dt,
              "episode_return": episode_return, "terminated": terminated, "truncated": truncated,
              "completed": completed, "evaluation_cutoff": not completed,
              "active_end_terms": end_terms, "policy_unchanged": True, "normalizer_frozen": True,
              "normalizer_counts": {name: group.count.item() for name, group in normalizer.stats.items()}}
    return result, cpu_snapshot(trace)


def compare_checkpoints(initial_path, trained_path, make_env, *, seeds=(42, 43), max_steps=128,
                        device="cpu", capture_conditions=None, on_episode=None):
    """Use a fresh, same-seed environment for EACH checkpoint/seed pair.

    make_env(seed) must set the seed before startup events and create the same cfg
    every time. Model loading happens before env construction so random weight
    initialization cannot perturb the environment's seeded random stream.
    """
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide nonempty, distinct episode seeds")
    results, traces, comparisons = {"initial": [], "trained": []}, {"initial": [], "trained": []}, []
    for seed in seeds:
        expected = None
        for label, path in (("initial", initial_path), ("trained", trained_path)):
            actor, critic, normalizer = load_policy(Path(path), device)
            env = make_env(seed)
            try:
                result, trace = evaluate_episode(env, actor, critic, normalizer, seed=seed,
                                                  max_steps=max_steps, capture_conditions=capture_conditions,
                                                  expected_conditions=expected)
            finally:
                env.close()
                del env
                gc.collect()
            if label == "initial":
                expected = trace["initial_conditions"]
            result["initial_conditions_matched"] = True
            results[label].append(result)
            traces[label].append(trace)
            if on_episode is not None:
                on_episode(label, result)
        initial, trained = results["initial"][-1], results["trained"][-1]
        comparisons.append({"seed": seed, "return_delta_trained_minus_initial":
                            trained["episode_return"] - initial["episode_return"],
                            "duration_delta_s": trained["duration_s"] - initial["duration_s"],
                            "both_completed": initial["completed"] and trained["completed"],
                            "initial_conditions_matched": True})
    aggregates = {}
    for label, episodes in results.items():
        completed = [episode for episode in episodes if episode["completed"]]
        aggregates[label] = {"trials": len(episodes), "completed_episodes": len(completed),
                             "evaluation_cutoffs": sum(x["evaluation_cutoff"] for x in episodes),
                             "mean_recorded_return": sum(x["episode_return"] for x in episodes) / len(episodes),
                             "mean_recorded_duration_s": sum(x["duration_s"] for x in episodes) / len(episodes),
                             "mean_completed_episode_return":
                             sum(x["episode_return"] for x in completed) / len(completed) if completed else None}
    return {"episodes": results, "paired_comparisons": comparisons, "aggregates": aggregates}, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--trained", type=Path, default=LATEST_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")
    if args.seed < 0 or args.seed + args.episodes > 2**32:
        parser.error("episode seeds must fit in [0, 2**32)")
    if not args.initial.is_file() or not args.trained.is_file():
        parser.error("both checkpoint paths must exist")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU simulation fallback")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from bam.mjlab import BamActuator
    from training.inspect_env import ACTION_DIM, TASK_ID

    configure_torch_backends(allow_tf32=False)
    template = load_env_cfg(TASK_ID, play=False)
    template.scene.num_envs = 1
    template.auto_reset = False

    def make_env(seed):
        cfg = copy.deepcopy(template)
        cfg.seed = seed
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
        try:
            actuators = env.scene["robot"].actuators
            if not actuators or any(not isinstance(a, BamActuator) or
                                    (a.cfg.motor_name, a.cfg.model) != ("xl330", "m6") for a in actuators):
                raise RuntimeError("expected original XL330 M6 BAM actuators")
            if env.action_manager.total_action_dim != ACTION_DIM or not math.isclose(env.step_dt, .02):
                raise RuntimeError("expected 14 actions at 50 Hz")
        except Exception:
            env.close()
            raise
        return env

    def capture_conditions(env):
        data = env.scene["robot"].data
        return {name: getattr(data, name) for name in (
            "root_link_pose_w", "root_com_lin_vel_w", "root_com_ang_vel_w", "joint_pos", "joint_vel")}

    def log_episode(label, result):
        print(f"[evaluate] {label} seed={result['seed']} return={result['episode_return']:.6f} "
              f"steps={result['steps']} duration_s={result['duration_s']:.3f} "
              f"terminated={result['terminated']} truncated={result['truncated']} "
              f"cutoff={result['evaluation_cutoff']} terms={result['active_end_terms']} "
              f"normalizer_counts={result['normalizer_counts']} frozen=PASS unchanged=PASS", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    seeds = tuple(range(args.seed, args.seed + args.episodes))
    print(f"[evaluate] task={TASK_ID} seeds={seeds} max_steps={args.max_steps} "
          "actor=mean render=None cfg=train fresh_env_per_trial=True", flush=True)
    report, traces = compare_checkpoints(args.initial, args.trained, make_env, seeds=seeds,
                                         max_steps=args.max_steps, device="cuda:0",
                                         capture_conditions=capture_conditions, on_episode=log_episode)
    report["metadata"] = {"task_id": TASK_ID, "env_config": "original train cfg (play=False)",
                          "num_envs": 1, "step_dt": .02, "seeds": list(seeds), "max_steps": args.max_steps,
                          "action_mode": "actor mean; original HOME offsets; no clipping/filtering",
                          "render_mode": None, "auto_reset": False, "allow_tf32": False,
                          "reward_units": "sum of original dt-scaled env rewards; no extra scaling",
                          "duration_units": "simulated seconds", "fresh_env_per_trial": True,
                          "initial_observations_and_physics_matched": True,
                          "curricula": "fresh env counters per trial; no checkpoint training counter restore",
                          "checkpoints": {label: {"path": str(path.resolve()),
                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                          for label, path in (("initial", args.initial), ("trained", args.trained))}}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save({"metadata": report["metadata"], "traces": traces}, args.output_dir / "trajectories.pt")
    print(f"[evaluate] aggregates={report['aggregates']}")
    print(f"[evaluate] saved={args.output_dir.resolve()}")
    print("[evaluate] PASS: paired initial conditions, mean actions, finite traces, frozen stats/weights")
    print("[evaluate] This short comparison is not evidence of a reliable walking policy.")


if __name__ == "__main__":
    main()
