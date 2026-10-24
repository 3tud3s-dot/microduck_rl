"""Precollect observation moments with a fixed policy; no PPO or optimizer.

The sampling policy ALWAYS uses its original frozen normalizer. A separate,
initially empty candidate fits raw policy inputs. Holdout observations never fit
the candidate. Save statistics separately; do not silently replace a model's norm.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from training.metrics import EpisodeTracker
from training.normalization import ObservationNormalizer, observation_diagnostics
from training.rollout import collect_rollout, load_policy
from training.train import assert_same_state, cpu_snapshot


def _positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _seed(value):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
        raise ValueError("seeds must be integers in [0, 2**32)")


def moment_change(before, after):
    """Mean drift in old effective-std units; log std ratio is dimensionless."""
    result = {}
    for name in after["dims"]:
        old, new = before["state_dict"], after["state_dict"]
        prefix = f"stats.{name}."
        if old[prefix + "count"].item() == 0:
            result[name] = {"comparable": False, "mean_drift_rms": None, "log_std_drift_rms": None}
            continue
        old_std = old[prefix + "var"].sqrt() + before["eps"][name]
        new_std = new[prefix + "var"].sqrt() + after["eps"][name]
        mean_shift = (new[prefix + "mean"] - old[prefix + "mean"]) / old_std
        std_shift = (new_std / old_std).log()
        result[name] = {"comparable": True, "mean_drift_rms": mean_shift.square().mean().sqrt().item(),
                        "mean_drift_max_abs": mean_shift.abs().max().item(),
                        "log_std_drift_rms": std_shift.square().mean().sqrt().item(),
                        "log_std_drift_max_abs": std_shift.abs().max().item()}
    return result


def stability_summary(changes, *, stable_batches=3, mean_tolerance=.05, std_tolerance=.05):
    _positive(stable_batches, "stable_batches")
    for value in (mean_tolerance, std_tolerance):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("stability tolerances must be finite and positive")
    tail = changes[-stable_batches:]
    enough = len(tail) == stable_batches
    passes = [all(g["comparable"] and g["mean_drift_rms"] <= mean_tolerance
                  and g["log_std_drift_rms"] <= std_tolerance for g in change.values()) for change in tail]
    stable = enough and all(passes)
    return {"status": "stable_on_collected_distribution" if stable else "needs_more_data",
            "stable": stable, "consecutive_batches_required": stable_batches,
            "mean_drift_rms_tolerance": mean_tolerance, "log_std_drift_rms_tolerance": std_tolerance,
            "tail_batch_passes": passes, "scope": "fixed sampling policy and current task curriculum only"}


@torch.no_grad()
def policy_difference(actor, critic, before, after, observations):
    """Analytic KL(old norm || new norm), same weights/raw obs, no action sampling."""
    device = next(actor.parameters()).device
    raw = {name: value.to(device) for name, value in observations.items()}
    old, new = before(raw), after(raw)
    old_dist, new_dist = actor.distribution(old["actor"]), actor.distribution(new["actor"])
    kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1).double()
    result = {"gaussian_kl_mean": kl.mean().item(), "gaussian_kl_p95": kl.flatten().quantile(.95).item(),
              "gaussian_kl_max": kl.max().item(),
              "mean_action_shift_rms": (new_dist.mean.double() - old_dist.mean).square().mean().sqrt().item(),
              "value_shift_rms": (critic(new["critic"]).double() - critic(old["critic"])).square().mean().sqrt().item()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite normalization-induced policy change")
    return result


def _merge(traces):
    result = {name: torch.cat([b[name] for b in traces], dim=0)
              for name in ("actions", "rewards", "terminated", "truncated")}
    result["observations"] = {name: torch.cat([b["observations"][name] for b in traces], dim=0)
                              for name in traces[0]["observations"]}
    return result


def feature_profile(normalizer, observations):
    """Per-feature coverage and aggregate holdout diagnostics, not a success gate."""
    result = {}
    normalized = normalizer(observations)
    for name, values in observations.items():
        data = values.double().reshape(-1, values.shape[-1])
        norm = normalized[name].double().reshape_as(data)
        if not torch.isfinite(norm).all():
            raise RuntimeError("non-finite calibrated observations")
        raw_std = data.std(0, correction=0)
        result[name] = {"samples": data.shape[0], "features": data.shape[1],
                        "raw_min": data.amin(0).tolist(), "raw_max": data.amax(0).tolist(),
                        "raw_mean": data.mean(0).tolist(), "raw_std": raw_std.tolist(),
                        "near_constant_feature_indices": (raw_std < normalizer.stats[name].eps * .1).nonzero().flatten().tolist(),
                        "normalized_abs_p99": norm.abs().flatten().quantile(.99).item(),
                        "normalized_abs_p99_per_feature": norm.abs().quantile(.99, dim=0).tolist(),
                        "normalized_max_abs": norm.abs().max().item(),
                        "fraction_abs_gt_5": (norm.abs() > 5).double().mean().item()}
    if observations["actor"].shape[-1] == 61:
        # Shared Microduck actor layout: 48 proprioception + twist3/head4/body6.
        commands = observations["actor"][..., 48:61].double().reshape(-1, 13)
        result["command_block"] = {
            "actor_indices": list(range(48, 61)), "layout": "twist(3), head_pose(4), body_pose(6)",
            "min": commands.amin(0).tolist(), "max": commands.amax(0).tolist(),
            "std": commands.std(0, correction=0).tolist(),
            "varying_dimensions": ((commands.amax(0) - commands.amin(0)) > 1e-8).sum().item(),
            "exact_zero_vector_fraction": (commands == 0).all(-1).double().mean().item()}
    return result


@torch.no_grad()
def run_calibration(make_env, actor, critic, source_normalizer, *, fit_seeds=(42, 43),
                    holdout_seed=44, batches_per_seed=6, steps_per_batch=128, holdout_steps=256,
                    stable_batches=3, mean_tolerance=.05, std_tolerance=.05, on_batch=None):
    """Collect fit seeds continuously, then a fresh holdout env; never train weights.

    make_env(seed) must seed before startup events and use the SAME task recipe.
    A new seed starts a new segment, not a continuation of the previous seed.
    """
    for value, name in ((batches_per_seed, "batches_per_seed"), (steps_per_batch, "steps_per_batch"),
                        (holdout_steps, "holdout_steps")):
        _positive(value, name)
    fit_seeds = tuple(fit_seeds)
    if not fit_seeds or len(set(fit_seeds)) != len(fit_seeds):
        raise ValueError("fit_seeds must be nonempty and distinct")
    for seed in (*fit_seeds, holdout_seed):
        _seed(seed)
    if holdout_seed in fit_seeds:
        raise ValueError("holdout seed must be excluded from fitting")
    stability_summary([], stable_batches=stable_batches, mean_tolerance=mean_tolerance, std_tolerance=std_tolerance)
    if (actor.training or critic.training or source_normalizer.training
            or any(s.training for s in source_normalizer.stats.values())):
        raise ValueError("sampling models and normalizer must be in eval mode")
    if any(p.grad is not None for model in (actor, critic) for p in model.parameters()):
        raise ValueError("calibration expects models with no existing gradients")
    source = {"actor": cpu_snapshot(actor.state_dict()), "critic": cpu_snapshot(critic.state_dict()),
              "normalizer": source_normalizer.checkpoint()}
    candidate = ObservationNormalizer(source["normalizer"]["dims"]).to(next(actor.parameters()).device).eval()
    for name, stats in candidate.stats.items():
        stats.eps = source_normalizer.stats[name].eps
    histories, changes, batch_reports, episodes, segments = [], [], [], [], []
    traces = {"fit": [], "holdout": []}
    offsets = {"fit": 0, "holdout": 0}
    env_count = None

    def check_source():
        assert_same_state({"actor": actor.state_dict(), "critic": critic.state_dict(),
                           "normalizer": source_normalizer.checkpoint()}, source)
        if actor.training or critic.training or any(p.grad is not None for m in (actor, critic) for p in m.parameters()):
            raise RuntimeError("sampling models changed mode or acquired gradients")

    for phase, seeds in (("fit", fit_seeds), ("holdout", (holdout_seed,))):
        for seed in seeds:
            env = make_env(seed)
            try:
                if env.cfg.auto_reset or env.num_envs < 1:
                    raise ValueError("calibration requires positive num_envs and auto_reset=False")
                if env_count is None:
                    env_count = env.num_envs
                if env.num_envs != env_count:
                    raise ValueError("every seed must use the same num_envs")
                tracker = EpisodeTracker(env)
                continuation = None
                count = batches_per_seed if phase == "fit" else 1
                for local_batch in range(count):
                    check_source()
                    tracker.begin_iteration()
                    steps = steps_per_batch if phase == "fit" else holdout_steps
                    batch = collect_rollout(env, steps, actor=actor, critic=critic, normalizer=source_normalizer,
                                            bootstrap=True, initial_observations=continuation,
                                            on_transition=tracker.observe)
                    if continuation is not None:
                        for name, value in continuation.items():
                            torch.testing.assert_close(batch["observations"][name][0], value, rtol=0, atol=0)
                    continuation = {name: value[-1].clone() for name, value in batch["next_observations"].items()}
                    check_source()
                    before = candidate.checkpoint()
                    if phase == "fit":
                        candidate.train()
                        try:
                            candidate.update(batch["observations"])
                        finally:
                            candidate.eval()
                        after = candidate.checkpoint()
                        for name, stats in candidate.stats.items():
                            if stats.count.item() != before["state_dict"][f"stats.{name}.count"].item() + steps * env_count:
                                raise RuntimeError("candidate sample count mismatch")
                        change = moment_change(before, after)
                        changes.append(change)
                        histories.append(after)
                    else:
                        assert_same_state(candidate.checkpoint(), before)
                        change = None
                    trace = {name: cpu_snapshot(batch[name]) for name in (
                        "observations", "actions", "rewards", "terminated", "truncated")}
                    traces[phase].append(trace)
                    summary = tracker.summary()
                    episodes.extend({"phase": phase, "seed": seed, **e} for e in summary.pop("records"))
                    segments.append({"phase": phase, "seed": seed, "batch": local_batch + 1,
                                     "time_start": offsets[phase], "time_stop": offsets[phase] + steps,
                                     "continued_from_previous_batch": local_batch > 0})
                    offsets[phase] += steps
                    record = {"phase": phase, "seed": seed, "batch": local_batch + 1,
                              "counts": {name: s.count.item() for name, s in candidate.stats.items()},
                              "moment_change": change, "episodes": summary,
                              "source_normalizer_frozen": True, "model_parameters_unchanged": True}
                    batch_reports.append(record)
                    if on_batch is not None:
                        on_batch(record)
            finally:
                env.close()

    data = {phase: _merge(traces[phase]) for phase in traces}
    data["segments"] = segments
    holdout = data["holdout"]["observations"]
    frozen_candidate = candidate.checkpoint()
    comparison = policy_difference(actor, critic, source_normalizer, candidate, holdout)
    history_policy_changes = []
    for index in range(1, len(histories)):
        old = ObservationNormalizer.from_checkpoint(histories[index - 1]).to(next(actor.parameters()).device)
        new = ObservationNormalizer.from_checkpoint(histories[index]).to(next(actor.parameters()).device)
        history_policy_changes.append({"fit_batch": index + 1, **policy_difference(actor, critic, old, new, holdout)})
    check_source()
    assert_same_state(candidate.checkpoint(), frozen_candidate)
    report = {"fit_seeds": list(fit_seeds), "holdout_seed": holdout_seed, "num_envs": env_count,
              "fit_samples_per_group": offsets["fit"] * env_count,
              "holdout_samples_per_group": offsets["holdout"] * env_count,
              "source_counts": {name: s.count.item() for name, s in source_normalizer.stats.items()},
              "calibrated_counts": {name: s.count.item() for name, s in candidate.stats.items()},
              "stability": stability_summary(changes, stable_batches=stable_batches,
                                             mean_tolerance=mean_tolerance, std_tolerance=std_tolerance),
              "fit_profile": feature_profile(candidate, data["fit"]["observations"]),
              "holdout_profile": feature_profile(candidate, holdout),
              "holdout_normalization": observation_diagnostics(candidate, holdout),
              "source_to_calibrated_policy_change_on_holdout": comparison,
              "successive_calibration_policy_changes_on_holdout": history_policy_changes,
              "direct_swap_kl_threshold": .01, "direct_swap_recommended": comparison["gaussian_kl_mean"] <= .01,
              "sampling_policy_and_source_normalizer_unchanged": True, "holdout_excluded_from_fit": True,
              "optimizer_created": False, "backward_called": False, "candidate_frozen": True,
              "completed_episodes": {phase: sum(e["phase"] == phase for e in episodes) for phase in traces},
              "limitations": ["statistics describe this fixed policy and current curriculum, not future learned walking",
                              "stable moments do not establish compatibility with an existing model",
                              "statistics artifact is separate; no model/checkpoint has been replaced"],
              "batches": batch_reports, "episodes": episodes}
    return candidate, report, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from training.paths import INITIAL_CHECKPOINT
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--holdout-seed", type=int, default=44)
    parser.add_argument("--batches-per-seed", type=int, default=6)
    parser.add_argument("--steps-per-batch", type=int, default=128)
    parser.add_argument("--holdout-steps", type=int, default=256)
    parser.add_argument("--stable-batches", type=int, default=3)
    parser.add_argument("--mean-tolerance", type=float, default=.05)
    parser.add_argument("--std-tolerance", type=float, default=.05)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.output_dir.exists():
        parser.error("checkpoint must exist and output directory must be new")
    for name in ("num_envs", "batches_per_seed", "steps_per_batch", "holdout_steps", "stable_batches"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU simulation fallback")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from bam.mjlab import BamActuator
    from training.inspect_env import TASK_ID, ACTION_DIM, OBS_DIMS

    configure_torch_backends(allow_tf32=False)
    actor, critic, source_norm = load_policy(args.checkpoint, "cuda:0")
    if {name: s.dim for name, s in source_norm.stats.items()} != OBS_DIMS:
        raise RuntimeError("expected 61D actor and 76D critic")
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {"task_id": TASK_ID, "checkpoint": str(args.checkpoint.resolve()), "source_sha256": source_hash,
              "num_envs": args.num_envs, "fit_seeds": args.fit_seeds, "holdout_seed": args.holdout_seed,
              "batches_per_seed": args.batches_per_seed, "steps_per_batch": args.steps_per_batch,
              "holdout_steps": args.holdout_steps, "play": False, "auto_reset": False,
              "sampling": "fixed source actor, stochastic Gaussian actions, original frozen normalization",
              "statistics": "fresh population moments; raw policy-input observations only",
              "optimizer_updates": 0, "original_task_cfg_files_modified": False}
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    def make_env(seed):
        cfg = load_env_cfg(TASK_ID, play=False)
        cfg.scene.num_envs, cfg.seed, cfg.auto_reset = args.num_envs, seed, False
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
        try:
            actuators = env.scene["robot"].actuators
            if not actuators or any(not isinstance(a, BamActuator) or (a.cfg.motor_name, a.cfg.model) != ("xl330", "m6") for a in actuators):
                raise RuntimeError("expected original XL330 M6 BAM actuators")
            if env.action_manager.total_action_dim != ACTION_DIM or not math.isclose(env.step_dt, .02):
                raise RuntimeError("expected original 14 actions at 50 Hz")
        except BaseException:
            env.close()
            raise
        return env

    def log_batch(record):
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"[calibration] phase={record['phase']} seed={record['seed']} batch={record['batch']} "
              f"counts={record['counts']} completed={record['episodes']['completed_count']} "
              f"change={record['moment_change']}", flush=True)

    candidate, report, data = run_calibration(
        make_env, actor, critic, source_norm, fit_seeds=args.fit_seeds, holdout_seed=args.holdout_seed,
        batches_per_seed=args.batches_per_seed, steps_per_batch=args.steps_per_batch,
        holdout_steps=args.holdout_steps, stable_batches=args.stable_batches,
        mean_tolerance=args.mean_tolerance, std_tolerance=args.std_tolerance, on_batch=log_batch)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("source checkpoint changed")
    torch.save({"artifact_type": "observation_normalization_calibration", "normalizer": candidate.checkpoint(),
                "metadata": config, "automatically_applied_to_model": False}, args.output_dir / "normalizer.pt")
    torch.save(data, args.output_dir / "observations.pt")
    torch.save({"actor_config": actor.config, "critic_config": critic.config,
                "actor_state_dict": cpu_snapshot(actor.state_dict()), "critic_state_dict": cpu_snapshot(critic.state_dict()),
                "normalizer": source_norm.checkpoint()}, args.output_dir / "sampling_policy.pt")
    compact = {key: value for key, value in report.items() if key != "episodes"}
    (args.output_dir / "report.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "episodes.jsonl").open("w") as stream:
        for episode in report["episodes"]:
            stream.write(json.dumps(episode, allow_nan=False) + "\n")
    restored = ObservationNormalizer.from_checkpoint(torch.load(args.output_dir / "normalizer.pt", weights_only=True)["normalizer"])
    assert_same_state(restored.checkpoint(), candidate.checkpoint())
    print(f"[calibration] fit_samples={report['fit_samples_per_group']} holdout_samples={report['holdout_samples_per_group']} "
          f"stability={report['stability']['status']} direct_swap_recommended={report['direct_swap_recommended']}", flush=True)
    print("[calibration] COMPLETE: statistics saved separately; zero optimizer updates; original model unchanged", flush=True)
    print(f"[calibration] saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
