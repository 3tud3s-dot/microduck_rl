"""Vector-env PPO with explicit normalization boundaries and local JSONL logs."""

import argparse
import copy
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from training.advantages import compute_gae
from training.metrics import EpisodeTracker, optimization_metrics
from training.normalization import observation_diagnostics, update_between_iterations
from training.ppo import create_optimizer, optimizer_status, update_ppo
from training.rollout import collect_rollout, load_policy, verify_policy_rollout
from training.paths import INITIAL_CHECKPOINT


def cpu_snapshot(value):
    """Detach and copy a checkpoint tree so later steps cannot mutate saved state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def assert_same_state(current, reference):
    """Exact recursive comparison, including Adam moments and normalizer modes."""
    if isinstance(reference, torch.Tensor):
        if current.dtype != reference.dtype or not torch.equal(current.detach().cpu(), reference.cpu()):
            raise RuntimeError("checkpoint tensor changed unexpectedly")
    elif isinstance(reference, dict):
        if current.keys() != reference.keys():
            raise RuntimeError("checkpoint keys changed unexpectedly")
        for key in reference:
            assert_same_state(current[key], reference[key])
    elif isinstance(reference, (tuple, list)):
        if type(current) is not type(reference) or len(current) != len(reference):
            raise RuntimeError("checkpoint sequence changed unexpectedly")
        for item, saved in zip(current, reference):
            assert_same_state(item, saved)
    elif current != reference:
        raise RuntimeError("checkpoint metadata changed unexpectedly")


def run_training(env, actor, critic, normalizer, optimizer, *, iterations=3, steps=64,
                 gamma=0.99, gae_lambda=0.95, epochs=3, minibatch_size=32,
                 generator=None, on_iteration=None, normalization_mode="frozen",
                 start_iteration=0, start_env_steps=0, start_vector_steps=0):
    """Multiple epochs per fresh rollout; the same env/optimizer live throughout.

    Horizons do not reset the episode. Done rows have already been manually reset by
    the collector after capturing their true next-state bootstrap values.
    """
    if iterations < 1 or steps < 1:
        raise ValueError("iterations and steps must be positive")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0
           for x in (start_iteration, start_env_steps, start_vector_steps)):
        raise ValueError("starting counters must be nonnegative integers")
    if normalization_mode not in ("frozen", "running"):
        raise ValueError("normalization_mode must be frozen or running")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if minibatch_size is not None and (not isinstance(minibatch_size, int)
            or isinstance(minibatch_size, bool) or minibatch_size < 1):
        raise ValueError("minibatch_size must be a positive integer or None")
    if (not isinstance(env.num_envs, int) or isinstance(env.num_envs, bool)
            or env.num_envs < 1 or env.cfg.auto_reset):
        raise ValueError("training requires positive num_envs and auto_reset=False")
    if normalizer.training or any(group.training for group in normalizer.stats.values()):
        raise ValueError("normalizer must already be frozen")
    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    frozen = normalizer.checkpoint()
    previous_optimizer = cpu_snapshot(optimizer.state_dict())
    initial_step = optimizer_status(optimizer)["step"]
    samples_per_iteration = steps * env.num_envs
    updates_per_iteration = epochs * (1 if minibatch_size is None else
                                     math.ceil(samples_per_iteration / minibatch_size))
    tracker = EpisodeTracker(env)
    tracker.vector_steps = start_vector_steps
    device = torch.device(env.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    continuation, history = None, []
    for iteration in range(1, iterations + 1):
        assert_same_state(optimizer.state_dict(), previous_optimizer)
        assert_same_state(normalizer.checkpoint(), frozen)
        tracker.begin_iteration()
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        actor.eval()
        critic.eval()
        sampling_policy = {"actor_config": actor.config, "critic_config": critic.config,
                           "actor_state_dict": cpu_snapshot(actor.state_dict()),
                           "critic_state_dict": cpu_snapshot(critic.state_dict()),
                           "normalizer": normalizer.checkpoint()}
        batch = collect_rollout(env, steps, actor=actor, critic=critic, normalizer=normalizer,
                                bootstrap=True, initial_observations=continuation,
                                on_transition=tracker.observe)
        sync()
        collected = perf_counter()
        if continuation is not None:
            for name, x in continuation.items():
                torch.testing.assert_close(batch["observations"][name][0], x, rtol=0, atol=0)
        # Check the current sampling policy BEFORE updating any of its parameters.
        verify_policy_rollout(batch, actor, critic, normalizer)
        assert_same_state(actor.state_dict(), sampling_policy["actor_state_dict"])
        assert_same_state(critic.state_dict(), sampling_policy["critic_state_dict"])
        assert_same_state(optimizer.state_dict(), previous_optimizer)
        assert_same_state(normalizer.checkpoint(), frozen)
        continuation = {name: x[-1].detach().clone()
                        for name, x in batch["next_observations"].items()}
        targets = compute_gae(*(batch[key] for key in (
            "rewards", "values", "bootstrap_values", "terminated", "truncated")),
            gamma=gamma, gae_lambda=gae_lambda)
        normalization_inputs = observation_diagnostics(normalizer, batch["observations"])
        sync()
        update_started = perf_counter()
        report, _ = update_ppo(actor, critic, normalizer, batch, targets, optimizer=optimizer,
                               epochs=epochs, minibatch_size=minibatch_size, generator=generator)
        sync()
        updated = perf_counter()
        if report["samples"] != samples_per_iteration:
            raise RuntimeError("PPO sample count does not match time * num_envs")
        if (report["optimizer_steps"] != updates_per_iteration or
                report["optimizer_after"]["step"] != initial_step + iteration * updates_per_iteration):
            raise RuntimeError("Adam step count does not match epochs and minibatches")
        for name in ("actor", "critic"):
            if (report["gradients"][name]["before_clip_norm"] <= 0
                    or report["parameter_changes"][name]["changed_elements"] == 0):
                raise RuntimeError(f"{name}: expected nonzero gradients and parameter changes")
        assert_same_state(normalizer.checkpoint(), frozen)
        previous_optimizer = cpu_snapshot(optimizer.state_dict())
        boundary_update = {}
        boundary_policy_shift = {}
        if normalization_mode == "running":
            # Old log_prob/GAE and every PPO epoch used the frozen sampling stats.
            # Only the NEXT policy/checkpoint uses these newly accumulated moments.
            weights = cpu_snapshot({"actor": actor.state_dict(), "critic": critic.state_dict()})
            with torch.no_grad():
                normalized_before = normalizer(batch["observations"])
                distribution_before = actor.distribution(normalized_before["actor"])
                value_before = critic(normalized_before["critic"])
            boundary_update = update_between_iterations(normalizer, batch["observations"])
            with torch.no_grad():
                normalized_after = normalizer(batch["observations"])
                distribution_after = actor.distribution(normalized_after["actor"])
                value_after = critic(normalized_after["critic"])
                boundary_policy_shift = {
                    "gaussian_kl_mean": torch.distributions.kl_divergence(
                        distribution_before, distribution_after).sum(-1).mean().item(),
                    "mean_action_shift_rms": (distribution_after.mean.double() - distribution_before.mean).square().mean().sqrt().item(),
                    "value_shift_rms": (value_after.double() - value_before).square().mean().sqrt().item()}
                if not all(math.isfinite(v) for v in boundary_policy_shift.values()):
                    raise RuntimeError("non-finite policy shift after normalization update")
            assert_same_state({"actor": actor.state_dict(), "critic": critic.state_dict()}, weights)
            assert_same_state(optimizer.state_dict(), previous_optimizer)
            frozen = normalizer.checkpoint()
        sync()
        finished = perf_counter()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device) if device.type == "cuda" else (None, None)
        record = {"iteration": start_iteration + iteration,
                  "env_steps": start_env_steps + iteration * samples_per_iteration,
                  "vector_steps": start_vector_steps + iteration * steps, "num_envs": env.num_envs,
                  "samples_per_iteration": samples_per_iteration,
                  "flatten_order": "flat_index = time_index * num_envs + env_index",
                  "mean_step_reward": batch["rewards"].mean().item(),
                  "terminated": batch["terminated"].sum().item(),
                  "truncated": batch["truncated"].sum().item(),
                  "continued_from_previous_rollout": iteration > 1,
                  "optimizer_continuity_verified": True, "update": report,
                  "episodes": tracker.summary(),
                  "optimization": optimization_metrics(report, batch, targets, optimizer),
                  "normalization": {"mode": normalization_mode, "frozen_within_iteration": True,
                                    "sampling_inputs": normalization_inputs,
                                    "boundary_update": boundary_update,
                                    "boundary_policy_shift": boundary_policy_shift,
                                    "checkpoint_counts": {name: stats.count.item()
                                                          for name, stats in normalizer.stats.items()}},
                  "performance": {"collect_seconds": collected - started,
                                  "diagnostics_and_gae_seconds": update_started - collected,
                                  "update_seconds": updated - update_started,
                                  "post_update_seconds": finished - updated,
                                  "iteration_seconds_excluding_io": finished - started,
                                  "samples_per_second_excluding_io": samples_per_iteration / (finished - started),
                                  "collection_samples_per_second": samples_per_iteration / (collected - started),
                                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                                  if device.type == "cuda" else None,
                                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                                  if device.type == "cuda" else None,
                                  "cuda_device_free_bytes": free_bytes, "cuda_device_total_bytes": total_bytes}}
        history.append(record)
        if on_iteration is not None:
            on_iteration(record, batch, targets, sampling_policy)
    return history


def main(argv=None, *, on_iteration=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="restore counters and RNG; reset simulator")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--normalization", choices=("frozen", "running"), default="frozen",
                        help="running updates raw-observation statistics only AFTER each PPO iteration")
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.iterations < 1 or args.steps < 1:
        parser.error("--iterations and --steps must be positive")
    if args.epochs < 1 or args.minibatch_size < 1:
        parser.error("--epochs and --minibatch-size must be positive")
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("--lr must be finite and positive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory to preserve past runs")
    if not args.checkpoint.is_file():
        parser.error("--checkpoint does not exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU simulation fallback")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from bam.mjlab import BamActuator
    from training.inspect_env import ACTION_DIM, OBS_DIMS, TASK_ID

    # Full FP32 keeps sampling and flattened update comparisons reliable.
    # This is process-local and does not change the viewer/GPU launch script.
    configure_torch_backends(allow_tf32=False)
    torch.manual_seed(args.seed)
    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.auto_reset = False
    print(f"[train] task={TASK_ID} envs={args.num_envs} device=cuda:0 seed={args.seed} "
          f"iterations={args.iterations} steps={args.steps} epochs={args.epochs} "
          f"minibatch_size={args.minibatch_size}", flush=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        actuators = env.scene["robot"].actuators
        if not actuators or any(not isinstance(a, BamActuator) or
                                (a.cfg.motor_name, a.cfg.model) != ("xl330", "m6") for a in actuators):
            raise RuntimeError("expected original XL330 M6 BAM actuators")
        if env.action_manager.total_action_dim != ACTION_DIM or not math.isclose(env.step_dt, .02):
            raise RuntimeError("expected 14 actions at 50 Hz")
        actor, critic, normalizer = load_policy(args.checkpoint, env.device)
        if {name: group.dim for name, group in normalizer.stats.items()} != OBS_DIMS:
            raise RuntimeError("expected original actor/critic observation dimensions")
        optimizer = create_optimizer(actor, critic, args.lr if args.lr is not None else 1e-5)
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "optimizer_state_dict" in source:
            optimizer.load_state_dict(source["optimizer_state_dict"])
            if args.lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
        initial_step = optimizer_status(optimizer)["step"]
        counters = {"iteration": 0, "env_steps": 0, "vector_steps": 0}
        if args.resume:
            from training.checkpoint import restore_training_state
            counters = restore_training_state(source, env, actor, critic, normalizer, optimizer,
                                               normalization_mode=args.normalization)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        metadata = {"task_id": TASK_ID, "num_envs": args.num_envs, "step_dt": env.step_dt,
                    "seed": args.seed, "source_checkpoint": str(args.checkpoint.resolve()),
                    "iterations": args.iterations, "steps_per_iteration": args.steps,
                    "samples_per_iteration": args.steps * args.num_envs,
                    "env_steps": "total transitions across all envs",
                    "flatten_order": "flat_index = time_index * num_envs + env_index",
                    "gamma": .99, "gae_lambda": .95, "optimizer_initial_step": initial_step,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "device": str(env.device), "normalization_mode": args.normalization,
                    "normalization_frozen": args.normalization == "frozen",
                    "normalization_frozen_within_iteration": True,
                    "normalization_update": "raw policy inputs once, after PPO, before next iteration"
                    if args.normalization == "running" else "none",
                    "normalization_changes_policy_at_boundary": args.normalization == "running",
                    "reward_term_units": "weighted per-step contributions; dt applied exactly once",
                    "episode_statistics": "completed episodes only; unfinished episodes tracked separately",
                    "throughput_excludes_checkpoint_and_log_io": True, "log_schema_version": 1,
                    "gpu_memory_scope": "peak allocated/reserved = PyTorch only; free/total = whole CUDA device",
                    "optimization_metrics": "after PPO, before normalization boundary update",
                    "environment_reset_on_start": True, "simulator_state_saved": False,
                    "resume": args.resume, "initial_counters": counters, "save_every": args.save_every,
                    "ppo_epochs": args.epochs, "minibatch_size": args.minibatch_size,
                    "updates_per_iteration": args.epochs * math.ceil(
                        args.steps * args.num_envs / args.minibatch_size),
                    "allow_tf32": False}
        metrics_path = args.output_dir / "metrics.jsonl"
        (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

        def save_iteration(record, batch, targets, sampling_policy):
            payload = {"actor_config": actor.config, "critic_config": critic.config,
                       "actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
                       "actor_training": False, "critic_training": False,
                       "normalizer": normalizer.checkpoint(), "optimizer_state_dict": optimizer.state_dict(),
                       "metadata": metadata, "iteration": record["iteration"], "metrics": record,
                       "sampling_policy": sampling_policy, "rollout": batch, "targets": targets}
            from training.checkpoint import capture_training_state
            payload["training_state"] = capture_training_state(record, env)
            if (record["iteration"] % args.save_every == 0 or
                    record["iteration"] == counters["iteration"] + args.iterations):
                checkpoint = args.output_dir / f"iteration_{record['iteration']:04d}.pt"
                torch.save(cpu_snapshot(payload), checkpoint)
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            progress = {key: value for key, value in record.items() if key != "update"}
            progress["episodes"] = {key: value for key, value in record["episodes"].items()
                                    if key not in ("records", "unfinished")}
            with (args.output_dir / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(progress, allow_nan=False) + "\n")
            with (args.output_dir / "episodes.jsonl").open("a") as stream:
                for episode in record["episodes"]["records"]:
                    stream.write(json.dumps({"iteration": record["iteration"], **episode}, allow_nan=False) + "\n")
            report = record["update"]
            print(f"[train] iteration={record['iteration']} env_steps={record['env_steps']} "
                  f"vector_steps={record['vector_steps']} samples={record['samples_per_iteration']} "
                  f"loss={report['before']['loss']:.6f}->{report['after']['loss']:.6f} "
                  f"optimizer_step={report['optimizer_before']['step']}->{report['optimizer_after']['step']} "
                  f"updates={report['optimizer_steps']} clip_fraction={report['after']['clip_fraction']:.4f} "
                  f"moment_norm={report['optimizer_after']['exp_avg_norm']:.6f} "
                  f"mean_step_reward={record['mean_step_reward']:.6f} "
                  f"terminated={record['terminated']} truncated={record['truncated']} "
                  f"completed_episodes={record['episodes']['completed_count']} "
                  f"episode_return={record['episodes']['mean_return']} "
                  f"normalization={args.normalization} sampling_counts={report['normalizer_counts']} "
                  f"next_counts={record['normalization']['checkpoint_counts']} frozen_within_iteration=PASS", flush=True)
            if on_iteration is not None:
                on_iteration(record, batch, targets, sampling_policy)

        history = run_training(env, actor, critic, normalizer, optimizer,
                               iterations=args.iterations, steps=args.steps,
                               epochs=args.epochs, minibatch_size=args.minibatch_size,
                               on_iteration=save_iteration, normalization_mode=args.normalization,
                               start_iteration=counters["iteration"], start_env_steps=counters["env_steps"],
                               start_vector_steps=counters["vector_steps"])
        print(f"[train] PASS: {len(history)} fresh rollouts/updates, "
              f"Adam step {initial_step}->{optimizer_status(optimizer)['step']}, "
              "moments retained, fixed GAE/log_prob, full epoch coverage, normalization frozen within iterations, "
              "no reset at rollout horizons")
        print(f"[train] saved={args.output_dir.resolve()}")
        print("[train] Bounded training finished; assess policy quality with a separate evaluation.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
