"""Fixed-command, paired-seed behavior diagnostics; no optimizer or reward edits."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import torch

from training.evaluate import _check_observations
from training.rollout import load_policy
from training.train import assert_same_state, cpu_snapshot


CASES = (
    ("standing", "standing", (0., 0., 0.)),
    ("forward", "straight", (.2, 0., 0.)),
    ("turn_left", "turn_in_place", (0., 0., .5)),
    ("turn_right", "turn_in_place", (0., 0., -.5)),
    ("moving_left", "moving_turn", (.2, 0., .5)),
    ("moving_right", "moving_turn", (.2, 0., -.5)),
)


def _finite_vector(name, value):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim < 1 or value.shape[-1] != 3 or not value.numel()
            or not torch.isfinite(value).all().item()):
        raise ValueError(name + " must be finite floating [...,3]")


def compute_measurements(linear_b, angular_b, gravity_b, command):
    for name, value in (("linear", linear_b), ("angular", angular_b),
                        ("gravity", gravity_b), ("command", command)):
        _finite_vector(name, value)
        if value.shape != linear_b.shape or value.device != linear_b.device or value.dtype != linear_b.dtype:
            raise ValueError("measurement shape/device/dtype must match")
    gravity_norm = gravity_b.norm(dim=-1)
    if (gravity_norm <= 0).any().item():
        raise ValueError("gravity vector must be nonzero")
    target_linear = torch.cat((command[..., :2], torch.zeros_like(command[..., 2:])), -1)
    target_angular = torch.cat((torch.zeros_like(command[..., :2]), command[..., 2:]), -1)
    linear_error = linear_b - target_linear
    angular_error = angular_b - target_angular
    return {"linear_error_b": linear_error, "angular_error_b": angular_error,
            "linear_xy_error_mps": linear_error[..., :2].norm(dim=-1),
            "yaw_error_radps": angular_error[..., 2],
            "tilt_deg": torch.acos((-gravity_b[..., 2] / gravity_norm).clamp(-1, 1)).rad2deg()}


def aggregate_episode(trace, *, step_dt, end_terms):
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive")
    command = trace["commands"]
    if command.ndim != 2:
        raise ValueError("trace commands must be [time,3]")
    measurements = compute_measurements(trace["linear_velocity_b"], trace["angular_velocity_b"],
                                        trace["projected_gravity_b"], command)
    count = len(command)
    for key in ("actions", "action_std"):
        value = trace[key]
        if (value.ndim != 2 or value.shape[0] != count or value.shape[1] < 1 or
                not value.is_floating_point() or not torch.isfinite(value).all().item()):
            raise ValueError(key + " must be finite [time,action]")
    if trace["actions"].shape != trace["action_std"].shape or (trace["action_std"] <= 0).any().item():
        raise ValueError("positive action std must match actions")
    if trace["rewards"].shape != (count,) or not torch.isfinite(trace["rewards"]).all().item():
        raise ValueError("reward must be finite [time]")
    for key in ("terminated", "truncated"):
        if trace[key].shape != (count,) or trace[key].dtype != torch.bool:
            raise ValueError("flags must be boolean [time]")
    if (trace["terminated"][:-1] | trace["truncated"][:-1]).any().item():
        raise ValueError("trace continued past a done transition")
    terminated, truncated = trace["terminated"][-1].item(), trace["truncated"][-1].item()
    fell = "fell_over" in end_terms
    if fell and not terminated:
        raise ValueError("fell_over requires a terminated transition")
    linear_error, angular_error = measurements["linear_error_b"].double(), measurements["angular_error_b"].double()
    duration = count * step_dt
    return {"steps": count, "duration_s": duration, "episode_return": trace["rewards"].double().sum().item(),
            "terminated": terminated, "truncated": truncated, "evaluation_cutoff": not (terminated or truncated),
            "active_end_terms": list(end_terms), "fall_detected": fell, "fall_time_s": duration if fell else None,
            "linear_xy_rmse_mps": linear_error[:, :2].square().sum(-1).mean().sqrt().item(),
            "linear_xy_mae_mps": measurements["linear_xy_error_mps"].double().mean().item(),
            "linear_signed_bias_mps": linear_error.mean(0).tolist(),
            "yaw_rmse_radps": angular_error[:, 2].square().mean().sqrt().item(),
            "yaw_mae_radps": angular_error[:, 2].abs().mean().item(),
            "yaw_signed_bias_radps": angular_error[:, 2].mean().item(),
            "vertical_speed_rms_mps": linear_error[:, 2].square().mean().sqrt().item(),
            "roll_pitch_rate_rms_radps": angular_error[:, :2].square().sum(-1).mean().sqrt().item(),
            "mean_linear_velocity_b": trace["linear_velocity_b"].double().mean(0).tolist(),
            "mean_angular_velocity_b": trace["angular_velocity_b"].double().mean(0).tolist(),
            "tilt_mean_deg": measurements["tilt_deg"].double().mean().item(),
            "tilt_max_deg": measurements["tilt_deg"].max().item(),
            "mean_action_std": trace["action_std"].double().mean().item(),
            "max_action_abs": trace["actions"].abs().max().item()}


def install_fixed_commands(env, command):
    """Wrap live instances before reset; preserve original samplers and RNG draws."""
    twist = env.command_manager.get_term("twist")
    if getattr(twist.cfg, "init_velocity_prob", 0.) != 0.:
        raise ValueError("diagnostics require init_velocity_prob=0 to preserve initial physics")
    targets = {"twist": torch.tensor([command], device=env.device, dtype=torch.float32),
               "head_pose": torch.zeros((1, 4), device=env.device),
               "body_pose": torch.zeros((1, 6), device=env.device)}
    for name, target in targets.items():
        term = env.command_manager.get_term(name)
        if term.command.shape != target.shape:
            raise ValueError("unexpected command dimensions: " + name)
        resample, update = term._resample_command, term._update_command

        def pin(this, value=target):
            this.command.copy_(value)
            if hasattr(this, "vel_command_w"):
                this.vel_command_w.copy_(value)

        def resample_fixed(this, ids, original=resample, force=pin):
            original(ids)
            force(this)

        def update_fixed(this, original=update, force=pin):
            original()
            force(this)

        term._resample_command = MethodType(resample_fixed, term)
        term._update_command = MethodType(update_fixed, term)
    return targets


def check_commands(env, observations, targets):
    for name, target in targets.items():
        torch.testing.assert_close(env.command_manager.get_command(name), target, rtol=0, atol=0)
    # Actor's deployment contract: 48 proprioception + twist3 + head4 + body6.
    expected = torch.cat(tuple(targets.values()), -1)
    torch.testing.assert_close(observations["actor"][:, 48:61], expected, rtol=0, atol=0)


def initial_conditions(env):
    from training.vector_reset import _temporal_buffers
    robot = env.scene["robot"]
    result = {"physics": {name: getattr(env.sim.data, name) for name in
                           ("qpos", "qvel", "ctrl", "time", "qacc_warmstart")},
              "model": {name: getattr(env.sim.model, name) for name in
                         sorted(env.event_manager.domain_randomization_fields)},
              "encoder_bias": robot.data.encoder_bias,
              "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
              "actuators": [], "observation_history": [],
              "interval_times": env.event_manager._interval_term_time_left,
              "reset_event_steps": env.event_manager._reset_term_last_triggered_step_id}
    if hasattr(env, "_imu_misalign_quat"):
        result["imu_misalignment"] = env._imu_misalign_quat
    for actuator in robot.actuators:
        fields = {name: value for name, value in vars(actuator).items() if isinstance(value, torch.Tensor)}
        delay = actuator._delay_buffer
        if delay is not None:
            fields["delay"] = {name: value for name, value in vars(delay).items() if isinstance(value, torch.Tensor)}
            circular = delay._buffer
            fields["motor_history"] = {"pointer": circular._pointer, "storage": circular._buffer,
                                       "pushes": circular._num_pushes}
        result["actuators"].append(fields)
    for circular, delay in _temporal_buffers(env.observation_manager):
        item = {"pointer": circular._pointer, "storage": circular._buffer, "pushes": circular._num_pushes}
        if delay is not None:
            item["delay"] = {name: value for name, value in vars(delay).items() if isinstance(value, torch.Tensor)}
        result["observation_history"].append(item)
    return cpu_snapshot(result)


def assert_conditions(current, expected, path="initial"):
    if isinstance(expected, dict):
        if current.keys() != expected.keys():
            raise RuntimeError(path + ": differing keys")
        for key in expected:
            assert_conditions(current[key], expected[key], path + "/" + str(key))
    elif isinstance(expected, (tuple, list)):
        if type(current) is not type(expected) or len(current) != len(expected):
            raise RuntimeError(path + ": differing sequence")
        for index, (live, reference) in enumerate(zip(current, expected)):
            assert_conditions(live, reference, path + "/" + str(index))
    else:
        try:
            if path.endswith("/model/body_iquat"):
                # CUDA's inertia eigendecomposition can differ by one FP32 ULP
                # for the same exact randomized inertia/pose. All source fields,
                # actual physics, observations, histories and RNG remain exact.
                torch.testing.assert_close(current, expected, rtol=0, atol=1e-7)
            else:
                assert_same_state(current, expected)
        except (RuntimeError, AssertionError) as error:
            detail = ""
            if isinstance(expected, torch.Tensor) and current.shape == expected.shape:
                detail = f" shape={tuple(expected.shape)} max_abs_difference={(current.double()-expected.double()).abs().max().item()}"
            raise RuntimeError(path + detail) from error


@torch.no_grad()
def diagnose_episode(env, actor, critic, normalizer, *, seed, command, max_steps,
                     expected=None, expected_physics=None):
    if env.num_envs != 1 or env.cfg.auto_reset or max_steps < 1:
        raise ValueError("diagnostics require one manual-reset env and positive steps")
    if actor.training or critic.training or normalizer.training or any(g.training for g in normalizer.stats.values()):
        raise ValueError("models and normalization must already be frozen in eval mode")
    frozen = cpu_snapshot({"actor": actor.state_dict(), "critic": critic.state_dict(), "normalizer": normalizer.checkpoint()})
    targets = install_fixed_commands(env, command)
    observations, _ = env.reset(seed=seed)
    _check_observations(observations, normalizer, env.device)
    check_commands(env, observations, targets)
    conditions = {"observations": cpu_snapshot(observations), "state": initial_conditions(env)}
    if expected is not None:
        assert_conditions(conditions, expected)
    if expected_physics is not None:
        assert_conditions(conditions["state"], expected_physics)
    records = []
    for _ in range(max_steps):
        check_commands(env, observations, targets)
        record = {"observations": cpu_snapshot(observations), "commands": targets["twist"][0].clone()}
        distribution = actor.distribution(normalizer(observations)["actor"])
        action = distribution.mean
        record.update({"actions": action[0].clone(), "action_std": distribution.stddev[0].clone()})
        observations, reward, terminated, truncated, _ = env.step(action)
        _check_observations(observations, normalizer, env.device)
        check_commands(env, observations, targets)
        data = env.scene["robot"].data
        record.update({"linear_velocity_b": data.root_link_lin_vel_b[0].clone(),
                       "angular_velocity_b": data.root_link_ang_vel_b[0].clone(),
                       "projected_gravity_b": data.projected_gravity_b[0].clone(),
                       "root_position_w": data.root_link_pos_w[0].clone(),
                       "root_quaternion_wxyz": data.root_link_quat_w[0].clone(),
                       "rewards": reward[0].clone(), "terminated": terminated[0].clone(), "truncated": truncated[0].clone()})
        records.append(cpu_snapshot(record))
        if "nan_state" in env.termination_manager.active_terms and env.termination_manager.get_term("nan_state").any().item():
            raise RuntimeError("nan_state during diagnosis")
        if terminated.item() or truncated.item():
            break
    trace = {key: torch.stack([row[key] for row in records]) for key in records[0] if key != "observations"}
    trace["observations"] = {name: torch.stack([row["observations"][name] for row in records]) for name in observations}
    terms = [name for name in env.termination_manager.active_terms if env.termination_manager.get_term(name).any().item()]
    result = aggregate_episode(trace, step_dt=env.step_dt, end_terms=terms)
    for name, module in (("actor", actor), ("critic", critic)):
        assert_same_state(module.state_dict(), frozen[name])
        if any(p.grad is not None for p in module.parameters()):
            raise RuntimeError("diagnosis unexpectedly created gradients")
    assert_same_state(normalizer.checkpoint(), frozen["normalizer"])
    result.update({"seed": seed, "command": list(command), "policy_unchanged": True, "normalizer_unchanged": True,
                   "normalizer_counts": {name: int(g.count) for name, g in normalizer.stats.items()},
                   "initial_conditions_matched": expected is not None, "fixed_commands_verified": True})
    return result, trace, conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from training.paths import REFERENCE_RUN
    parser.add_argument("--run", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    parser.add_argument("--max-seconds", type=float, default=10.)
    args = parser.parse_args()
    if (args.output_dir.exists() or len(set(args.seeds)) != len(args.seeds) or
            any(seed < 0 or seed >= 2**32 for seed in args.seeds) or
            not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("require new output directory, distinct uint32 seeds, positive finite duration")
    checkpoints = {i: args.run / f"train/iteration_{i:04d}.pt" for i in (25, 50, 100)}
    if any(not path.is_file() for path in checkpoints.values()):
        parser.error("all three stage checkpoints must exist")
    hashes = {i: hashlib.sha256(path.read_bytes()).hexdigest() for i, path in checkpoints.items()}
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from bam.mjlab import BamActuator
    from training.inspect_env import TASK_ID
    configure_torch_backends(allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "traces").mkdir()
    rows, common_windows = [], []
    for seed in args.seeds:
        seed_physics = None
        for case, task, command in CASES:
            reference, traces = None, {}
            for iteration, checkpoint in checkpoints.items():
                actor, critic, normalizer = load_policy(checkpoint, "cuda:0")
                cfg = load_env_cfg(TASK_ID, play=False)
                cfg.seed, cfg.scene.num_envs, cfg.auto_reset = seed, 1, False
                env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
                try:
                    actuators = env.scene["robot"].actuators
                    if (not actuators or any(not isinstance(a, BamActuator) or
                            (a.cfg.motor_name, a.cfg.model) != ("xl330", "m6") for a in actuators)
                            or env.action_manager.total_action_dim != 14 or not math.isclose(env.step_dt, .02)):
                        raise RuntimeError("original BAM/14D/50 Hz contract changed")
                    max_steps = max(1, math.floor(args.max_seconds / env.step_dt))
                    result, trace, conditions = diagnose_episode(env, actor, critic, normalizer, seed=seed,
                        command=command, max_steps=max_steps, expected=reference, expected_physics=seed_physics)
                finally:
                    env.close()
                    del env
                    gc.collect()
                if reference is None:
                    reference = conditions
                if seed_physics is None:
                    seed_physics = conditions["state"]
                result.update({"iteration": iteration, "case": case, "task": task,
                               "paired_initial_conditions_verified": True,
                               "cross_case_initial_physics_verified": True})
                rows.append(result)
                traces[iteration] = trace
                filename = f"{case}_seed{seed}_iteration{iteration:04d}.pt"
                torch.save({"result": result, "trace": trace, "initial_conditions": conditions}, args.output_dir / "traces" / filename)
                with (args.output_dir / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                print(f"[diagnose] {case} seed={seed} iteration={iteration} duration={result['duration_s']:.2f}s "
                      f"fell={result['fall_detected']} xy_rmse={result['linear_xy_rmse_mps']:.4f}m/s "
                      f"yaw_rmse={result['yaw_rmse_radps']:.4f}rad/s tilt_max={result['tilt_max_deg']:.1f}deg "
                      "paired_conditions=PASS commands=PASS frozen=PASS", flush=True)
                del actor, critic, normalizer
            length = min(len(trace["rewards"]) for trace in traces.values())
            for iteration, trace in traces.items():
                m = compute_measurements(*(trace[key][:length] for key in
                    ("linear_velocity_b", "angular_velocity_b", "projected_gravity_b", "commands")))
                common_windows.append({"case": case, "seed": seed, "iteration": iteration,
                    "steps": length, "duration_s": length * .02,
                    "linear_xy_rmse_mps": m["linear_xy_error_mps"].double().square().mean().sqrt().item(),
                    "yaw_rmse_radps": m["yaw_error_radps"].double().square().mean().sqrt().item(),
                    "tilt_mean_deg": m["tilt_deg"].double().mean().item()})
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[i] for i, path in checkpoints.items())
    metadata = {"task_id": TASK_ID, "device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
                "seeds": args.seeds, "iterations": list(checkpoints), "cases": CASES,
                "max_seconds": args.max_seconds, "step_dt": .02,
                "action_mode": "actor Gaussian mean; no sampling/clipping/filtering",
                "velocity_frame": "true root-link velocity in body frame; same view as original velocity reward",
                "tilt": "true root-link projected gravity; acos(-gz/||g||) in degrees",
                "fall_time": "first original fell_over termination at step*dt; null if none",
                "state_timing": "fresh post-step derived state; original termination/reward use previous physics substep",
                "head_body_commands": "zero delta from HOME; all 61 actor slots retained",
                "conditions": "fresh train-config env per trial; same seed before startup and reset; exact physics/DR/delay/RNG comparisons except derived inertia quaternion",
                "derived_model_tolerance": {"body_iquat": {"rtol": 0, "atol": 1e-7}},
                "curricula": "fresh counters at 0 for every trial, matching previous paired evaluation",
                "checkpoint_sha256": hashes, "checkpoint_unchanged": True,
                "training_started": False, "reward_changed": False}
    report = {"metadata": metadata, "episodes": rows, "common_windows": common_windows,
              "caveats": ["Errors use observed pre-end duration; early falls can reduce apparent error.",
                          "Common windows compare identical time prefixes within each case/seed across all three models.",
                          "Ten-second cutoff is neither a fall nor evidence of task success.",
                          "Four seeds are a diagnostic sample, not a deployment qualification."]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[diagnose] PASS {len(rows)} episodes; no optimizer/updates; output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
