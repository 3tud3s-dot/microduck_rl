"""Opt-in real CUDA train.py checks: python -m training.tests.integration.vector_train_smoke.

Run the CLI for 2/4 envs, three iterations, two epochs and minibatches of 50.
The 4-env case adds test-only termination schedules. Original task files stay put.
"""

import json
import argparse
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from training.paths import INITIAL_CHECKPOINT

from training import train
from training.tests.unit.test_vector_train import audited_update
from training.tests.integration.vector_rollout_smoke import scheduled_end
from training.vector_reset import _row_tensors, _temporal_buffers, reset_done_envs


class ResetAudit:
    def __init__(self):
        self.records = []

    def __call__(self, env, ids, terminal):
        n = env.num_envs
        keep = torch.ones(n, device=env.device, dtype=torch.bool)
        keep[ids] = False
        physics = {name: getattr(env.sim.data, name).clone() for name in (
            "qpos", "qvel", "ctrl", "time", "qacc_warmstart")}
        model = {name: getattr(env.sim.model, name).clone()
                 for name in ("dof_frictionloss", "dof_damping")}
        lengths, action = env.episode_length_buf.clone(), env.action_manager.action.clone()
        owners = list(env.command_manager._terms.values()) + list(env.action_manager._terms.values())
        temporal = list(_temporal_buffers(env.observation_manager))
        for entity in env.scene.entities.values():
            for actuator in entity.actuators:
                owners.append(actuator)
                if actuator._delay_buffer is not None:
                    temporal.append((actuator._delay_buffer._buffer, actuator._delay_buffer))
        rows = [s for owner in owners for s in _row_tensors(owner, n)]
        for name, value in vars(env).items():
            if name.startswith("_prev_") and isinstance(value, torch.Tensor) and value.shape[0] == n:
                rows.append((env, name, False, value.clone()))
        snapshots = []
        for circular, delay in temporal:
            snapshots.append((circular, circular._pointer, circular._buffer.clone(),
                              circular._num_pushes.clone()))
            if delay is not None:
                rows += _row_tensors(delay, n)
        result = reset_done_envs(env, ids, terminal)
        for name, before in physics.items():
            torch.testing.assert_close(getattr(env.sim.data, name)[keep], before[keep], rtol=0, atol=0)
        for name, before in model.items():
            torch.testing.assert_close(getattr(env.sim.model, name)[keep], before[keep], rtol=0, atol=0)
        torch.testing.assert_close(env.episode_length_buf[keep], lengths[keep], rtol=0, atol=0)
        torch.testing.assert_close(env.action_manager.action[keep], action[keep], rtol=0, atol=0)
        for circular, pointer, storage, pushes in snapshots:
            assert circular._pointer == pointer
            torch.testing.assert_close(circular._buffer[:, keep], storage[:, keep], rtol=0, atol=0)
            torch.testing.assert_close(circular._num_pushes[keep], pushes[keep], rtol=0, atol=0)
        for owner, name, is_dict, before in rows:
            after = owner[name] if is_dict else getattr(owner, name)
            torch.testing.assert_close(after[keep], before[keep], rtol=0, atol=0)
        for name in result:
            torch.testing.assert_close(result[name][keep], terminal[name][keep], rtol=0, atol=0)
            torch.testing.assert_close(env.observation_manager.compute()[name], result[name], rtol=0, atol=0)
        self.records.append({"vector_step": env.common_step_counter, "reset_ids": ids.cpu().tolist(),
                             "continuing_ids": keep.nonzero().flatten().cpu().tolist(),
                             "continuing_physics_commands_actions_delay_history_unchanged": True})
        return result


def run_case(n, output, checkpoint=INITIAL_CHECKPOINT):
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.managers import TerminationTermCfg

    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    initial_step = int(next(iter(source["optimizer_state_dict"]["state"].values()))["step"])
    directory = output / f"{n}envs"
    if directory.exists():
        raise FileExistsError(f"preserve previous results: {directory}")

    def config(task, **kwargs):
        cfg = load_env_cfg(task, **kwargs)
        if n == 4:
            for kind in ("terminated", "truncated"):
                cfg.terminations[f"validation_{kind}"] = TerminationTermCfg(
                    func=scheduled_end, params={"kind": kind}, time_out=kind == "truncated")
        return cfg

    audit = ResetAudit()
    argv = ["training.train", "--num-envs", str(n), "--steps", "64", "--iterations", "3",
            "--epochs", "2", "--minibatch-size", "50", "--checkpoint", str(checkpoint),
            "--output-dir", str(directory)]
    with patch.object(sys, "argv", argv), \
            patch("mjlab.tasks.registry.load_env_cfg", side_effect=config), \
            patch("training.train.update_ppo", side_effect=audited_update), \
            patch("training.rollout.reset_done_envs", side_effect=audit):
        train.main()

    saved = [torch.load(directory / f"iteration_{i:04d}.pt", map_location="cpu", weights_only=True)
             for i in (1, 2, 3)]
    updates = 2 * math.ceil(64 * n / 50)
    for i, item in enumerate(saved):
        report, batch = item["metrics"]["update"], item["rollout"]
        assert report["sample_alignment_verified"] and report["sample_axes"] == [64, n]
        assert report["samples"] == 64 * n and report["optimizer_steps"] == updates
        assert report["optimizer_before"]["step"] == initial_step + i * updates
        assert report["optimizer_after"]["step"] == initial_step + (i + 1) * updates
        assert item["metrics"]["env_steps"] == (i + 1) * 64 * n
        assert item["metrics"]["vector_steps"] == (i + 1) * 64
        train.assert_same_state(item["normalizer"], source["normalizer"])
        predecessor = source if i == 0 else saved[i-1]
        for name in ("actor", "critic"):
            train.assert_same_state(item["sampling_policy"][f"{name}_state_dict"],
                                    predecessor[f"{name}_state_dict"])
        if i:
            for name in batch["observations"]:
                torch.testing.assert_close(batch["observations"][name][0],
                                           predecessor["rollout"]["next_observations"][name][-1], rtol=0, atol=0)
        done = batch["terminated"] | batch["truncated"]
        delta = batch["rewards"] + .99 * batch["bootstrap_values"] - batch["values"]
        torch.testing.assert_close(item["targets"]["advantages"][done], delta[done])
        assert (batch["bootstrap_values"][batch["terminated"]] == 0).all()
        assert torch.isfinite(item["targets"]["returns"]).all()
    assert audit.records and any(r["continuing_ids"] for r in audit.records)
    if n == 4:
        batch = saved[0]["rollout"]
        assert batch["terminated"][6, 0]
        assert batch["truncated"][10, 1] and not batch["terminated"][10, 1]
        assert batch["terminated"][12, 0] and batch["truncated"][12, 0]
        assert batch["truncated"][16].all()
    result = {"num_envs": n, "iterations": 3, "steps_per_iteration": 64,
              "samples_per_iteration": 64 * n, "epochs": 2, "minibatch_size": 50,
              "last_minibatch_size": 64 * n % 50, "updates_per_iteration": updates,
              "optimizer_steps": [initial_step] + [s["metrics"]["update"]["optimizer_after"]["step"] for s in saved],
              "env_steps": saved[-1]["metrics"]["env_steps"],
              "normalizer_counts": saved[-1]["metrics"]["update"]["normalizer_counts"],
              "frozen_normalizer_and_minibatch_alignment_verified": True,
              "optimizer_and_policy_continuity_verified": True,
              "test_only_scheduled_terminations": n == 4, "reset_audits": audit.records}
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[vector-train] " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Import before BAM/task modules to respect their existing initialization order.
    from mjlab.envs import ManagerBasedRlEnv  # noqa: F401
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"preserve previous results: {output}")
    output.mkdir(parents=True)
    results = [run_case(n, output, args.checkpoint) for n in (2, 4)]
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("[vector-train] PASS: three fresh collect/GAE/PPO iterations for both 2 and 4 envs")


if __name__ == "__main__":
    main()
