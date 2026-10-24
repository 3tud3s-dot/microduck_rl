"""Opt-in CUDA probe: python -m training.tests.integration.vector_rollout_smoke.

Two envs use the unchanged training task. Four envs additionally get test-only
termination terms to deterministically cover timeout, both flags and all-row reset.
No optimizer updates, configuration files or source checkpoints are changed.
"""

import json
import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from training.paths import INITIAL_CHECKPOINT

from training.advantages import compute_gae
from training.rollout import collect_rollout, load_policy, verify_policy_rollout
from training.train import assert_same_state, cpu_snapshot
from training.vector_reset import _row_tensors, _temporal_buffers, reset_done_envs


def scheduled_end(env, kind):
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    step = env.common_step_counter
    if kind == "terminated" and step in (7, 13):
        result[0] = True
    if kind == "truncated":
        if step == 11:
            result[1] = True
        if step == 13:
            result[0] = True
        if step == 17:
            result[:] = True
    return result


def run_case(n, controlled, output, checkpoint=INITIAL_CHECKPOINT):
    # mjlab.envs must precede bam imports (same order as the existing entrypoints).
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.managers import TerminationTermCfg, ObservationGroupCfg, ObservationTermCfg
    from training.inspect_env import TASK_ID

    actor, critic, norm = load_policy(checkpoint, "cuda:0")
    saved = cpu_snapshot({"actor": actor.state_dict(), "critic": critic.state_dict(),
                          "normalizer": norm.checkpoint()})
    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = n, 42, False
    if controlled:
        for kind in ("terminated", "truncated"):
            cfg.terminations[f"validation_{kind}"] = TerminationTermCfg(
                func=scheduled_end, params={"kind": kind}, time_out=kind == "truncated")
        # A diagnostic group exercises actual GPU history buffers without changing
        # the actor/critic inputs or the loaded checkpoint's 61D/76D statistics.
        cfg.observations["history_probe"] = ObservationGroupCfg(terms={
            "joints": ObservationTermCfg(func=joint_history_probe, history_length=3)})
        from training.normalization import RunningMeanStd
        norm.stats["history_probe"] = RunningMeanStd(42).to("cuda:0").eval()
        saved["normalizer"] = cpu_snapshot(norm.checkpoint())
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    audits = []

    def checked_reset(raw, ids, terminal):
        keep = torch.ones(n, device=raw.device, dtype=torch.bool)
        keep[ids] = False
        physics = {name: getattr(raw.sim.data, name).clone() for name in (
            "qpos", "qvel", "ctrl", "time", "qacc_warmstart")}
        model_fields = {name: getattr(raw.sim.model, name).clone()
                        for name in ("dof_frictionloss", "dof_damping")}
        lengths = raw.episode_length_buf.clone()
        action = raw.action_manager.action.clone()
        rows = []
        for owner in raw.command_manager._terms.values():
            rows += _row_tensors(owner, n)
        for owner in raw.action_manager._terms.values():
            rows += _row_tensors(owner, n)
        for entity in raw.scene.entities.values():
            for actuator in entity.actuators:
                rows += _row_tensors(actuator, n)
        # Original reward action-history caches must remain intact too.
        for name, value in vars(raw).items():
            if name.startswith("_prev_") and isinstance(value, torch.Tensor) and value.shape[0] == n:
                rows.append((raw, name, False, value.clone()))
        temporal = []
        for circular, delay in _temporal_buffers(raw.observation_manager):
            temporal.append((circular, circular._pointer, circular._buffer.clone(),
                             circular._num_pushes.clone(), delay,
                             _row_tensors(delay, n) if delay is not None else []))
        for entity in raw.scene.entities.values():
            for actuator in entity.actuators:
                delay = actuator._delay_buffer
                if delay is not None:
                    circular = delay._buffer
                    temporal.append((circular, circular._pointer, circular._buffer.clone(),
                                     circular._num_pushes.clone(), delay, _row_tensors(delay, n)))
        result = reset_done_envs(raw, ids, terminal)
        for name, saved_tensor in physics.items():
            torch.testing.assert_close(getattr(raw.sim.data, name)[keep], saved_tensor[keep],
                                       rtol=0, atol=0, msg=f"continuing physics changed: {name}")
        for name, saved_tensor in model_fields.items():
            torch.testing.assert_close(getattr(raw.sim.model, name)[keep], saved_tensor[keep], rtol=0, atol=0)
        torch.testing.assert_close(raw.episode_length_buf[keep], lengths[keep], rtol=0, atol=0)
        torch.testing.assert_close(raw.episode_length_buf[ids], torch.zeros_like(lengths[ids]), rtol=0, atol=0)
        torch.testing.assert_close(raw.action_manager.action[keep], action[keep], rtol=0, atol=0)
        for circular, pointer, storage, pushes, delay, delay_rows in temporal:
            assert circular._pointer == pointer
            torch.testing.assert_close(circular._buffer[:, keep], storage[:, keep], rtol=0, atol=0)
            torch.testing.assert_close(circular._num_pushes[keep], pushes[keep], rtol=0, atol=0)
            assert (circular._num_pushes[ids] == 1).all()
            rows += delay_rows
        for owner, name, is_dict, saved_tensor in rows:
            current = owner[name] if is_dict else getattr(owner, name)
            torch.testing.assert_close(current[keep], saved_tensor[keep], rtol=0, atol=0)
        for name in result:
            torch.testing.assert_close(result[name][keep], terminal[name][keep], rtol=0, atol=0)
            torch.testing.assert_close(raw.observation_manager.compute()[name], result[name], rtol=0, atol=0)
        audits.append({"step": raw.common_step_counter, "reset_ids": ids.cpu().tolist(),
                       "continuing_ids": keep.nonzero().flatten().cpu().tolist(),
                       "physics_commands_actions_delay_history_unchanged": True})
        return result

    try:
        kwargs = dict(actor=actor, critic=critic, normalizer=norm, bootstrap=True)
        with patch("training.rollout.reset_done_envs", side_effect=checked_reset):
            batch = collect_rollout(env, 64, **kwargs)
            continuation = {k: x[-1] for k, x in batch["next_observations"].items()}
            following = collect_rollout(env, 8, initial_observations=continuation, **kwargs)
        verify_policy_rollout(batch, actor, critic, norm)
        verify_policy_rollout(following, actor, critic, norm)
        for name, x in continuation.items():
            torch.testing.assert_close(following["observations"][name][0], x, rtol=0, atol=0)
            torch.testing.assert_close(batch["observations"][name][1:],
                                       batch["next_observations"][name][:-1], rtol=0, atol=0)
        targets = compute_gae(*(batch[k] for k in (
            "rewards", "values", "bootstrap_values", "terminated", "truncated")))
        done = batch["terminated"] | batch["truncated"]
        delta = batch["rewards"] + .99 * batch["bootstrap_values"] - batch["values"]
        torch.testing.assert_close(targets["advantages"][done], delta[done])
        # Changing rewards in the NEW episode must not leak backwards across done.
        poisoned = batch["rewards"].clone()
        if controlled:
            assert batch["terminated"][6, 0] and not batch["truncated"][6, 0]
            assert batch["truncated"][10, 1] and not batch["terminated"][10, 1]
            assert batch["terminated"][12, 0] and batch["truncated"][12, 0]
            assert batch["truncated"][16].all()
            poisoned[11, 1] += 10000
            poison_targets = compute_gae(poisoned, *(batch[k] for k in (
                "values", "bootstrap_values", "terminated", "truncated")))
            torch.testing.assert_close(poison_targets["advantages"][:11, 1],
                                       targets["advantages"][:11, 1], rtol=0, atol=0)
        assert audits and any(a["continuing_ids"] for a in audits)
        assert not done[-1].all()  # Unfinished horizon also retains bootstrap.
        assert not norm.training and not any(s.training for s in norm.stats.values())
        assert_same_state({"actor": actor.state_dict(), "critic": critic.state_dict(),
                           "normalizer": norm.checkpoint()}, saved)
        report = {"num_envs": n, "steps": 64, "continuation_steps": 8,
                  "test_only_scheduled_terminations": controlled,
                  "test_only_3frame_history_group": controlled,
                  "terminated": int(batch["terminated"].sum()), "truncated": int(batch["truncated"].sum()),
                  "both_flags": int((batch["terminated"] & batch["truncated"]).sum()),
                  "normalizer_counts": {k: int(s.count) for k, s in norm.stats.items()},
                  "normalization_and_policy_unchanged": True, "bootstrap_and_gae_checked": True,
                  "reset_audits": audits}
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"rollout": cpu_snapshot(batch), "targets": cpu_snapshot(targets),
                    "continuation": cpu_snapshot(following), "report": report}, output / f"{n}envs.pt")
        print(json.dumps(report), flush=True)
        return report
    finally:
        env.close()


def joint_history_probe(env):
    return env.scene["robot"].data.joint_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"preserve previous results: {output}")
    reports = [run_case(2, False, output, args.checkpoint), run_case(4, True, output, args.checkpoint)]
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("[vector-rollout] PASS: local reset isolation, bootstrap, GAE, continuation and frozen normalization")


if __name__ == "__main__":
    main()
