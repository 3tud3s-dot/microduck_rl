"""Bounded local check: 64 envs x 24 steps x 5 iterations, frozen calibration."""

import hashlib
import json
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from training.paths import INITIAL_CHECKPOINT

from training import train
from training.tests.unit.test_vector_train import audited_update
from training.tests.integration.vector_train_smoke import ResetAudit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from mjlab.envs import ManagerBasedRlEnv  # noqa: F401
    initial_path = args.checkpoint
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"preserve previous run: {output}")
    source_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    source = torch.load(initial_path, map_location="cpu", weights_only=True)
    audit = ResetAudit()
    argv = ["training.train", "--checkpoint", str(initial_path), "--output-dir", str(output),
            "--num-envs", "64", "--steps", "24", "--iterations", "5", "--epochs", "2",
            "--minibatch-size", "256", "--normalization", "frozen"]
    with patch.object(sys, "argv", argv), \
            patch("training.train.update_ppo", side_effect=audited_update), \
            patch("training.rollout.reset_done_envs", side_effect=audit):
        train.main()
    checkpoints = [torch.load(output / f"iteration_{i:04d}.pt", map_location="cpu", weights_only=True)
                   for i in range(1, 6)]
    rows = []
    penalty_names = ("body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2",
                     "foot_clearance", "foot_swing_height", "foot_slip", "self_collisions", "head_pose_bias")
    for i, item in enumerate(checkpoints):
        record, update = item["metrics"], item["metrics"]["update"]
        assert update["sample_alignment_verified"] and update["sample_axes"] == [24, 64]
        assert update["optimizer_before"]["step"] == i * 12
        assert update["optimizer_after"]["step"] == (i + 1) * 12
        train.assert_same_state(item["normalizer"], source["normalizer"])
        train.assert_same_state(item["sampling_policy"]["normalizer"], source["normalizer"])
        previous = source if i == 0 else checkpoints[i-1]
        for name in ("actor", "critic"):
            train.assert_same_state(item["sampling_policy"][f"{name}_state_dict"], previous[f"{name}_state_dict"])
        if i:
            for name in item["rollout"]["observations"]:
                torch.testing.assert_close(item["rollout"]["observations"][name][0],
                                           previous["rollout"]["next_observations"][name][-1], rtol=0, atol=0)
        terms = record["episodes"]["rollout_reward_terms"]
        for name in penalty_names:
            assert terms[name]["mean_step_contribution"] <= 1e-9
        assert all(e["end_terms"] != ["nan_state"] for e in record["episodes"]["records"])
        rows.append({"iteration": i + 1, "optimizer_step": update["optimizer_after"]["step"],
                     "completed_episodes": record["episodes"]["completed_count"],
                     "mean_episode_return": record["episodes"]["mean_return"],
                     "loss_before": update["before"]["loss"], "loss_after": update["after"]["loss"],
                     "approx_kl": update["after"]["approx_kl"], "clip_fraction": update["after"]["clip_fraction"],
                     "normalization_inputs": record["normalization"]["sampling_inputs"],
                     "performance": record["performance"], "reward_terms": terms})
    assert audit.records and any(r["continuing_ids"] for r in audit.records)
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == source_hash
    summary = {"device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
               "num_envs": 64, "iterations": 5, "steps_per_iteration": 24,
               "total_transitions": 7680, "optimizer_steps": [0, 12, 24, 36, 48, 60],
               "normalizer_counts": checkpoints[-1]["metrics"]["normalization"]["checkpoint_counts"],
               "normalizer_frozen": True, "minibatch_alignment_verified": True,
               "penalty_signs_verified": True, "nan_state_detected": False,
               "reset_checks": len(audit.records), "reset_audits": audit.records, "iterations_detail": rows}
    (output / "validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"[64env-smoke] PASS gpu={summary['gpu']} transitions=7680 Adam=0->60 resets={len(audit.records)}", flush=True)


if __name__ == "__main__":
    main()
