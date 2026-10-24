"""Run the unchanged task for 2/4 envs from ONE fresh calibrated checkpoint.

Three 64-step iterations, two epochs, minibatches of 50. Reuse the existing
minibatch and reset auditors. No task/config/legacy checkpoint is edited.
"""

import hashlib
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
from training.tests.integration.vector_train_smoke import ResetAudit


def run_case(n, initial_path, output):
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    assert initial["optimizer_state_dict"]["state"] == {}
    before_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    directory = output / f"{n}envs"
    if directory.exists():
        raise FileExistsError(f"preserve previous results: {directory}")
    audit = ResetAudit()
    argv = ["training.train", "--checkpoint", str(initial_path), "--output-dir", str(directory),
            "--num-envs", str(n), "--iterations", "3", "--steps", "64", "--epochs", "2",
            "--minibatch-size", "50", "--normalization", "frozen"]
    with patch.object(sys, "argv", argv), \
            patch("training.train.update_ppo", side_effect=audited_update), \
            patch("training.rollout.reset_done_envs", side_effect=audit):
        train.main()
    saved = [torch.load(directory / f"iteration_{i:04d}.pt", map_location="cpu", weights_only=True)
             for i in (1, 2, 3)]
    updates = 2 * math.ceil(64 * n / 50)
    for i, item in enumerate(saved):
        record, report, batch = item["metrics"], item["metrics"]["update"], item["rollout"]
        assert report["sample_alignment_verified"] and report["sample_axes"] == [64, n]
        assert report["samples"] == 64 * n and report["optimizer_steps"] == updates
        assert report["optimizer_before"]["step"] == i * updates
        assert report["optimizer_after"]["step"] == (i + 1) * updates
        assert record["normalization"]["mode"] == "frozen"
        assert record["normalization"]["boundary_update"] == {}
        assert record["normalization"]["boundary_policy_shift"] == {}
        train.assert_same_state(item["normalizer"], initial["normalizer"])
        train.assert_same_state(item["sampling_policy"]["normalizer"], initial["normalizer"])
        previous = initial if i == 0 else saved[i-1]
        for name in ("actor", "critic"):
            train.assert_same_state(item["sampling_policy"][f"{name}_state_dict"], previous[f"{name}_state_dict"])
            assert report["parameter_changes"][name]["changed_elements"] > 0
        assert report["optimizer_after"]["exp_avg_norm"] > 0
        assert report["optimizer_after"]["exp_avg_sq_norm"] > 0
        if i:
            for name in batch["observations"]:
                torch.testing.assert_close(batch["observations"][name][0],
                                           previous["rollout"]["next_observations"][name][-1], rtol=0, atol=0)
        done = batch["terminated"] | batch["truncated"]
        delta = batch["rewards"] + .99 * batch["bootstrap_values"] - batch["values"]
        torch.testing.assert_close(item["targets"]["advantages"][done], delta[done])
        assert (batch["bootstrap_values"][batch["terminated"]] == 0).all()
        for episode in record["episodes"]["records"]:
            assert abs(sum(episode["reward_terms"].values()) - episode["return"]) < 1e-5
    assert audit.records and any(r["continuing_ids"] for r in audit.records)
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == before_hash
    result = {"num_envs": n, "iterations": 3, "steps_per_iteration": 64,
              "epochs": 2, "minibatch_size": 50, "samples_per_iteration": 64*n,
              "updates_per_iteration": updates, "last_minibatch_size": 64*n % 50,
              "optimizer_steps": [0] + [s["metrics"]["update"]["optimizer_after"]["step"] for s in saved],
              "normalizer_counts": saved[-1]["metrics"]["normalization"]["checkpoint_counts"],
              "normalizer_exactly_frozen": True, "minibatch_alignment_verified": True,
              "fresh_policy_and_persistent_optimizer_verified": True,
              "initial_checkpoint_unchanged": True,
              "completed_episodes": sum(s["metrics"]["episodes"]["completed_count"] for s in saved),
              "loss_before_after": [[s["metrics"]["update"][phase]["loss"] for phase in ("before", "after")] for s in saved],
              "normalization_inputs": [s["metrics"]["normalization"]["sampling_inputs"] for s in saved],
              "reset_audits": audit.records}
    (directory / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"[fresh-training] envs={n} Adam={result['optimizer_steps']} "
          f"normalizer_counts={result['normalizer_counts']} reset_checks={len(audit.records)} PASS", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from mjlab.envs import ManagerBasedRlEnv  # noqa: F401
    initial_path = args.checkpoint
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"preserve previous results: {output}")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    assert initial["optimizer_state_dict"]["state"] == {} and initial["iteration"] == 0
    output.mkdir(parents=True, exist_ok=False)
    results = [run_case(n, initial_path, output) for n in (2, 4)]
    summary = {"initial_checkpoint": str(initial_path.resolve()),
               "frozen_calibrated_statistics_preserved": True, "optimizer_initial_state_empty": True,
               "formal_training_started": False, "results": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("[fresh-training] COMPLETE: both short checks passed; no long training launched", flush=True)


if __name__ == "__main__":
    main()
