"""Initialize new CPU networks and empty Adam from calibrated observation stats.

Only normalization is imported from the artifact. Network weights are generated
from the initialization seed, and Adam starts at step zero with no moments.
The checkpoint is directly usable by training.train/evaluate/play/load_policy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from training.inspect_env import ACTION_DIM, OBS_DIMS, TASK_ID
from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer
from training.ppo import create_optimizer, optimizer_status
from training.rollout import load_policy
from training.train import assert_same_state, cpu_snapshot


def initialize_models(calibration, *, seed=2026, hidden_dims=(64, 64), init_std=.05, learning_rate=1e-5):
    """Create fresh FP32 CPU models; preserve the caller's CPU random stream."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if calibration.get("artifact_type") != "observation_normalization_calibration":
        raise ValueError("expected a standalone observation normalization calibration artifact")
    normalizer = ObservationNormalizer.from_checkpoint(calibration["normalizer"])
    if {name: stats.dim for name, stats in normalizer.stats.items()} != OBS_DIMS:
        raise ValueError("expected calibrated actor 61D / critic 76D observations")
    if normalizer.training or any(stats.training for stats in normalizer.stats.values()):
        raise ValueError("calibrated statistics must already be frozen")
    counts = []
    for name, stats in normalizer.stats.items():
        raw_count = calibration["normalizer"]["state_dict"][f"stats.{name}.count"]
        if raw_count.dtype != torch.long or raw_count.shape != () or stats.count.item() < 1:
            raise ValueError("calibrated counts must be positive integer scalars")
        if (not torch.isfinite(stats.mean).all() or not torch.isfinite(stats.var).all()
                or (stats.var < 0).any()):
            raise ValueError("calibrated moments must be finite with nonnegative variance")
        counts.append(stats.count.item())
    if len(set(counts)) != 1:
        raise ValueError("actor and critic must have the same calibration sample count")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(init_std) or init_std <= 0:
        raise ValueError("init_std must be finite and positive")
    # Seed CPU initialization only; do not alter a caller's CUDA RNG streams.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        actor = GaussianActor(obs_dim=OBS_DIMS["actor"], action_dim=ACTION_DIM,
                              hidden_dims=tuple(hidden_dims), init_std=init_std).float().eval()
        critic = Critic(obs_dim=OBS_DIMS["critic"], hidden_dims=tuple(hidden_dims)).float().eval()
    optimizer = create_optimizer(actor, critic, learning_rate)
    if optimizer.state_dict()["state"] or optimizer_status(optimizer)["step"] != 0:
        raise RuntimeError("new Adam must have no inherited state")
    if any(p.grad is not None or not torch.isfinite(p).all() for m in (actor, critic) for p in m.parameters()):
        raise RuntimeError("new models must be finite and have no gradients")
    assert_same_state(normalizer.checkpoint(), calibration["normalizer"])
    return actor, critic, normalizer, optimizer


def checkpoint_payload(actor, critic, normalizer, optimizer, metadata):
    """Flat model keys match the existing training/loading entrypoints."""
    return cpu_snapshot({"artifact_type": "fresh_calibrated_policy",
                         "actor_config": actor.config, "critic_config": critic.config,
                         "actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
                         "actor_training": False, "critic_training": False,
                         "normalizer": normalizer.checkpoint(), "optimizer_state_dict": optimizer.state_dict(),
                         "iteration": 0, "metadata": metadata})


@torch.no_grad()
def forward_probe(actor, critic, normalizer, raw):
    before = normalizer.checkpoint()
    normalized = normalizer(raw)
    distribution = actor.distribution(normalized["actor"])
    values = critic(normalized["critic"])
    log_prob, entropy = actor.evaluate_actions(normalized["actor"], distribution.mean)
    for tensor in (distribution.mean, distribution.stddev, values, log_prob, entropy):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("non-finite forward result for calibrated inputs")
    assert_same_state(normalizer.checkpoint(), before)
    return {"sample_axes": list(values.shape), "actions": list(distribution.mean.shape),
            "action_mean_max_abs": distribution.mean.abs().max().item(),
            "action_std_min": distribution.stddev.min().item(), "action_std_max": distribution.stddev.max().item(),
            "value_max_abs": values.abs().max().item(), "normalizer_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from training.paths import INITIAL_CHECKPOINT
    parser.add_argument("--calibration", type=Path, default=INITIAL_CHECKPOINT,
                        help="frozen calibration artifact or checkpoint; only normalization statistics are read")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--init-std", type=float, default=.05)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if not args.calibration.is_file():
        parser.error("--calibration must be an existing calibration artifact")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    digest = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    artifact = torch.load(args.calibration, map_location="cpu", weights_only=True)
    if "actor_config" in artifact and "normalizer" in artifact:
        artifact = {"artifact_type": "observation_normalization_calibration", "normalizer": artifact["normalizer"]}
    actor, critic, normalizer, optimizer = initialize_models(
        artifact, seed=args.seed, hidden_dims=tuple(args.hidden_dims), init_std=args.init_std, learning_rate=args.lr)
    # Offline holdout forward only; no simulation, sampling, backward or Adam step.
    observations_path = args.calibration.parent / "observations.pt"
    if observations_path.is_file():
        data = torch.load(observations_path, map_location="cpu", weights_only=True)
        raw = {name: x[:32].clone() for name, x in data["holdout"]["observations"].items()}
        probe_source = "first 32 holdout time steps; no statistics updates"
    else:
        raw = {name: stats.mean.float().unsqueeze(0).repeat(4, 1) for name, stats in normalizer.stats.items()}
        probe_source = "synthetic observations at calibrated means; holdout file unavailable"
    probe = forward_probe(actor, critic, normalizer, raw)
    metadata = {"task_id": TASK_ID, "initialization_seed": args.seed, "initialization_device": "cpu",
                "calibration": str(args.calibration.resolve()), "calibration_sha256": digest,
                "normalizer_counts": {name: stats.count.item() for name, stats in normalizer.stats.items()},
                "normalization_frozen": True, "network_weights": "new random initialization",
                "optimizer": "new Adam, lazy moments allocated on first update", "optimizer_initial_step": 0,
                "learning_rate": args.lr, "hidden_dims": args.hidden_dims, "init_std": args.init_std,
                "training_iterations": 0, "simulator_state_saved": False,
                "probe_source": probe_source, "probe": probe}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / "initial.pt"
    torch.save(checkpoint_payload(actor, critic, normalizer, optimizer, metadata), checkpoint)
    restored_actor, restored_critic, restored_norm = load_policy(checkpoint, "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored_optimizer = create_optimizer(restored_actor, restored_critic, args.lr)
    restored_optimizer.load_state_dict(saved["optimizer_state_dict"])
    for current, reference in ((restored_actor.state_dict(), actor.state_dict()),
                               (restored_critic.state_dict(), critic.state_dict()),
                               (restored_norm.checkpoint(), normalizer.checkpoint()),
                               (restored_optimizer.state_dict(), optimizer.state_dict())):
        assert_same_state(current, reference)
    restored_probe = forward_probe(restored_actor, restored_critic, restored_norm, raw)
    assert_same_state(restored_probe, probe)
    assert not restored_optimizer.state_dict()["state"]
    if hashlib.sha256(args.calibration.read_bytes()).hexdigest() != digest:
        raise RuntimeError("calibration source changed")
    (args.output_dir / "initialization.json").write_text(json.dumps(
        {**metadata, "checkpoint_restore_verified": True, "optimizer_state_empty": True}, indent=2, allow_nan=False) + "\n")
    print(f"[initialize] new actor/critic hidden_dims={args.hidden_dims} seed={args.seed} "
          f"normalizer_counts={metadata['normalizer_counts']} frozen=PASS Adam_step=0 moments=empty")
    print(f"[initialize] forward/restore=PASS checkpoint={checkpoint.resolve()}")
    print("[initialize] No simulation or parameter updates; calibration and old models unchanged")


if __name__ == "__main__":
    main()
