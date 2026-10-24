"""Adapt custom checkpoints to the ORIGINAL Microduck normalized ONNX export path.

No converted checkpoint or hand-written ONNX graph is produced. run_export() from
mjlab_microduck.export still builds the env/metadata and calls the existing
MjlabOnPolicyRunner.export_policy_to_onnx. The adapter supplies a real RSL MLPModel
containing the exact custom MLP and frozen normalization module.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from training.rollout import load_policy
from training.train import assert_same_state


def export_runner_class():
    from mjlab.envs import ManagerBasedRlEnv  # noqa: F401; preserve BAM import order
    from mjlab.rl import MjlabOnPolicyRunner
    from rsl_rl.models import MLPModel

    class InferenceAlgorithm:
        def __init__(self, actor):
            self.actor = actor

        def get_policy(self):
            return self.actor

        def eval_mode(self):
            self.actor.eval()

    class CustomCheckpointExportRunner(MjlabOnPolicyRunner):
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            # Export-only: do not construct another PPO optimizer or training loop.
            self.env, self.device = env, device

        def load(self, path, **kwargs):
            actor, critic, norm = load_policy(Path(path), self.device)
            obs = self.env.get_observations()
            model = MLPModel(obs, {"actor": ["actor"]}, "actor", actor.config["action_dim"],
                             hidden_dims=actor.config["hidden_dims"], activation="elu", obs_normalization=True,
                             distribution_cfg={"class_name": "GaussianDistribution", "init_std": .05, "std_type": "log"})
            model.mlp = copy.deepcopy(actor.mean_net)
            model.obs_normalizer = copy.deepcopy(norm.stats["actor"])
            model = model.to(self.device).eval()
            with torch.no_grad():
                model.distribution.log_std_param.copy_(actor.log_std.clamp(-5, 2))
                expected = actor(norm.stats["actor"](obs["actor"]))
                actual = model(obs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.alg = InferenceAlgorithm(model)
            self.current_learning_iteration = 0
            return {"custom_checkpoint": str(path), "normalizer_baked_in": True}

    return CustomCheckpointExportRunner


def export_checkpoint(checkpoint, output_dir):
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    if not checkpoint.is_file() or output_dir.exists():
        raise ValueError("checkpoint must exist and export output directory must be new")
    from mjlab.envs import ManagerBasedRlEnv  # noqa: F401
    import mjlab_microduck.export as original_export
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx
    from training.inspect_env import TASK_ID

    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["normalizer"]["training"] or any(state["normalizer"]["group_training"].values()):
        raise ValueError("export requires frozen normalization")
    runner_cls = export_runner_class()
    cfg = original_export.ExportConfig(checkpoint_file=str(checkpoint.resolve()), num_envs=1,
                                       device="cuda:0", onnx_file=str(output_dir / "policy.onnx"))
    with patch.object(original_export, "load_runner_cls", return_value=runner_cls):
        result = original_export.run_export(TASK_ID, cfg)
    attach_metadata_to_onnx(str(result.onnx_path), {
        "normalization_baked_in": True,
        "normalization_formula": "(raw_obs - mean) / (sqrt(var) + eps)",
        "normalization_count": int(state["normalizer"]["state_dict"]["stats.actor.count"]),
        "source_checkpoint_sha256": original_hash,
        "export_route": "mjlab_microduck.export.run_export -> MjlabOnPolicyRunner.export_policy_to_onnx"})
    report = verify_onnx(checkpoint, result.onnx_path)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("export changed source checkpoint")
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


@torch.no_grad()
def verify_onnx(checkpoint, onnx_path):
    import numpy as np
    import onnx
    import onnxruntime as ort

    checkpoint, onnx_path = Path(checkpoint), Path(onnx_path)
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    ops = {node.op_type for node in graph.graph.node}
    if not {"Sub", "Div"} <= ops:
        raise RuntimeError("export graph lacks observation normalization operations")
    actor, critic, norm = load_policy(checkpoint, "cpu")
    frozen = norm.checkpoint()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw = state["rollout"]["observations"]["actor"].reshape(-1, actor.config["obs_dim"])
    indices = torch.linspace(0, len(raw) - 1, steps=min(256, len(raw))).long()
    raw = raw[indices].float()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
        raise RuntimeError("expected raw 61D observations -> 14D actions")
    expected, exported, unnormalized = [], [], []
    for row in raw.split(1):
        expected.append(actor(norm.stats["actor"](row)).numpy())
        unnormalized.append(actor(row).numpy())
        exported.append(session.run(None, {session.get_inputs()[0].name: row.numpy()})[0])
    expected, exported, unnormalized = map(np.concatenate, (expected, exported, unnormalized))
    if not np.isfinite(exported).all():
        raise RuntimeError("ONNX output contains NaN or Inf")
    np.testing.assert_allclose(exported, expected, rtol=1e-5, atol=1e-6)
    actor_only_difference = float(np.max(np.abs(expected - unnormalized)))
    if actor_only_difference < 1e-5:
        raise RuntimeError("validation inputs cannot distinguish missing normalization")
    assert_same_state(norm.checkpoint(), frozen)
    return {"onnx_path": str(onnx_path.resolve()), "source_checkpoint": str(checkpoint.resolve()),
            "route": "original Microduck run_export and mjlab runner ONNX exporter",
            "input": [1, 61], "output": [1, 14], "raw_observation_inputs": True,
            "normalization_baked_in": True, "normalizer_frozen": True,
            "comparison_samples": len(raw), "max_abs_error": float(np.max(np.abs(exported - expected))),
            "mean_abs_error": float(np.mean(np.abs(exported - expected))),
            "missing_normalizer_max_abs_difference": actor_only_difference,
            "onnxruntime_provider": "CPUExecutionProvider", "rtol": 1e-5, "atol": 1e-6,
            "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = export_checkpoint(args.checkpoint, args.output_dir)
    print("[export] PASS " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
