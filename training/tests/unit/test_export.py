"""Exact custom normalization travels through the original mjlab ONNX exporter."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from training.export import export_checkpoint, export_runner_class, verify_onnx
from training.models import GaussianActor, Critic
from training.normalization import ObservationNormalizer
from training.train import assert_same_state


class ExportTests(unittest.TestCase):
    def test_original_runner_export_bakes_exact_normalizer_and_matches_ort(self):
        torch.manual_seed(8)
        raw = {"actor": torch.randn(4, 2, 61) * 2 + 3, "critic": torch.randn(4, 2, 76)}
        norm = ObservationNormalizer(eps=.03)
        norm.update(raw)
        norm.eval()
        actor, critic = GaussianActor(hidden_dims=(8,)).eval(), Critic(hidden_dims=(8,)).eval()
        source = {"actor_config": actor.config, "critic_config": critic.config,
                  "actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
                  "normalizer": norm.checkpoint(), "rollout": {"observations": raw}}
        obs = TensorDict({name: values[0, :1] for name, values in raw.items()}, batch_size=[1])
        env = SimpleNamespace(get_observations=lambda: obs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save(source, checkpoint)
            cls = export_runner_class()
            from mjlab.rl import MjlabOnPolicyRunner
            self.assertIs(cls.export_policy_to_onnx, MjlabOnPolicyRunner.export_policy_to_onnx)
            with patch("torch.optim.Adam", side_effect=AssertionError("export must not create optimizer")):
                runner = cls(env, {}, device="cpu")
                runner.load(checkpoint)
                actual = runner.get_inference_policy("cpu")
                with torch.no_grad():
                    torch.testing.assert_close(actual(obs), actor(norm.stats["actor"](obs["actor"])), rtol=0, atol=0)
                runner.export_policy_to_onnx(str(root), "policy.onnx")
            report = verify_onnx(checkpoint, root / "policy.onnx")
            self.assertTrue(report["normalization_baked_in"])
            self.assertLess(report["max_abs_error"], 1e-6)
            self.assertGreater(report["missing_normalizer_max_abs_difference"], 1e-5)
            assert_same_state(norm.checkpoint(), source["normalizer"])

    def test_existing_output_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint.pt"
            source.write_bytes(b"preserve")
            with self.assertRaises(ValueError):
                export_checkpoint(source, root)
            self.assertEqual(source.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
