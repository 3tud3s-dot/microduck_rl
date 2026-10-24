"""Fresh weights/Adam, calibrated stats only, portable checkpoint and no overwrite."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from training.advantages import compute_gae
from training.initialize import checkpoint_payload, forward_probe, initialize_models, main
from training.normalization import ObservationNormalizer
from training.ppo import create_optimizer, optimizer_status, update_ppo
from training.rollout import load_policy
from training.train import assert_same_state


class InitializationTests(unittest.TestCase):
    def artifact(self):
        torch.manual_seed(11)
        raw = {"actor": torch.randn(3, 4, 61), "critic": torch.randn(3, 4, 76) * 3}
        norm = ObservationNormalizer()
        norm.update(raw)
        norm.eval()
        return {"artifact_type": "observation_normalization_calibration", "normalizer": norm.checkpoint()}, raw

    def test_seeded_fresh_cpu_models_empty_adam_and_rng_is_preserved(self):
        artifact, raw = self.artifact()
        # Even if someone bundles weights/Adam in the file, only stats are read.
        artifact["actor_state_dict"] = {"poison": torch.tensor(float("nan"))}
        artifact["optimizer_state_dict"] = {"state": "must never load this"}
        rng = torch.get_rng_state().clone()
        a, c, norm, opt = initialize_models(artifact, seed=2026, hidden_dims=(8, 8))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        a2, c2, _, _ = initialize_models(artifact, seed=2026, hidden_dims=(8, 8))
        a3, c3, _, _ = initialize_models(artifact, seed=2027, hidden_dims=(8, 8))
        assert_same_state(a.state_dict(), a2.state_dict())
        assert_same_state(c.state_dict(), c2.state_dict())
        self.assertTrue(any(not torch.equal(v, a3.state_dict()[k]) for k, v in a.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, c3.state_dict()[k]) for k, v in c.state_dict().items()))
        self.assertEqual(opt.state_dict()["state"], {})
        self.assertEqual(optimizer_status(opt)["step"], 0)
        self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32 and p.grad is None
                            for model in (a, c) for p in model.parameters()))
        assert_same_state(norm.checkpoint(), artifact["normalizer"])
        self.assertTrue(forward_probe(a, c, norm, raw)["normalizer_unchanged"])
        self.assertEqual(opt.state_dict()["state"], {})
        artifact["normalizer"]["state_dict"]["stats.actor.mean"].fill_(999)
        self.assertFalse(torch.equal(norm.stats["actor"].mean, artifact["normalizer"]["state_dict"]["stats.actor.mean"]))

    def test_checkpoint_roundtrip_and_first_ppo_updates_allocate_new_moments(self):
        artifact, raw = self.artifact()
        actor, critic, norm, optimizer = initialize_models(artifact, hidden_dims=(8,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.pt"
            torch.save(checkpoint_payload(actor, critic, norm, optimizer, {"seed": 2026}), path)
            restored_actor, restored_critic, restored_norm = load_policy(path, "cpu")
            restored_optimizer = create_optimizer(restored_actor, restored_critic)
            restored_optimizer.load_state_dict(torch.load(path, weights_only=True)["optimizer_state_dict"])
            self.assertEqual(restored_optimizer.state_dict()["state"], {})
            with torch.no_grad():
                normalized = norm(raw)
                actions, log_prob = actor.sample(normalized["actor"])
                values = critic(normalized["critic"])
            flags = torch.zeros(3, 4, dtype=torch.bool)
            rewards = torch.arange(12).float().reshape(3, 4) * .01
            targets = compute_gae(rewards, values, values, flags, flags)
            batch = {"observations": raw, "normalized_observations": normalized,
                     "actions": actions, "log_prob": log_prob, "values": values}
            initial_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            for a, c, n, opt in ((actor, critic, norm, optimizer),
                                 (restored_actor, restored_critic, restored_norm, restored_optimizer)):
                report, _ = update_ppo(a, c, n, batch, targets, optimizer=opt, epochs=2,
                                       minibatch_size=5, generator=torch.Generator().manual_seed(17))
                self.assertEqual(report["optimizer_before"]["step"], 0)
                self.assertEqual(report["optimizer_after"]["step"], 6)
                self.assertGreater(report["optimizer_after"]["exp_avg_norm"], 0)
                assert_same_state(n.checkpoint(), artifact["normalizer"])
            assert_same_state(actor.state_dict(), restored_actor.state_dict())
            assert_same_state(critic.state_dict(), restored_critic.state_dict())
            assert_same_state(optimizer.state_dict(), restored_optimizer.state_dict())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), initial_hash)

    def test_invalid_calibration_counts_modes_and_moments_are_rejected(self):
        artifact, _ = self.artifact()
        bad = []
        x = copy.deepcopy(artifact); x["artifact_type"] = "trained_policy"; bad.append(x)
        x = copy.deepcopy(artifact); x["normalizer"]["training"] = True; bad.append(x)
        x = copy.deepcopy(artifact); x["normalizer"]["group_training"]["critic"] = True; bad.append(x)
        for key, value in (("stats.actor.count", torch.tensor(0)),
                           ("stats.actor.count", torch.tensor(12.)),
                           ("stats.actor.count", torch.tensor(13)),
                           ("stats.actor.mean", torch.full((61,), float("nan"))),
                           ("stats.critic.var", torch.full((76,), -1.))):
            x = copy.deepcopy(artifact)
            x["normalizer"]["state_dict"][key] = value
            bad.append(x)
        for value in bad:
            with self.assertRaises(ValueError):
                initialize_models(value)
        for kwargs in ({"seed": -1}, {"seed": True}, {"learning_rate": 0},
                       {"init_std": float("nan")}, {"hidden_dims": (0,)}):
            with self.assertRaises(ValueError):
                initialize_models(artifact, **kwargs)

    def test_cli_preserves_source_and_refuses_existing_output(self):
        artifact, raw = self.artifact()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "normalizer.pt", root / "fresh"
            torch.save(artifact, source)
            torch.save({"holdout": {"observations": raw}}, root / "observations.pt")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            argv = ["training.initialize", "--calibration", str(source), "--output-dir", str(output),
                    "--seed", "2026", "--hidden-dims", "8"]
            with patch.object(sys, "argv", argv):
                main()
                report = json.loads((output / "initialization.json").read_text())
                self.assertTrue(report["optimizer_state_empty"])
                self.assertTrue(report["checkpoint_restore_verified"])
                self.assertEqual(report["normalizer_counts"], {"actor": 12, "critic": 12})
                initial = (output / "initial.pt").read_bytes()
                with self.assertRaises(SystemExit):
                    main()
                self.assertEqual((output / "initial.pt").read_bytes(), initial)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
