"""Offline-fit isolation, no training, independent holdout and stability diagnostics."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from training.calibrate_normalization import (feature_profile, moment_change, policy_difference,
                                              run_calibration, stability_summary)
from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer
from training.tests.unit.test_vector_reset import HistoryEnv
from training.train import assert_same_state, cpu_snapshot


class CalibrationTests(unittest.TestCase):
    def components(self):
        torch.manual_seed(9)
        actor = GaussianActor(obs_dim=1, action_dim=1, hidden_dims=(8,)).eval()
        critic = Critic(obs_dim=3, hidden_dims=(8,)).eval()
        norm = ObservationNormalizer({"actor": 1, "critic": 3})
        norm.update({"actor": torch.randn(6, 1), "critic": torch.randn(6, 3)})
        return actor, critic, norm.eval()

    def test_fresh_statistics_raw_inputs_once_holdout_excluded_models_frozen(self):
        actor, critic, norm = self.components()
        before = cpu_snapshot({"actor": actor.state_dict(), "critic": critic.state_dict(), "normalizer": norm.checkpoint()})
        created, records = [], []

        def make_env(seed):
            env = HistoryEnv(2)
            env.step_dt = .02
            env.closed = False
            env.close = lambda: setattr(env, "closed", True)
            created.append((seed, env))
            return env

        with patch("torch.optim.Adam", side_effect=AssertionError("no optimizer")), \
                patch.object(torch.Tensor, "backward", side_effect=AssertionError("no backward")):
            candidate, report, data = run_calibration(
                make_env, actor, critic, norm, fit_seeds=(11, 12), holdout_seed=99,
                batches_per_seed=3, steps_per_batch=4, holdout_steps=5, on_batch=records.append)
        self.assertEqual([seed for seed, env in created], [11, 12, 99])
        self.assertTrue(all(env.closed for seed, env in created))
        self.assertEqual([env.t for seed, env in created], [12, 12, 5])
        self.assertEqual(len(records), 7)
        self.assertEqual(report["fit_samples_per_group"], 48)
        self.assertEqual(report["holdout_samples_per_group"], 10)
        self.assertEqual(report["source_counts"], {"actor": 6, "critic": 6})
        self.assertEqual(report["calibrated_counts"], {"actor": 48, "critic": 48})
        self.assertEqual(len(report["successive_calibration_policy_changes_on_holdout"]), 5)
        for name, stats in candidate.stats.items():
            raw = data["fit"]["observations"][name].double().reshape(-1, stats.dim)
            torch.testing.assert_close(stats.mean, raw.mean(0), rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(stats.var, raw.var(0, correction=0), rtol=1e-10, atol=1e-12)
            self.assertFalse(stats.training)
        self.assertFalse(candidate.training)
        restored = ObservationNormalizer.from_checkpoint(candidate.checkpoint())
        assert_same_state(restored.checkpoint(), candidate.checkpoint())
        assert_same_state({"actor": actor.state_dict(), "critic": critic.state_dict(), "normalizer": norm.checkpoint()}, before)
        self.assertFalse(report["optimizer_created"])
        self.assertTrue(report["holdout_excluded_from_fit"])
        self.assertEqual(data["segments"][3]["seed"], 12)
        self.assertFalse(data["segments"][3]["continued_from_previous_batch"])

    def test_stability_requires_comparable_consecutive_batches_and_detects_drift(self):
        norm = ObservationNormalizer({"actor": 1, "critic": 3}).eval()
        empty = norm.checkpoint()
        norm.train()
        norm.update({"actor": torch.tensor([[0.], [2.]]), "critic": torch.tensor([[0., 1., 2.], [2., 3., 4.]])})
        norm.eval()
        fitted = norm.checkpoint()
        first, same = moment_change(empty, fitted), moment_change(fitted, fitted)
        self.assertFalse(first["actor"]["comparable"])
        self.assertFalse(stability_summary([same, same])["stable"])
        self.assertTrue(stability_summary([first, same, same, same])["stable"])
        shifted = cpu_snapshot(fitted)
        shifted["state_dict"]["stats.actor.mean"] += 100
        drift = moment_change(fitted, shifted)
        self.assertFalse(stability_summary([same, same, drift])["stable"])

    def test_policy_kl_has_correct_direction_units_and_does_not_sample(self):
        actor = GaussianActor(obs_dim=1, action_dim=1, hidden_dims=(), init_std=.5).eval()
        critic = Critic(obs_dim=3, hidden_dims=()).eval()
        with torch.no_grad():
            actor.mean_net[0].weight.fill_(2)
            actor.mean_net[0].bias.zero_()
        old = ObservationNormalizer({"actor": 1, "critic": 3}).eval()
        new = ObservationNormalizer.from_checkpoint(old.checkpoint())
        new.stats["actor"].mean.fill_(2)
        new.stats["actor"].var.fill_(4)
        obs = {"actor": torch.tensor([[[0.], [1.]], [[2.], [3.]]]), "critic": torch.zeros(2, 2, 3)}
        rng = torch.get_rng_state().clone()
        with patch.object(actor, "sample", side_effect=AssertionError("no samples")):
            identical = policy_difference(actor, critic, old, old, obs)
            change = policy_difference(actor, critic, old, new, obs)
        self.assertEqual(identical["gaussian_kl_mean"], 0)
        with torch.no_grad():
            delta = actor(old(obs)["actor"]) - actor(new(obs)["actor"])
            expected = delta.square().sum(-1).mean() / (2 * .5**2)
        self.assertAlmostEqual(change["gaussian_kl_mean"], expected.item(), places=4)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_command_profile_respects_shared_61d_layout(self):
        actor, critic = torch.zeros(2, 2, 61), torch.zeros(2, 2, 76)
        actor[..., 48:61] = torch.arange(4).reshape(2, 2, 1)
        norm = ObservationNormalizer()
        norm.update({"actor": actor, "critic": critic})
        norm.eval()
        profile = feature_profile(norm, {"actor": actor, "critic": critic})
        self.assertEqual(profile["command_block"]["varying_dimensions"], 13)
        self.assertEqual(profile["command_block"]["exact_zero_vector_fraction"], .25)
        self.assertEqual(profile["actor"]["near_constant_feature_indices"], list(range(48)))

    def test_invalid_inputs_do_not_create_an_env_and_failures_close_it(self):
        actor, critic, norm = self.components()
        factory = unittest.mock.Mock(side_effect=AssertionError("must not create env"))
        for kwargs in ({"fit_seeds": ()}, {"fit_seeds": (1, 1)}, {"holdout_seed": 42},
                       {"fit_seeds": (-1,)}, {"steps_per_batch": 0}, {"batches_per_seed": True},
                       {"mean_tolerance": float("nan")}, {"stable_batches": 0}):
            with self.assertRaises(ValueError):
                run_calibration(factory, actor, critic, norm, **kwargs)
        self.assertEqual(factory.call_count, 0)
        env = HistoryEnv(2)
        env.closed = False
        env.close = lambda: setattr(env, "closed", True)
        with self.assertRaisesRegex(RuntimeError, "probe"):
            run_calibration(lambda seed: env, actor, critic, norm, fit_seeds=(1,), holdout_seed=2,
                            batches_per_seed=1, steps_per_batch=2, holdout_steps=2,
                            on_batch=lambda record: (_ for _ in ()).throw(RuntimeError("probe")))
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
