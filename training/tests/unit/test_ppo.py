"""PPO clipping branches, fixed targets, one optimizer step and input provenance."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from training.advantages import compute_gae
from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer
from training.ppo import create_optimizer, load_inputs, optimizer_status, ppo_loss, update_once, update_ppo


class PpoTests(unittest.TestCase):
    def test_clipped_objective_for_positive_and_negative_advantages(self):
        log_prob = torch.tensor([1.5, .5, 1.5, .5]).log().requires_grad_()
        old = torch.zeros(4, requires_grad=True)
        values = torch.tensor([0., 1., 2., 3.], requires_grad=True)
        returns = torch.ones(4, requires_grad=True)
        advantages = torch.tensor([2., 2., -2., -2.], requires_grad=True)
        entropy = torch.tensor([.2, .4, .6, .8], requires_grad=True)
        result = ppo_loss(log_prob, old, values, returns, advantages, entropy, entropy_coef=.1)
        # min(ratio*A, clipped_ratio*A) = [2.4, 1.0, -3.0, -1.6].
        self.assertAlmostEqual(result["policy_loss"].item(), .3, places=6)
        self.assertAlmostEqual(result["value_loss"].item(), 1.5, places=6)
        self.assertAlmostEqual(result["loss"].item(), 1., places=6)
        self.assertEqual(result["clip_fraction"].item(), 1.)
        result["loss"].backward()
        torch.testing.assert_close(log_prob.grad, torch.tensor([0., -.25, .75, 0.]))
        self.assertIsNone(old.grad)
        self.assertIsNone(returns.grad)
        self.assertIsNone(advantages.grad)
        self.assertIsNotNone(values.grad)
        torch.testing.assert_close(entropy.grad, torch.full((4,), -.025))

    @staticmethod
    def make_batch():
        torch.manual_seed(29)
        actor = GaussianActor(obs_dim=3, action_dim=2, hidden_dims=(8,)).eval()
        critic = Critic(obs_dim=5, hidden_dims=(8,)).eval()
        raw = {"actor": torch.randn(4, 2, 3), "critic": torch.randn(4, 2, 5)}
        norm = ObservationNormalizer({"actor": 3, "critic": 5})
        norm.update(raw)
        norm.eval()
        with torch.no_grad():
            normalized = norm(raw)
            actions, log_prob = actor.sample(normalized["actor"])
            values = critic(normalized["critic"])
        batch = {"observations": raw, "normalized_observations": normalized,
                 "actions": actions, "log_prob": log_prob, "values": values}
        targets = {"advantages": torch.linspace(-1, 2, 8).reshape(4, 2), "returns": values + .5}
        return actor, critic, norm, batch, targets

    def test_exactly_one_step_changes_both_networks_and_preserves_fixed_data(self):
        actor, critic, norm, batch, targets = self.make_batch()
        original = copy.deepcopy((batch, targets, norm.checkpoint()))
        report, optimizer = update_once(actor, critic, norm, batch, targets, max_grad_norm=.01)
        self.assertEqual(report["samples"], 8)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertLessEqual(report["total_grad_norm_after"], .0100001)
        self.assertGreater(report["total_grad_norm_before"], .01)
        for name in ("actor", "critic"):
            self.assertGreater(report["parameter_changes"][name]["l2"], 0)
            self.assertGreater(report["parameter_changes"][name]["changed_elements"], 0)
            self.assertGreater(report["gradients"][name]["before_clip_norm"], 0)
        for state in optimizer["state"].values():
            self.assertEqual(state["step"].item(), 1)
            self.assertTrue(torch.isfinite(state["exp_avg"]).all())
            self.assertTrue(torch.isfinite(state["exp_avg_sq"]).all())
        for name in ("before", "after"):
            self.assertTrue(all(torch.isfinite(torch.tensor(x)).item() for x in report[name].values()))
        self.assertAlmostEqual(report["before"]["ratio_min"], 1., places=5)
        self.assertAlmostEqual(report["before"]["ratio_max"], 1., places=5)
        for saved, current in zip(original[:2], (batch, targets)):
            for name, x in saved.items():
                if isinstance(x, dict):
                    for group, value in x.items():
                        self.assertTrue(torch.equal(value, current[name][group]))
                else:
                    self.assertTrue(torch.equal(x, current[name]))
        for name, value in original[2]["state_dict"].items():
            self.assertTrue(torch.equal(value, norm.state_dict()[name]))
        self.assertFalse(norm.training)
        self.assertTrue(all(not x.training and x.count.item() == 8 for x in norm.stats.values()))

    def test_unfrozen_or_mismatched_sampling_policy_is_rejected_before_update(self):
        actor, critic, norm, batch, targets = self.make_batch()
        original = copy.deepcopy(actor.state_dict())
        norm.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            update_once(actor, critic, norm, batch, targets)
        norm.eval()
        norm.stats["critic"].train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            update_once(actor, critic, norm, batch, targets)
        norm.eval()
        batch["log_prob"] = batch["log_prob"] + .1
        with self.assertRaises(AssertionError):
            update_once(actor, critic, norm, batch, targets)
        for name, value in original.items():
            self.assertTrue(torch.equal(value, actor.state_dict()[name]))

    def test_loss_rejects_broadcasting_nonfinite_and_overflow(self):
        x = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "broadcasting"):
            ppo_loss(x, x[:, None], x, x, x, x)
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            ppo_loss(x, x, x, x, x * float("nan"), x)
        with self.assertRaisesRegex(ValueError, "overflowed"):
            ppo_loss(x * 1000, x, x, x, x, x)
        for kwargs in ({"clip_ratio": 0}, {"value_coef": -1}, {"entropy_coef": float("nan")}):
            with self.assertRaises(ValueError):
                ppo_loss(x, x, x, x, x, x, **kwargs)

    def test_nonfinite_gradient_prevents_optimizer_step(self):
        actor, critic, norm, batch, targets = self.make_batch()
        originals = [copy.deepcopy(module.state_dict()) for module in (actor, critic, norm)]
        hook = actor.log_std.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        try:
            with self.assertRaisesRegex(RuntimeError, "non-finite gradient"):
                update_once(actor, critic, norm, batch, targets)
        finally:
            hook.remove()
        for module, original in zip((actor, critic, norm), originals):
            for name, value in original.items():
                self.assertTrue(torch.equal(value, module.state_dict()[name]))

    def test_reused_adam_retains_steps_and_both_moments(self):
        actor, critic, norm, batch, targets = self.make_batch()
        optimizer = create_optimizer(actor, critic)
        self.assertEqual(optimizer_status(optimizer)["step"], 0)
        parameters = [*actor.parameters(), *critic.parameters()]
        for iteration in range(1, 4):
            with torch.no_grad():
                batch["actions"], batch["log_prob"] = actor.sample(batch["normalized_observations"]["actor"])
                batch["values"] = critic(batch["normalized_observations"]["critic"])
                targets["returns"] = batch["values"] + .5
            previous = {id(p): {key: optimizer.state[p][key].clone() if p in optimizer.state
                               else torch.zeros_like(p) for key in ("exp_avg", "exp_avg_sq")}
                        for p in parameters}
            report, _ = update_once(actor, critic, norm, batch, targets, optimizer=optimizer)
            self.assertEqual(report["optimizer_before"]["step"], iteration - 1)
            self.assertEqual(report["optimizer_after"]["step"], iteration)
            for p in parameters:
                old, state = previous[id(p)], optimizer.state[p]
                # Adam must carry the old moments, rather than initialize at zero
                # each batch. p.grad here is the clipped gradient used by step().
                torch.testing.assert_close(state["exp_avg"], .9 * old["exp_avg"] + .1 * p.grad)
                torch.testing.assert_close(state["exp_avg_sq"], .999 * old["exp_avg_sq"] + .001 * p.grad.square())

    def test_optimizer_must_belong_to_current_networks_and_own_learning_rate(self):
        actor, critic, norm, batch, targets = self.make_batch()
        wrong = create_optimizer(GaussianActor(obs_dim=3, action_dim=2), critic)
        with self.assertRaisesRegex(ValueError, "exactly this actor"):
            update_once(actor, critic, norm, batch, targets, optimizer=wrong)
        optimizer = create_optimizer(actor, critic)
        with self.assertRaisesRegex(ValueError, "param_groups"):
            update_once(actor, critic, norm, batch, targets, optimizer=optimizer, learning_rate=1e-4)

    def test_epochs_keep_fixed_targets_and_cover_nondivisible_minibatches(self):
        actor, critic, norm, batch, targets = self.make_batch()
        original = copy.deepcopy((batch, targets, norm.checkpoint()))
        calls = []

        def capture_loss(log_prob, old_log_prob, values, returns, advantages, entropy, **kwargs):
            calls.append({"old_log_prob": old_log_prob.detach().clone(),
                          "returns": returns.detach().clone(), "advantages": advantages.detach().clone(),
                          "log_prob": log_prob.detach().clone()})
            return ppo_loss(log_prob, old_log_prob, values, returns, advantages, entropy, **kwargs)

        with patch("training.ppo.ppo_loss", side_effect=capture_loss):
            report, state = update_ppo(actor, critic, norm, batch, targets, epochs=3,
                                       minibatch_size=3, generator=torch.Generator().manual_seed(151))
        self.assertEqual(report["optimizer_steps"], 9)
        self.assertEqual([x["optimizer_step"] for x in report["minibatch_reports"]], list(range(1, 10)))
        self.assertTrue(report["fixed_targets_verified"])
        self.assertTrue(all(x["step"].item() == 9 for x in state["state"].values()))
        reference = {"old_log_prob": original[0]["log_prob"].flatten(),
                     "returns": original[1]["returns"].flatten()}
        raw = original[1]["advantages"].flatten()
        reference["advantages"] = (raw - raw.mean()) / (raw.std(correction=0) + 1e-8)
        cursor = 0

        def check_call(indices):
            nonlocal cursor
            for name, value in reference.items():
                torch.testing.assert_close(calls[cursor][name], value[indices], rtol=0, atol=0)
            cursor += 1

        check_call(torch.arange(8))  # Full-batch diagnostics before any update.
        orders = []
        for epoch in report["epoch_reports"]:
            order = torch.tensor(epoch["sample_order"])
            orders.append(epoch["sample_order"])
            self.assertEqual(sorted(epoch["sample_order"]), list(range(8)))
            self.assertEqual(epoch["minibatch_sizes"], [3, 3, 2])
            self.assertEqual(epoch["samples_seen"], 8)
            self.assertTrue(epoch["coverage_verified"])
            for start in range(0, 8, 3):
                check_call(order[start:start + 3])
            check_call(torch.arange(8))  # After this epoch, with the SAME old targets.
        self.assertEqual(cursor, len(calls))
        self.assertNotEqual(orders[0], orders[1])
        self.assertTrue(any(not torch.equal(x["log_prob"], x["old_log_prob"]) for x in calls[1:]))
        for key in ("actions", "log_prob", "values"):
            self.assertTrue(torch.equal(batch[key], original[0][key]))
        for key in ("advantages", "returns"):
            self.assertTrue(torch.equal(targets[key], original[1][key]))
        for key, value in original[2]["state_dict"].items():
            self.assertTrue(torch.equal(norm.state_dict()[key], value))
        self.assertFalse(norm.training)
        self.assertTrue(all(x["total_grad_norm_after"] <= .500001 for x in report["minibatch_reports"]))

    def test_minibatch_boundaries_and_invalid_epoch_settings(self):
        for size, steps in ((1, 16), (8, 2), (99, 2), (None, 2)):
            with self.subTest(minibatch_size=size):
                actor, critic, norm, batch, targets = self.make_batch()
                report, _ = update_ppo(actor, critic, norm, batch, targets, epochs=2,
                                       minibatch_size=size, generator=torch.Generator().manual_seed(7))
                self.assertEqual(report["optimizer_steps"], steps)
                self.assertTrue(all(epoch["samples_seen"] == 8 for epoch in report["epoch_reports"]))
        actor, critic, norm, batch, targets = self.make_batch()
        optimizer = create_optimizer(actor, critic)
        for kwargs in ({"epochs": 0}, {"epochs": 1.5}, {"epochs": True},
                       {"minibatch_size": 0}, {"minibatch_size": 2.5}, {"minibatch_size": True}):
            with self.assertRaises(ValueError):
                update_ppo(actor, critic, norm, batch, targets, optimizer=optimizer, **kwargs)
        self.assertEqual(optimizer_status(optimizer)["step"], 0)

    def test_gae_path_and_numeric_targets_must_match_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            rollout_path, targets_path = Path(directory) / "rollout.pt", Path(directory) / "gae.pt"
            values = torch.tensor([[.1], [.2]])
            flags = torch.zeros(2, 1, dtype=torch.bool)
            batch = {"rewards": torch.ones(2, 1), "values": values,
                     "bootstrap_values": values + .1, "terminated": flags, "truncated": flags}
            targets = compute_gae(*(batch[k] for k in (
                "rewards", "values", "bootstrap_values", "terminated", "truncated")))
            targets["metadata"] = {"source_rollout": str(rollout_path), "gamma": .99,
                                   "gae_lambda": .95, "advantage_normalization": False}
            torch.save({"rollout": batch}, rollout_path)
            torch.save(targets, targets_path)
            load_inputs(rollout_path, targets_path)
            targets["returns"][0] += .01
            torch.save(targets, targets_path)
            with self.assertRaises(AssertionError):
                load_inputs(rollout_path, targets_path)
            targets["metadata"]["source_rollout"] = str(Path(directory) / "other.pt")
            torch.save(targets, targets_path)
            with self.assertRaisesRegex(ValueError, "source_rollout"):
                load_inputs(rollout_path, targets_path)


if __name__ == "__main__":
    unittest.main()
