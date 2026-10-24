"""Check batch axes, Gaussian action pairing, gradients and frozen normalization."""

import io
import math
import unittest

import torch

from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_forward_preserves_time_and_single_env_axes(self):
        actor, critic = GaussianActor(), Critic()
        self.assertEqual(actor(torch.randn(3, 1, 61)).shape, (3, 1, 14))
        self.assertEqual(critic(torch.randn(3, 1, 76)).shape, (3, 1))
        self.assertEqual(critic(torch.randn(1, 76)).shape, (1,))
        self.assertEqual(critic(torch.randn(76)).shape, ())
        self.assertEqual(actor(torch.randn(1, 61, dtype=torch.float64)).dtype, torch.float32)
        with self.assertRaises(ValueError):
            actor(torch.randn(1, 76))
        with self.assertRaises(ValueError):
            critic(torch.randn(1, 61))

    def test_joint_density_uses_exact_actions_and_last_axis(self):
        actor = GaussianActor(obs_dim=3, action_dim=2, hidden_dims=(8,))
        with torch.no_grad():
            actor.log_std.copy_(torch.tensor([-0.3, -0.9]))
        obs = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 10
        actions = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 7 - 0.5
        mean = actor(obs)
        std = actor.log_std.exp().expand_as(mean)
        expected = (-0.5 * ((actions - mean) / std).square() - std.log()
                    - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob, entropy = actor.evaluate_actions(obs, actions)
        self.assertEqual(log_prob.shape, (2, 3))
        torch.testing.assert_close(log_prob, expected)
        torch.testing.assert_close(entropy, (std.log() + 0.5 * math.log(2 * math.pi * math.e)).sum(-1))
        sampled, sampled_log_prob = actor.sample(obs)
        recomputed, _ = actor.evaluate_actions(obs, sampled)
        torch.testing.assert_close(sampled_log_prob, recomputed, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "broadcasting"):
            actor.evaluate_actions(obs, actions[0, 0])

    def test_policy_and_value_gradients_without_parameter_updates(self):
        actor = GaussianActor(obs_dim=3, action_dim=2, hidden_dims=(8,))
        critic = Critic(obs_dim=5, hidden_dims=(8,))
        obs = torch.randn(4, 3)
        fixed_actions = actor(obs).detach() + torch.tensor([0.4, -0.7])
        log_prob, _ = actor.evaluate_actions(obs, fixed_actions)
        (-log_prob.mean()).backward()
        for parameter in actor.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertGreater(actor.log_std.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in actor.mean_net.parameters()))
        critic(torch.randn(4, 5)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            for p in critic.parameters()))
        self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in critic.parameters()))

    def test_std_limits_and_unfiltered_sampling_in_eval(self):
        actor = GaussianActor(obs_dim=3, action_dim=2, hidden_dims=(8,)).eval()
        obs = torch.zeros(2, 3)
        with torch.no_grad():
            actor.log_std.copy_(torch.tensor([-100., 100.]))
        std = actor.distribution(obs).stddev
        torch.testing.assert_close(std[0], torch.tensor([math.exp(-5), math.exp(2)]))
        with torch.no_grad():
            actor.log_std.fill_(math.log(0.05))
            actor.mean_net[-1].weight.zero_()
            actor.mean_net[-1].bias.fill_(3)
        first, _ = actor.sample(obs)
        second, _ = actor.sample(obs)
        self.assertTrue(first.gt(1).all().item())  # No tanh or [-1, 1] clipping.
        self.assertFalse(torch.equal(first, second))  # eval does not disable sampling.
        for invalid_std in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GaussianActor(init_std=invalid_std)

    def test_normalizer_integration_and_checkpoint_round_trip(self):
        raw = {"actor": torch.randn(4, 1, 3), "critic": torch.randn(4, 1, 5) * 100}
        norm = ObservationNormalizer({"actor": 3, "critic": 5})
        norm.update(raw)
        norm.eval()
        frozen = {name: value.clone() for name, value in norm.state_dict().items()}
        obs = norm(raw)
        actor = GaussianActor(obs_dim=3, action_dim=2, hidden_dims=(8,)).eval()
        critic = Critic(obs_dim=5, hidden_dims=(8,)).eval()
        with torch.no_grad():
            actions, log_prob = actor.sample(obs["actor"])
            mean, values = actor(obs["actor"]), critic(obs["critic"])
        stream = io.BytesIO()
        torch.save({"normalizer": norm.checkpoint(), "actor_config": actor.config,
                    "critic_config": critic.config, "actor": actor.state_dict(),
                    "critic": critic.state_dict()}, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        restored_norm = ObservationNormalizer.from_checkpoint(loaded["normalizer"])
        restored_actor = GaussianActor(**loaded["actor_config"])
        restored_critic = Critic(**loaded["critic_config"])
        restored_actor.load_state_dict(loaded["actor"], strict=True)
        restored_critic.load_state_dict(loaded["critic"], strict=True)
        restored_obs = restored_norm(raw)
        torch.testing.assert_close(restored_actor(restored_obs["actor"]), mean, rtol=0, atol=0)
        torch.testing.assert_close(restored_critic(restored_obs["critic"]), values, rtol=0, atol=0)
        restored_log_prob, _ = restored_actor.evaluate_actions(restored_obs["actor"], actions)
        torch.testing.assert_close(restored_log_prob, log_prob, rtol=0, atol=0)
        for name, value in norm.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]))
        self.assertFalse(restored_norm.training)


if __name__ == "__main__":
    unittest.main()
