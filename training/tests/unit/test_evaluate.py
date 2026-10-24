"""Mean-action evaluation, honest cutoffs, immutable stats and paired fresh envs."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from training.evaluate import compare_checkpoints, evaluate_episode
from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer
from training.train import assert_same_state, cpu_snapshot


class EpisodeEnv:
    def __init__(self, *, done_after=3, flags=(True, False), offset=0):
        self.cfg = SimpleNamespace(auto_reset=False)
        self.num_envs, self.device, self.step_dt = 1, "cpu", .02
        self.action_manager = SimpleNamespace(total_action_dim=2)
        self.termination_manager = SimpleNamespace(active_terms=["fell_over", "time_out"],
                                                   get_term=self.get_term)
        self.obs = {"actor": torch.zeros(1, 3), "critic": torch.zeros(1, 5)}
        self.reward = torch.zeros(1)
        self.terminated = torch.zeros(1, dtype=torch.bool)
        self.truncated = torch.zeros_like(self.terminated)
        self.done_after, self.flags, self.offset = done_after, flags, offset
        self.reset_count = self.steps = 0
        self.closed = False

    def get_term(self, name):
        return self.terminated if name == "fell_over" else self.truncated

    def reset(self, *, seed):
        self.seed = seed
        self.reset_count += 1
        self.steps = 0
        self.terminated.zero_()
        self.truncated.zero_()
        for x in self.obs.values():
            x.copy_(torch.arange(x.numel()).reshape_as(x) * .1 + seed * .001 + self.offset)
        return self.obs, {}

    def step(self, action):
        if (self.terminated | self.truncated).any():
            raise RuntimeError("evaluation continued past done")
        self.steps += 1
        sent = action.detach().clone()
        for x in self.obs.values():
            x.add_(.1 + sent.mean().item() * .01)
        self.reward.copy_(self.steps * .1 + sent.sum(-1) * .01)
        self.terminated.fill_(self.steps == self.done_after and self.flags[0])
        self.truncated.fill_(self.steps == self.done_after and self.flags[1])
        action.fill_(999)  # The evaluator must snapshot the executed mean beforehand.
        return self.obs, self.reward, self.terminated, self.truncated, {}

    def close(self):
        self.closed = True


class EvaluateTests(unittest.TestCase):
    def make_policy(self):
        torch.manual_seed(81)
        actor = GaussianActor(obs_dim=3, action_dim=2, hidden_dims=(8,)).eval()
        critic = Critic(obs_dim=5, hidden_dims=(8,)).eval()
        norm = ObservationNormalizer({"actor": 3, "critic": 5})
        norm.update({"actor": torch.randn(4, 3), "critic": torch.randn(4, 5)})
        with torch.no_grad():
            actor.mean_net[-1].weight.zero_()
            actor.mean_net[-1].bias.copy_(torch.tensor([2., -3.]))
        return actor, critic, norm.eval()

    def test_means_are_executed_without_sampling_or_clipping_and_snapshots_survive(self):
        actor, critic, norm = self.make_policy()
        before = cpu_snapshot({"actor": actor.state_dict(), "critic": critic.state_dict(),
                               "normalizer": norm.checkpoint()})
        actor.sample = Mock(side_effect=AssertionError("sampling is forbidden"))
        env = EpisodeEnv()
        result, trace = evaluate_episode(env, actor, critic, norm, seed=42, max_steps=8)
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(result["steps"], 3)
        self.assertAlmostEqual(result["episode_return"], .57, places=6)
        self.assertAlmostEqual(result["duration_s"], .06)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertTrue(result["completed"])
        self.assertFalse(result["evaluation_cutoff"])
        self.assertEqual(result["active_end_terms"], ["fell_over"])
        actor.sample.assert_not_called()
        torch.testing.assert_close(trace["actions"], torch.tensor([[[2., -3.]]] * 3))
        self.assertFalse(trace["actions"].requires_grad)
        env.reward.fill_(999)
        env.terminated.zero_()
        env.obs["actor"].fill_(999)
        torch.testing.assert_close(trace["rewards"], torch.tensor([[.09], [.19], [.29]]))
        self.assertTrue(trace["terminated"][-1, 0])
        assert_same_state(actor.state_dict(), before["actor"])
        assert_same_state(critic.state_dict(), before["critic"])
        assert_same_state(norm.checkpoint(), before["normalizer"])

    def test_environment_truncation_both_flags_and_evaluation_cutoff_are_distinct(self):
        for done_after, flags, limit, expected_steps, cutoff in (
                (3, (False, True), 8, 3, False), (3, (True, True), 8, 3, False),
                (10, (True, False), 2, 2, True), (3, (True, False), 3, 3, False)):
            with self.subTest(flags=flags, limit=limit):
                actor, critic, norm = self.make_policy()
                result, _ = evaluate_episode(EpisodeEnv(done_after=done_after, flags=flags),
                                              actor, critic, norm, seed=42, max_steps=limit)
                self.assertEqual(result["steps"], expected_steps)
                self.assertEqual(result["evaluation_cutoff"], cutoff)
                self.assertEqual(result["completed"], not cutoff)
                self.assertEqual((result["terminated"], result["truncated"]),
                                 (False, False) if cutoff else flags)

    def write_checkpoints(self, directory):
        actor, critic, norm = self.make_policy()
        policy = {"actor_config": actor.config, "critic_config": critic.config,
                  "actor_state_dict": cpu_snapshot(actor.state_dict()),
                  "critic_state_dict": cpu_snapshot(critic.state_dict()), "normalizer": norm.checkpoint()}
        initial, trained = Path(directory) / "initial.pt", Path(directory) / "trained.pt"
        torch.save(policy, initial)
        policy["actor_state_dict"]["mean_net.2.bias"] += .1
        torch.save(policy, trained)
        return initial, trained

    def test_checkpoint_pairs_use_fresh_same_seed_envs_and_close_them(self):
        with tempfile.TemporaryDirectory() as directory:
            initial, trained = self.write_checkpoints(directory)
            environments = []

            def make_env(seed):
                env = EpisodeEnv()
                environments.append(env)
                return env

            report, traces = compare_checkpoints(initial, trained, make_env, seeds=(42, 43))
            self.assertEqual(len(environments), 4)
            self.assertTrue(all(env.closed and env.reset_count == 1 for env in environments))
            self.assertEqual([env.seed for env in environments], [42, 42, 43, 43])
            for pair in report["paired_comparisons"]:
                self.assertTrue(pair["initial_conditions_matched"])
                self.assertTrue(pair["both_completed"])
                self.assertAlmostEqual(pair["return_delta_trained_minus_initial"], .006, places=6)
            for index in range(2):
                assert_same_state(traces["initial"][index]["initial_conditions"],
                                  traces["trained"][index]["initial_conditions"])
            self.assertEqual(report["aggregates"]["trained"]["completed_episodes"], 2)

    def test_initial_condition_mismatch_fails_before_first_action_and_closes_env(self):
        with tempfile.TemporaryDirectory() as directory:
            initial, trained = self.write_checkpoints(directory)
            environments = []

            def make_env(seed):
                env = EpisodeEnv(offset=len(environments))
                environments.append(env)
                return env

            with self.assertRaisesRegex(RuntimeError, "changed unexpectedly"):
                compare_checkpoints(initial, trained, make_env, seeds=(42,))
            self.assertTrue(all(env.closed for env in environments))
            self.assertEqual(environments[1].steps, 0)

    def test_invalid_modes_and_limits_are_rejected_before_reset(self):
        actor, critic, norm = self.make_policy()
        env = EpisodeEnv()
        for kwargs in ({"max_steps": 0}, {"seed": -1}):
            with self.assertRaises(ValueError):
                evaluate_episode(env, actor, critic, norm, **({"seed": 42} | kwargs))
        norm.stats["actor"].train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            evaluate_episode(env, actor, critic, norm, seed=42)
        norm.eval()
        actor.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            evaluate_episode(env, actor, critic, norm, seed=42)
        self.assertEqual(env.reset_count, 0)

    def test_nan_state_aborts_instead_of_counting_a_normal_episode(self):
        actor, critic, norm = self.make_policy()
        env = EpisodeEnv()
        env.termination_manager = SimpleNamespace(active_terms=["nan_state"],
                                                  get_term=lambda _: torch.ones(1, dtype=torch.bool))
        with self.assertRaisesRegex(RuntimeError, "nan_state"):
            evaluate_episode(env, actor, critic, norm, seed=42)


if __name__ == "__main__":
    unittest.main()
