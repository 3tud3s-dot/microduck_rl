"""Episode boundaries across horizons and exact reward-unit accounting."""

from types import SimpleNamespace
import unittest

import torch

from training.metrics import EpisodeTracker, optimization_metrics


class TrainingMetricsTests(unittest.TestCase):
    def make_env(self, scale=True):
        self.flags = {"fell_over": torch.zeros(2, dtype=torch.bool),
                      "time_out": torch.zeros(2, dtype=torch.bool)}
        reward = SimpleNamespace(active_terms=["tracking", "penalty"],
                                 _step_reward=torch.tensor([[3., -1.], [6., -2.]]), _scale_by_dt=scale,
                                 get_term_cfg=lambda name: SimpleNamespace(weight=2 if name == "tracking" else -1))
        return SimpleNamespace(num_envs=2, device="cpu", step_dt=.02, reward_manager=reward,
                               termination_manager=SimpleNamespace(active_terms=list(self.flags),
                                                                   get_term=lambda name: self.flags[name]))

    def step(self, tracker, env, terminated=(False, False), truncated=(False, False)):
        self.flags["fell_over"].copy_(torch.tensor(terminated))
        self.flags["time_out"].copy_(torch.tensor(truncated))
        tracker.observe(env, {"rewards": env.reward_manager._step_reward.sum(-1) *
                             (env.step_dt if env.reward_manager._scale_by_dt else 1.),
                             "terminated": torch.tensor(terminated), "truncated": torch.tensor(truncated)})

    def test_incomplete_episodes_span_rounds_and_terms_sum_to_actual_return(self):
        env = self.make_env()
        tracker = EpisodeTracker(env)
        self.step(tracker, env)
        first = tracker.summary()
        self.assertIsNone(first["mean_return"])
        self.assertEqual(first["completed_count"], 0)
        self.assertEqual([x["steps"] for x in first["unfinished"]], [1, 1])
        tracker.begin_iteration()
        self.step(tracker, env, terminated=(True, False))
        self.step(tracker, env, truncated=(False, True))
        report = tracker.summary()
        self.assertEqual(report["completed_count"], 2)
        self.assertAlmostEqual(report["mean_return"], (.08 + .24) / 2)
        self.assertEqual(report["mean_steps"], 2.5)
        self.assertEqual(report["mean_duration_s"], .05)
        self.assertEqual(report["terminated_fraction"], .5)
        self.assertEqual(report["truncated_fraction"], .5)
        self.assertEqual(report["fell_over_fraction"], .5)
        for episode in report["records"]:
            self.assertAlmostEqual(sum(episode["reward_terms"].values()), episode["return"], places=7)
        self.assertAlmostEqual(report["rollout_reward_terms"]["tracking"]["mean_step_contribution"], .09)
        self.assertEqual(report["rollout_reward_terms"]["penalty"]["weight_min"], -1)
        self.assertEqual([x["steps"] for x in report["unfinished"]], [1, 0])

    def test_both_flags_count_once_and_unscaled_manager_has_no_extra_dt(self):
        env = self.make_env(scale=False)
        tracker = EpisodeTracker(env)
        self.step(tracker, env, terminated=(True, False), truncated=(True, False))
        report = tracker.summary()
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(report["both_flags_count"], 1)
        self.assertEqual(report["records"][0]["return"], 2)
        self.assertEqual(report["records"][0]["reward_terms"], {"tracking": 3, "penalty": -1})
        self.assertEqual(set(report["records"][0]["end_terms"]), {"fell_over", "time_out"})

    def test_bad_reward_units_are_rejected_and_zero_variance_ev_is_null(self):
        env = self.make_env()
        tracker = EpisodeTracker(env)
        with self.assertRaises(AssertionError):
            tracker.observe(env, {"rewards": torch.ones(2), "terminated": torch.zeros(2, dtype=torch.bool),
                                  "truncated": torch.zeros(2, dtype=torch.bool)})
        metrics = optimization_metrics(
            {"after": {"approx_kl": 0.}, "total_grad_norm_before": 1., "total_grad_norm_after": .5},
            {"values": torch.zeros(2, 2), "actions": torch.ones(2, 2, 1)},
            {"returns": torch.ones(2, 2), "advantages": torch.zeros(2, 2)},
            SimpleNamespace(param_groups=[{"lr": 1e-5}]))
        self.assertIsNone(metrics["pre_update_explained_variance"])
        self.assertEqual(metrics["learning_rates"], [1e-5])


if __name__ == "__main__":
    unittest.main()
