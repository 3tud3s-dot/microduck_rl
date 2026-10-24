"""Audit the actual PPO minibatch inputs against independent [time, env] indexing."""

import copy
import math
import unittest
from unittest.mock import patch

import torch

from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer
from training.ppo import create_optimizer, optimizer_status, ppo_loss, update_ppo
from training.train import assert_same_state, cpu_snapshot, run_training
from training.tests.unit.test_vector_reset import HistoryEnv


def audited_update(actor, critic, normalizer, batch, targets, **kwargs):
    """Observe REAL network/loss calls; compare their complete sample tuples."""
    calls, inputs = [], {}
    original_evaluate = actor.evaluate_actions

    def actor_evaluate(obs, actions):
        inputs["actor"] = obs.detach().clone()
        inputs["actions"] = actions.detach().clone()
        return original_evaluate(obs, actions)

    def critic_input(module, args):
        inputs["critic"] = args[0].detach().clone()

    def loss(log_prob, old_log_prob, values, returns, advantages, entropy, **loss_kwargs):
        calls.append({**inputs, "old_log_prob": old_log_prob.detach().clone(),
                      "returns": returns.detach().clone(), "advantages": advantages.detach().clone()})
        return ppo_loss(log_prob, old_log_prob, values, returns, advantages, entropy, **loss_kwargs)

    hook = critic.register_forward_pre_hook(critic_input)
    try:
        with patch.object(actor, "evaluate_actions", side_effect=actor_evaluate), \
                patch("training.ppo.ppo_loss", side_effect=loss):
            report, state = update_ppo(actor, critic, normalizer, batch, targets, **kwargs)
    finally:
        hook.remove()
    t, n = batch["log_prob"].shape
    expected_advantages = targets["advantages"]
    expected_advantages = ((expected_advantages - expected_advantages.mean()) /
                           (expected_advantages.std(correction=0) + 1e-8))
    reference = {"actor": batch["normalized_observations"]["actor"],
                 "critic": batch["normalized_observations"]["critic"],
                 "actions": batch["actions"], "old_log_prob": batch["log_prob"],
                 "returns": targets["returns"], "advantages": expected_advantages}
    cursor = 0

    def check(indices):
        nonlocal cursor
        # Independent lookup into unflattened rollout: index k -> (k // N, k % N).
        times, envs = indices // n, indices % n
        for name, value in reference.items():
            torch.testing.assert_close(calls[cursor][name], value[times, envs],
                                       rtol=0, atol=0, msg=f"minibatch tuple mismatch: {name}")
        cursor += 1

    all_indices = torch.arange(t * n, device=batch["actions"].device)
    check(all_indices)
    for epoch in report["epoch_reports"]:
        order = torch.tensor(epoch["sample_order"], device=all_indices.device)
        torch.testing.assert_close(order.sort().values, all_indices, rtol=0, atol=0)
        start = 0
        for size in epoch["minibatch_sizes"]:
            check(order[start:start + size])
            start += size
        assert start == t * n
        check(all_indices)
    assert cursor == len(calls)
    report["sample_alignment_verified"] = True
    report["sample_axes"] = [t, n]
    return report, state


class VectorTrainTests(unittest.TestCase):
    def test_running_statistics_change_only_between_complete_ppo_iterations(self):
        from training.rollout import verify_policy_rollout

        torch.manual_seed(17)
        env = HistoryEnv(2)
        actor = GaussianActor(obs_dim=1, action_dim=1, hidden_dims=(8,))
        critic = Critic(obs_dim=3, hidden_dims=(8,))
        norm = ObservationNormalizer({"actor": 1, "critic": 3}).eval()
        optimizer = create_optimizer(actor, critic)
        saved = []

        def capture(record, batch, targets, sampling_policy):
            old_norm = ObservationNormalizer.from_checkpoint(sampling_policy["normalizer"])
            old_actor, old_critic = GaussianActor(**actor.config), Critic(**critic.config)
            old_actor.load_state_dict(sampling_policy["actor_state_dict"])
            old_critic.load_state_dict(sampling_policy["critic_state_dict"])
            verify_policy_rollout(batch, old_actor, old_critic, old_norm)
            self.assertFalse(norm.training)
            self.assertEqual(record["normalization"]["checkpoint_counts"]["actor"], record["iteration"] * 8)
            self.assertEqual(record["update"]["normalizer_counts"]["actor"], (record["iteration"] - 1) * 8)
            self.assertTrue(record["update"]["sample_alignment_verified"])
            saved.append(cpu_snapshot(batch))

        with patch("training.train.update_ppo", side_effect=audited_update):
            history = run_training(env, actor, critic, norm, optimizer,
                                   iterations=3, steps=4, epochs=2, minibatch_size=3,
                                   normalization_mode="running", on_iteration=capture)
        self.assertEqual(optimizer_status(optimizer)["step"], 18)
        for name, stats in norm.stats.items():
            reference = torch.cat([b["observations"][name] for b in saved]).double().reshape(-1, stats.dim)
            self.assertEqual(stats.count, 24)
            torch.testing.assert_close(stats.mean, reference.mean(0))
            torch.testing.assert_close(stats.var, reference.var(0, correction=0))
        for record in history:
            self.assertTrue(record["normalization"]["frozen_within_iteration"])
            self.assertGreaterEqual(record["normalization"]["boundary_policy_shift"]["gaussian_kl_mean"], 0)
            self.assertGreater(record["performance"]["samples_per_second_excluding_io"], 0)
            self.assertIsNone(record["performance"]["cuda_peak_allocated_bytes"])
        self.assertEqual(history[0]["episodes"]["completed_count"], 3)
        self.assertEqual(history[1]["episodes"]["completed_count"], 2)
        self.assertIsNone(history[2]["episodes"]["mean_return"])

    def test_three_vector_iterations_alignment_persistent_adam_and_replay(self):
        for n in (2, 4):
            with self.subTest(num_envs=n):
                torch.manual_seed(11)
                env = HistoryEnv(n)
                actor = GaussianActor(obs_dim=1, action_dim=1, hidden_dims=(8,))
                critic = Critic(obs_dim=3, hidden_dims=(8,))
                norm = ObservationNormalizer({"actor": 1, "critic": 3}).eval()
                optimizer = create_optimizer(actor, critic)
                frozen, saved = norm.checkpoint(), []
                generator = torch.Generator().manual_seed(81)

                def capture(record, batch, targets, sampling_policy):
                    saved.append(cpu_snapshot({"record": record, "batch": batch, "targets": targets,
                                               "sampling_policy": sampling_policy,
                                               "actor": actor.state_dict(), "critic": critic.state_dict(),
                                               "optimizer": optimizer.state_dict(),
                                               "shuffle": generator.get_state()}))

                with patch("training.train.update_ppo", side_effect=audited_update):
                    history = run_training(env, actor, critic, norm, optimizer, iterations=3,
                                           steps=7, epochs=2, minibatch_size=5,
                                           generator=generator, on_iteration=capture)
                updates = 2 * math.ceil(7 * n / 5)
                self.assertEqual(env.t, 21)
                self.assertEqual(env.reset_ids, [[0], [1], [0], list(range(n))])
                self.assertEqual(optimizer_status(optimizer)["step"], updates * 3)
                self.assertEqual([r["env_steps"] for r in history], [7*n, 14*n, 21*n])
                self.assertEqual([r["vector_steps"] for r in history], [7, 14, 21])
                for index, item in enumerate(saved):
                    report = item["record"]["update"]
                    self.assertTrue(report["sample_alignment_verified"])
                    self.assertEqual(report["sample_axes"], [7, n])
                    self.assertEqual(report["optimizer_before"]["step"], index * updates)
                    self.assertEqual(report["optimizer_after"]["step"], (index + 1) * updates)
                    self.assertEqual(report["epoch_reports"][0]["minibatch_sizes"][-1], (7*n) % 5)
                    if index:
                        assert_same_state(item["sampling_policy"]["actor_state_dict"], saved[index-1]["actor"])
                        assert_same_state(item["sampling_policy"]["critic_state_dict"], saved[index-1]["critic"])
                        for name in ("actor", "critic"):
                            torch.testing.assert_close(item["batch"]["observations"][name][0],
                                                       saved[index-1]["batch"]["next_observations"][name][-1],
                                                       rtol=0, atol=0)
                assert_same_state(norm.checkpoint(), frozen)
                # Exact NEXT update after restoring both moments, step and shuffle RNG.
                replay_actor = GaussianActor(**actor.config)
                replay_critic = Critic(**critic.config)
                replay_actor.load_state_dict(saved[1]["actor"])
                replay_critic.load_state_dict(saved[1]["critic"])
                replay_optimizer = create_optimizer(replay_actor, replay_critic)
                replay_optimizer.load_state_dict(copy.deepcopy(saved[1]["optimizer"]))
                audited_update(replay_actor, replay_critic, norm, saved[2]["batch"], saved[2]["targets"],
                               optimizer=replay_optimizer, epochs=2, minibatch_size=5,
                               generator=torch.Generator().set_state(saved[1]["shuffle"]))
                assert_same_state(replay_actor.state_dict(), saved[2]["actor"])
                assert_same_state(replay_critic.state_dict(), saved[2]["critic"])
                assert_same_state(replay_optimizer.state_dict(), saved[2]["optimizer"])

    def test_vector_full_batch_uses_time_times_env_and_auto_reset_is_rejected(self):
        env = HistoryEnv(4)
        actor = GaussianActor(obs_dim=1, action_dim=1, hidden_dims=(8,))
        critic = Critic(obs_dim=3, hidden_dims=(8,))
        norm = ObservationNormalizer({"actor": 1, "critic": 3}).eval()
        optimizer = create_optimizer(actor, critic)
        env.cfg.auto_reset = True
        with self.assertRaisesRegex(ValueError, "auto_reset=False"):
            run_training(env, actor, critic, norm, optimizer)
        self.assertEqual(env.t, 0)
        env.cfg.auto_reset = False
        report = run_training(env, actor, critic, norm, optimizer,
                              iterations=1, steps=3, epochs=2, minibatch_size=None)[0]
        self.assertEqual(report["samples_per_iteration"], 12)
        self.assertEqual(report["update"]["optimizer_steps"], 2)


if __name__ == "__main__":
    unittest.main()
