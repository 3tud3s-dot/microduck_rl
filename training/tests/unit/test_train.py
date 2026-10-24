"""Continuous collection, fresh policies and persistent Adam across a short loop."""

import copy
import unittest
from types import SimpleNamespace

import torch

from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer
from training.ppo import create_optimizer, optimizer_status, update_ppo
from training.rollout import verify_policy_rollout
from training.train import assert_same_state, cpu_snapshot, run_training


class LoopEnv:
    def __init__(self):
        self.cfg = SimpleNamespace(auto_reset=False)
        self.num_envs, self.device = 1, "cpu"
        self.action_manager = SimpleNamespace(total_action_dim=2)
        self.termination_manager = SimpleNamespace(active_terms=[])
        self.obs = {"actor": torch.zeros(1, 3), "critic": torch.zeros(1, 5)}
        self.reward = torch.zeros(1)
        self.terminated = torch.zeros(1, dtype=torch.bool)
        self.truncated = torch.zeros_like(self.terminated)
        self.total_steps = self.initial_resets = self.done_resets = 0
        self.pending = False

    def reset(self, *, env_ids=None):
        if env_ids is None:
            self.initial_resets += 1
        else:
            self.done_resets += 1
        for x in self.obs.values():
            x.copy_(torch.arange(x.numel()).reshape_as(x) * .1)
        self.pending = False
        # These shared buffers are deliberately destroyed by reset().
        self.terminated.zero_()
        self.truncated.zero_()
        self.reward.fill_(9999)
        return self.obs, {}

    def step(self, action):
        if self.pending:
            raise RuntimeError("done env must be reset before stepping")
        self.total_steps += 1
        for x in self.obs.values():
            x.add_(.03 + action.mean().item() * .01)
        self.reward.copy_(1 + action.sum(-1) * .1)
        self.terminated.fill_(self.total_steps % 5 == 0)
        self.truncated.fill_(self.total_steps % 7 == 0)
        self.pending = (self.terminated | self.truncated).item()
        return self.obs, self.reward, self.terminated, self.truncated, {}


class TrainTests(unittest.TestCase):
    def make_components(self):
        torch.manual_seed(41)
        actor = GaussianActor(obs_dim=3, action_dim=2, hidden_dims=(8,))
        critic = Critic(obs_dim=5, hidden_dims=(8,))
        norm = ObservationNormalizer({"actor": 3, "critic": 5})
        norm.update({"actor": torch.randn(8, 3), "critic": torch.randn(8, 5)})
        return actor, critic, norm.eval(), create_optimizer(actor, critic)

    def test_three_iterations_keep_env_optimizer_and_normalizer_state(self):
        actor, critic, norm, optimizer = self.make_components()
        frozen = norm.checkpoint()
        env, saved = LoopEnv(), []
        generator = torch.Generator().manual_seed(700)

        def capture(record, batch, targets, sampling_policy):
            saved.append(cpu_snapshot({"record": record, "batch": batch, "targets": targets,
                                       "sampling_policy": sampling_policy,
                                       "actor": actor.state_dict(), "critic": critic.state_dict(),
                                       "optimizer": optimizer.state_dict(),
                                       "shuffle_state": generator.get_state()}))

        history = run_training(env, actor, critic, norm, optimizer, iterations=3, steps=4,
                               epochs=2, minibatch_size=3, generator=generator, on_iteration=capture)
        self.assertEqual(env.total_steps, 12)
        self.assertEqual(env.initial_resets, 1)
        self.assertEqual(env.done_resets, 3)
        self.assertEqual(optimizer_status(optimizer)["step"], 12)
        self.assertEqual([x["update"]["optimizer_after"]["step"] for x in history], [4, 8, 12])
        for item in history:
            self.assertEqual(item["update"]["optimizer_steps"], 4)
            self.assertTrue(item["update"]["fixed_targets_verified"])
            self.assertEqual([epoch["minibatch_sizes"] for epoch in item["update"]["epoch_reports"]],
                             [[3, 1], [3, 1]])
        self.assertEqual(sum(x["terminated"] for x in history), 2)
        self.assertEqual(sum(x["truncated"] for x in history), 1)
        assert_same_state(norm.checkpoint(), frozen)
        for index, item in enumerate(saved):
            old_actor = GaussianActor(**item["sampling_policy"]["actor_config"])
            old_critic = Critic(**item["sampling_policy"]["critic_config"])
            old_actor.load_state_dict(item["sampling_policy"]["actor_state_dict"])
            old_critic.load_state_dict(item["sampling_policy"]["critic_state_dict"])
            verify_policy_rollout(item["batch"], old_actor, old_critic, norm)
            if index:
                assert_same_state(item["sampling_policy"]["actor_state_dict"], saved[index - 1]["actor"])
                assert_same_state(item["sampling_policy"]["critic_state_dict"], saved[index - 1]["critic"])
                for group in ("actor", "critic"):
                    torch.testing.assert_close(item["batch"]["observations"][group][0],
                                               saved[index - 1]["batch"]["next_observations"][group][-1])

        # Restoring weights + Adam state must give the same NEXT parameter update.
        restored_actor, restored_critic, restored_norm, restored_optimizer = self.make_components()
        restored_actor.load_state_dict(saved[1]["actor"])
        restored_critic.load_state_dict(saved[1]["critic"])
        restored_norm.load_state_dict(frozen["state_dict"])
        restored_optimizer.load_state_dict(copy.deepcopy(saved[1]["optimizer"]))
        restored_generator = torch.Generator().set_state(saved[1]["shuffle_state"])
        update_ppo(restored_actor, restored_critic, restored_norm, saved[2]["batch"],
                   saved[2]["targets"], optimizer=restored_optimizer, epochs=2,
                   minibatch_size=3, generator=restored_generator)
        assert_same_state(restored_actor.state_dict(), saved[2]["actor"])
        assert_same_state(restored_critic.state_dict(), saved[2]["critic"])
        assert_same_state(restored_optimizer.state_dict(), saved[2]["optimizer"])

    def test_invalid_loop_settings_do_not_step_or_reset(self):
        actor, critic, norm, optimizer = self.make_components()
        env = LoopEnv()
        for kwargs in ({"iterations": 0}, {"steps": 0}, {"gamma": -1},
                       {"epochs": 0}, {"minibatch_size": 0}):
            with self.assertRaises(ValueError):
                run_training(env, actor, critic, norm, optimizer, **kwargs)
        norm.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            run_training(env, actor, critic, norm, optimizer)
        self.assertEqual(env.total_steps, 0)
        self.assertEqual(env.initial_resets, 0)


if __name__ == "__main__":
    unittest.main()
