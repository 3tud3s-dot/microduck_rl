"""Regression checks for reused buffers and asynchronous auto-reset boundaries."""

import unittest
from types import SimpleNamespace

import torch

from training.models import Critic, GaussianActor
from training.normalization import ObservationNormalizer
from training.rollout import collect_rollout, verify_policy_rollout


class MutableEnv:
    """Two envs with deliberately reused tensors and recognizable reset values."""

    def __init__(self):
        self.cfg = SimpleNamespace(auto_reset=True)
        self.num_envs, self.device = 2, "cpu"
        self.action_manager = SimpleNamespace(total_action_dim=2)
        self.termination_manager = SimpleNamespace(active_terms=[])
        self.obs = {"actor": torch.zeros(2, 2), "critic": torch.zeros(2, 3)}
        self.reward = torch.zeros(2)
        self.terminated = torch.zeros(2, dtype=torch.bool)
        self.truncated = torch.zeros(2, dtype=torch.bool)
        self.step_count = 0

    def reset(self):
        for name, value in self.obs.items():
            value.copy_(torch.arange(value.numel()).reshape_as(value))
            if name == "critic":
                value.add_(100)
        self.step_count = 0
        return self.obs, {}

    def step(self, action):
        self.step_count += 1
        for value in self.obs.values():
            value.add_(10)
        # Env 0 terminates at step 2; env 1 is truncated at step 3.
        self.terminated.copy_(torch.tensor([self.step_count == 2, False]))
        self.truncated.copy_(torch.tensor([False, self.step_count == 3]))
        if self.step_count == 2:
            for value in self.obs.values():
                value[0].fill_(-200)
        if self.step_count == 3:
            for value in self.obs.values():
                value[1].fill_(-300)
        self.reward.copy_(torch.tensor([self.step_count, self.step_count + 0.5]))
        action.fill_(999)  # Stored action must be copied BEFORE env.step().
        return self.obs, self.reward, self.terminated, self.truncated, {}


class RolloutTests(unittest.TestCase):
    def test_continuation_does_not_reset_at_collection_horizons(self):
        class CountingEnv(MutableEnv):
            def reset(self):
                self.reset_count = getattr(self, "reset_count", 0) + 1
                return super().reset()

        env = CountingEnv()
        first = collect_rollout(env, 4)
        continuation = {name: x[-1] for name, x in first["next_observations"].items()}
        second = collect_rollout(env, 2, initial_observations=continuation)
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(env.step_count, 6)
        for name in continuation:
            torch.testing.assert_close(second["observations"][name][0], continuation[name])

    def test_transition_alignment_and_independent_snapshots(self):
        env = MutableEnv()
        action_fn = lambda obs: obs["actor"] * 0.1 + 1
        batch = collect_rollout(env, steps=4, action_fn=action_fn)
        actor = torch.tensor([[[0, 1], [2, 3]], [[10, 11], [12, 13]],
                              [[-200, -200], [22, 23]], [[-190, -190], [-300, -300.]]])
        next_actor = torch.cat((actor[1:], torch.tensor([[[-180., -180], [-290, -290]]])))
        critic = torch.tensor([[[100, 101, 102], [103, 104, 105]],
                               [[110, 111, 112], [113, 114, 115]],
                               [[-200, -200, -200], [123, 124, 125]],
                               [[-190, -190, -190], [-300., -300, -300]]])
        next_critic = torch.cat((critic[1:], torch.tensor([[[-180., -180, -180],
                                                          [-290, -290, -290]]])))
        # Poison live buffers after collection, including flags; history must survive.
        for value in (*env.obs.values(), env.reward):
            value.fill_(777)
        env.terminated.fill_(True)
        env.truncated.fill_(True)
        torch.testing.assert_close(batch["observations"]["actor"], actor)
        torch.testing.assert_close(batch["next_observations"]["actor"], next_actor)
        torch.testing.assert_close(batch["observations"]["critic"], critic)
        torch.testing.assert_close(batch["next_observations"]["critic"], next_critic)
        torch.testing.assert_close(batch["actions"], actor * 0.1 + 1)
        torch.testing.assert_close(batch["rewards"], torch.tensor([[1., 1.5], [2, 2.5],
                                                                 [3, 3.5], [4, 4.5]]))
        self.assertEqual(batch["terminated"].tolist(), [[False, False], [True, False],
                                                       [False, False], [False, False]])
        self.assertEqual(batch["truncated"].tolist(), [[False, False], [False, False],
                                                      [False, True], [False, False]])

    def test_both_flags_can_be_true(self):
        class BothFlagsEnv(MutableEnv):
            def step(self, action):
                result = super().step(action)
                self.terminated[0] = self.truncated[0] = True
                return result

        batch = collect_rollout(BothFlagsEnv(), steps=1)
        self.assertTrue(batch["terminated"][0, 0].item())
        self.assertTrue(batch["truncated"][0, 0].item())

    def test_zero_actions_and_invalid_collection(self):
        env = MutableEnv()
        batch = collect_rollout(env, steps=1)
        self.assertEqual(batch["actions"].shape, (1, 2, 2))
        self.assertEqual(torch.count_nonzero(batch["actions"]).item(), 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            collect_rollout(env, steps=0)
        env.cfg.auto_reset = False
        with self.assertRaisesRegex(ValueError, "auto_reset=True"):
            collect_rollout(env, steps=1)

    def test_policy_pairs_executed_actions_rewards_density_and_pre_step_values(self):
        class ActionRewardEnv(MutableEnv):
            def __init__(self):
                super().__init__()
                self.received = []

            def step(self, action):
                sent = action.detach().clone()
                self.received.append(sent)
                result = super().step(action)  # Reuses buffers and overwrites action.
                self.reward.copy_(sent.sum(-1) + self.step_count * 10)
                return result

        torch.manual_seed(11)
        actor = GaussianActor(obs_dim=2, action_dim=2, hidden_dims=(8,)).eval()
        critic = Critic(obs_dim=3, hidden_dims=(8,)).eval()
        norm = ObservationNormalizer({"actor": 2, "critic": 3})
        norm.update({"actor": torch.arange(8, dtype=torch.float32).reshape(4, 2),
                     "critic": torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100})
        norm.eval()
        modules = {"actor": actor, "critic": critic, "normalizer": norm}
        before = {name: {k: v.clone() for k, v in module.state_dict().items()}
                  for name, module in modules.items()}
        env = ActionRewardEnv()
        batch = collect_rollout(env, 4, actor=actor, critic=critic, normalizer=norm)
        torch.testing.assert_close(batch["actions"], torch.stack(env.received), rtol=0, atol=0)
        expected_reward = batch["actions"].sum(-1) + torch.arange(1, 5).unsqueeze(-1) * 10
        torch.testing.assert_close(batch["rewards"], expected_reward)
        verify_policy_rollout(batch, actor, critic, norm)
        self.assertEqual(batch["log_prob"].shape, (4, 2))
        self.assertEqual(batch["values"].shape, (4, 2))
        self.assertTrue(torch.count_nonzero(batch["actions"]).item() > 0)
        self.assertTrue(batch["terminated"][1, 0].item())
        self.assertTrue(batch["truncated"][2, 1].item())
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                self.assertTrue(torch.equal(value, before[name][key]))
        # The stored data is detached and remains usable for future gradient-based updates.
        for key in ("actions", "log_prob", "values"):
            self.assertFalse(batch[key].requires_grad)
            self.assertFalse(torch.is_inference(batch[key]))
        env.reward.fill_(999)
        env.terminated.fill_(True)
        for value in env.obs.values():
            value.fill_(999)
        torch.testing.assert_close(batch["rewards"], expected_reward)
        verify_policy_rollout(batch, actor, critic, norm)

    def test_policy_requires_complete_components_and_frozen_statistics(self):
        env = MutableEnv()
        actor = GaussianActor(obs_dim=2, action_dim=2, hidden_dims=(8,)).eval()
        critic = Critic(obs_dim=3, hidden_dims=(8,)).eval()
        norm = ObservationNormalizer({"actor": 2, "critic": 3})
        with self.assertRaisesRegex(ValueError, "together"):
            collect_rollout(env, 1, actor=actor)
        with self.assertRaisesRegex(ValueError, "frozen"):
            collect_rollout(env, 1, actor=actor, critic=critic, normalizer=norm)
        norm.eval()
        norm.stats["actor"].train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            collect_rollout(env, 1, actor=actor, critic=critic, normalizer=norm)
        norm.eval()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            collect_rollout(env, 1, action_fn=lambda obs: obs["actor"],
                            actor=actor, critic=critic, normalizer=norm)

    def test_invalid_policy_output_is_rejected_before_env_step(self):
        class BadActor:
            def sample(self, obs):
                return torch.zeros(2, 2), torch.zeros(2, 1)

        env = MutableEnv()
        norm = ObservationNormalizer({"actor": 2, "critic": 3}).eval()
        with self.assertRaisesRegex(ValueError, "log_prob"):
            collect_rollout(env, 1, actor=BadActor(), critic=lambda obs: obs[:, 0],
                            normalizer=norm)
        self.assertEqual(env.step_count, 0)

    def test_manual_reset_captures_true_next_state_before_shared_buffers_are_reset(self):
        class ManualEnv(MutableEnv):
            def __init__(self):
                super().__init__()
                self.cfg.auto_reset = False
                self.reset_ids = []

            def reset(self, *, env_ids=None):
                if env_ids is None:
                    return super().reset()
                self.reset_ids.append(env_ids.tolist())
                for index in env_ids.tolist():
                    for x in self.obs.values():
                        x[index].fill_(-200 if index == 0 else -300)
                self.reward.fill_(9999)
                self.terminated.zero_()
                self.truncated.zero_()
                return self.obs, {}

            def step(self, action):
                self.step_count += 1
                for x in self.obs.values():
                    x.add_(10)
                self.terminated.copy_(torch.tensor([self.step_count == 2, False]))
                self.truncated.copy_(torch.tensor([False, self.step_count == 3]))
                self.reward.fill_(self.step_count)
                action.fill_(999)
                return self.obs, self.reward, self.terminated, self.truncated, {}

        env = ManualEnv()
        norm = ObservationNormalizer({"actor": 2, "critic": 3}).eval()
        actor = GaussianActor(obs_dim=2, action_dim=2, hidden_dims=(8,)).eval()
        critic = lambda x: x[:, 0]
        batch = collect_rollout(env, 4, actor=actor, critic=critic, normalizer=norm,
                                bootstrap=True)
        verify_policy_rollout(batch, actor, critic, norm)
        self.assertEqual(env.reset_ids, [[0], [1]])
        self.assertTrue(batch["terminated"][1, 0].item())
        self.assertTrue(batch["truncated"][2, 1].item())
        self.assertEqual(batch["bootstrap_values"][1, 0].item(), 0)
        self.assertAlmostEqual(batch["bootstrap_values"][2, 1].item(), 133 / 1.01, places=4)
        self.assertEqual(batch["transition_next_observations"]["critic"][2, 1, 0].item(), 133)
        self.assertEqual(batch["next_observations"]["critic"][2, 1, 0].item(), -300)
        torch.testing.assert_close(batch["rewards"], torch.arange(1., 5).unsqueeze(1).expand(4, 2))
        for name in ("actor", "critic"):
            torch.testing.assert_close(batch["observations"][name][1:],
                                       batch["next_observations"][name][:-1])
        # Horizon values are saved even without a done, and are not inferred from
        # the last in-rollout value (which belongs to the PRE-action state).
        self.assertFalse((batch["terminated"][-1] | batch["truncated"][-1]).any())
        torch.testing.assert_close(batch["bootstrap_values"][-1],
                                   torch.tensor([-180., -290.]) / 1.01)
        self.assertFalse(batch["bootstrap_values"].requires_grad)
        self.assertEqual(norm.stats["critic"].count.item(), 0)

    def test_bootstrap_refuses_missing_terminal_observations(self):
        env = MutableEnv()
        norm = ObservationNormalizer({"actor": 2, "critic": 3}).eval()
        actor = GaussianActor(obs_dim=2, action_dim=2, hidden_dims=(8,)).eval()
        kwargs = dict(actor=actor, critic=lambda x: x[:, 0], normalizer=norm, bootstrap=True)
        with self.assertRaisesRegex(ValueError, "auto_reset=False"):
            collect_rollout(env, 1, **kwargs)
        env.cfg.auto_reset = False
        with self.assertRaisesRegex(ValueError, "requires a policy"):
            collect_rollout(env, 1, bootstrap=True)


if __name__ == "__main__":
    unittest.main()
