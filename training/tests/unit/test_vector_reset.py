"""Real mjlab delay/history buffers: partial resets must not advance other rows."""

import unittest
from types import SimpleNamespace

import torch
from mjlab.utils.buffers import CircularBuffer, DelayBuffer

from training.advantages import compute_gae
from training.models import GaussianActor
from training.normalization import ObservationNormalizer
from training.rollout import collect_rollout, verify_policy_rollout
from training.vector_reset import _isolate_actuator_writes, reset_done_envs


class HistoryEnv:
    def __init__(self, n=4):
        self.num_envs, self.device = n, "cpu"
        self.cfg = SimpleNamespace(auto_reset=False)
        self.action_manager = SimpleNamespace(total_action_dim=1)
        self.termination_manager = SimpleNamespace(active_terms=[])
        self.delay = DelayBuffer(min_lag=1, max_lag=1, batch_size=n, device="cpu")
        self.history = CircularBuffer(3, n, "cpu")
        self.command = SimpleNamespace(value=torch.arange(n).float(), metrics={"age": torch.zeros(n)})
        self.command_manager = SimpleNamespace(_terms={"command": self.command})
        self.observation_manager = SimpleNamespace(
            _group_obs_term_delay_buffer={"actor": {"sensor": self.delay}},
            _group_obs_term_history_buffer={"critic": {"sensor": self.history}},
            _group_obs_class_instances={}, _group_obs_class_term_cfgs={}, _obs_buffer=None)
        self.state = torch.arange(n).float().unsqueeze(1)
        self.t = 0
        self.reset_ids = []
        self.reward = torch.zeros(n)
        self.terminated = torch.zeros(n, dtype=torch.bool)
        self.truncated = torch.zeros(n, dtype=torch.bool)

    def observe(self):
        self.delay.append(self.state)
        delayed = self.delay.compute()
        self.history.append(delayed)
        self.obs_buf = {"actor": delayed.clone(), "critic": self.history.buffer.flatten(1)}
        self.observation_manager._obs_buffer = self.obs_buf
        return self.obs_buf

    def reset(self, *, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        if env_ids is not None:
            self.reset_ids.append(ids.tolist())
        self.state[ids] = -100 - ids.float().unsqueeze(1)
        self.delay.reset(ids)
        self.history.reset(ids)
        self.command.value[ids] = -10
        self.command.metrics["age"][ids] = 0
        self.command.metrics["age"].add_(1)  # Public reset recomputes ALL command metrics.
        self.reward.fill_(999)
        self.terminated.zero_()
        self.truncated.zero_()
        return self.observe(), {}

    def step(self, action):
        self.t += 1
        self.state.add_(10)
        self.reward.fill_(1)
        self.terminated.zero_()
        self.truncated.zero_()
        if self.t == 2:
            self.terminated[0] = True
        if self.t == 3:
            self.truncated[1] = True
        if self.t == 4:
            self.terminated[0] = self.truncated[0] = True
        if self.t == 5:
            self.truncated.fill_(True)  # Also exercise an all-row reset.
        return self.observe(), self.reward, self.terminated, self.truncated, {}


class VectorResetTests(unittest.TestCase):
    def test_extra_actuator_write_preserves_continuing_controls_and_motor_delay(self):
        n = 4
        delay = DelayBuffer(min_lag=1, max_lag=1, batch_size=n)
        delay.append(torch.arange(n).float().unsqueeze(1))
        delay.compute()
        actuator = SimpleNamespace(_delay_buffer=delay, _prev_motor_torque=torch.arange(n).float(),
                                   _bam_model=SimpleNamespace(actuator=SimpleNamespace(vin=torch.ones(n, 1))))
        sim = SimpleNamespace(data=SimpleNamespace(ctrl=torch.arange(n).float().unsqueeze(1)),
                              model=SimpleNamespace(dof_frictionloss=torch.ones(n, 1),
                                                    dof_damping=torch.ones(n, 1) * 2))

        class Scene:
            entities = {"robot": SimpleNamespace(actuators=[actuator])}

            def write_data_to_sim(self):
                delay.append(torch.full((n, 1), 100.))
                delay.compute()
                actuator._prev_motor_torque = torch.full((n,), 7.)
                actuator._bam_model.actuator.vin.add_(10)
                sim.data.ctrl.fill_(77)
                sim.model.dof_frictionloss.fill_(3)
                sim.model.dof_damping.fill_(4)

        env = SimpleNamespace(num_envs=n, sim=sim, scene=Scene())
        ids, keep = torch.tensor([1]), torch.tensor([True, False, True, True])
        before, pushes, clock = delay._buffer.buffer.clone(), delay._buffer._num_pushes.clone(), delay._step_count.clone()
        with _isolate_actuator_writes(env, ids, keep):
            delay.reset(ids)
            env.scene.write_data_to_sim()
            torch.testing.assert_close(sim.data.ctrl[keep, 0], torch.arange(n).float()[keep])
            torch.testing.assert_close(actuator._prev_motor_torque[keep], torch.arange(n).float()[keep])
            torch.testing.assert_close(actuator._bam_model.actuator.vin[keep], torch.ones(3, 1))
            torch.testing.assert_close(sim.model.dof_frictionloss[keep], torch.ones(3, 1))
            torch.testing.assert_close(sim.model.dof_damping[keep], torch.ones(3, 1) * 2)
            torch.testing.assert_close(delay._buffer.buffer[keep], before[keep])
            torch.testing.assert_close(delay._buffer._num_pushes[keep], pushes[keep])
            torch.testing.assert_close(delay._step_count[keep], clock[keep])
            self.assertEqual(sim.data.ctrl[1, 0], 77)
        self.assertNotIn("write_data_to_sim", vars(env.scene))
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with _isolate_actuator_writes(env, ids, keep):
                raise RuntimeError("probe")
        self.assertNotIn("write_data_to_sim", vars(env.scene))

    def test_partial_reset_preserves_storage_clock_commands_cache_and_next_history(self):
        env = HistoryEnv()
        env.reset()
        for _ in range(4):
            env.state.add_(10)
            obs = env.observe()
        before = {name: x.clone() for name, x in obs.items()}
        state = env.state.clone()
        delay = env.delay._buffer._buffer.clone()
        history = env.history.buffer.clone()
        clock = env.delay._step_count.clone()
        pointer = (env.delay._buffer._pointer, env.history._pointer)
        pushes = env.history._num_pushes.clone()
        metrics = env.command.metrics["age"].clone()
        result = reset_done_envs(env, torch.tensor([1, 3]), before)
        keep = torch.tensor([0, 2])
        self.assertEqual(pointer, (env.delay._buffer._pointer, env.history._pointer))
        torch.testing.assert_close(env.state[keep], state[keep], rtol=0, atol=0)
        torch.testing.assert_close(env.delay._buffer._buffer[:, keep], delay[:, keep], rtol=0, atol=0)
        torch.testing.assert_close(env.history.buffer[keep], history[keep], rtol=0, atol=0)
        torch.testing.assert_close(env.delay._step_count[keep], clock[keep], rtol=0, atol=0)
        torch.testing.assert_close(env.history._num_pushes[keep], pushes[keep], rtol=0, atol=0)
        torch.testing.assert_close(env.command.metrics["age"][keep], metrics[keep], rtol=0, atol=0)
        for name in before:
            torch.testing.assert_close(result[name][keep], before[name][keep], rtol=0, atol=0)
        self.assertIs(env.observation_manager._obs_buffer, result)
        torch.testing.assert_close(env.history.buffer[[1, 3]],
                                   env.state[[1, 3], None].expand(-1, 3, -1))
        # The NEXT real step must shift exactly once, retaining the same old frames.
        env.state.add_(10)
        following = env.observe()
        expected_history = torch.cat((history[keep, 1:], state[keep, None]), dim=1)
        torch.testing.assert_close(following["critic"][keep], expected_history.flatten(1))

    def test_vector_rollout_bootstrap_both_flags_gae_and_continuation(self):
        for n in (2, 4):
            with self.subTest(num_envs=n):
                env = HistoryEnv(n)
                actor = GaussianActor(obs_dim=1, action_dim=1, hidden_dims=(8,)).eval()
                critic = lambda x: x[:, -1]
                normalizer = ObservationNormalizer({"actor": 1, "critic": 3}).eval()
                kwargs = dict(actor=actor, critic=critic, normalizer=normalizer, bootstrap=True)
                batch = collect_rollout(env, 7, **kwargs)
                verify_policy_rollout(batch, actor, critic, normalizer)
                self.assertEqual(batch["rewards"].shape, (7, n))
                self.assertEqual(env.reset_ids, [[0], [1], [0], list(range(n))])
                self.assertEqual(batch["bootstrap_values"][1, 0], 0)
                self.assertEqual(batch["bootstrap_values"][3, 0], 0)  # Both flags: term wins.
                self.assertNotEqual(batch["bootstrap_values"][2, 1], 0)
                self.assertNotEqual(batch["transition_next_observations"]["critic"][2, 1, -1],
                                    batch["next_observations"]["critic"][2, 1, -1])
                targets = compute_gae(*(batch[k] for k in (
                    "rewards", "values", "bootstrap_values", "terminated", "truncated")))
                delta = batch["rewards"] + .99 * batch["bootstrap_values"] - batch["values"]
                done = batch["terminated"] | batch["truncated"]
                torch.testing.assert_close(targets["advantages"][done], delta[done])
                continuation = {k: x[-1] for k, x in batch["next_observations"].items()}
                following = collect_rollout(env, 2, initial_observations=continuation, **kwargs)
                for name, x in continuation.items():
                    torch.testing.assert_close(following["observations"][name][0], x, rtol=0, atol=0)
                self.assertEqual(env.t, 9)
                self.assertEqual(normalizer.stats["actor"].count, 0)

    def test_unsupported_managed_history_fails_explicitly(self):
        env = HistoryEnv()
        env.reset()
        env.observation_manager = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            reset_done_envs(env, torch.tensor([0]), env.obs_buf)


if __name__ == "__main__":
    unittest.main()
