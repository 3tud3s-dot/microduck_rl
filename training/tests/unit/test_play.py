"""Native-play adapters without opening a window during CPU unit tests."""

from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from training.play import configure_native_gpu, MeanPolicy, PlaybackEnv, _make_native_viewer, run_native_playback
from training.tests.unit import test_evaluate as fixtures
from training.train import assert_same_state


class PlayEnv(fixtures.EpisodeEnv):
    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None):
        return super().reset(seed=42 if seed is None else seed)


class FakeViewer:
    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self.closed = self.paused = False
        self.status = SimpleNamespace(last_error=None)

    def pause(self):
        self.paused = True

    def run(self, num_steps=None):
        for _ in range(num_steps or 6):
            action = self.policy(self.env.get_observations())
            self.env.step(action)

    def get_status(self):
        return self.status

    def close(self):
        self.closed = True


class PlayTests(unittest.TestCase):
    def make_policy(self):
        return fixtures.EvaluateTests().make_policy()

    def test_process_gpu_settings_match_the_preserved_launcher(self):
        from training.paths import TRAINING_ROOT
        launcher = TRAINING_ROOT.parent / "scripts/play_native_wsl.sh"
        expected = dict(line.removeprefix("export ").split("=", 1)
                        for line in launcher.read_text().splitlines() if line.startswith("export "))
        environ = {"OTHER": "keep", "MUJOCO_GL": "egl"}
        settings = configure_native_gpu(environ)
        self.assertEqual(settings, expected)
        self.assertEqual(environ, {"OTHER": "keep", **expected})

    def test_mean_policy_uses_frozen_inputs_without_sampling_or_clipping(self):
        actor, _, norm = self.make_policy()
        actor.sample = Mock(side_effect=AssertionError("no sampling"))
        env = PlayEnv()
        observations, _ = env.reset()
        policy = MeanPolicy(actor, norm, action_dim=2, device="cpu")
        frozen = norm.checkpoint()
        action = policy(observations)
        torch.testing.assert_close(action, torch.tensor([[2., -3.]]))
        self.assertFalse(action.requires_grad)
        self.assertEqual(policy.calls, 1)
        actor.sample.assert_not_called()
        assert_same_state(norm.checkpoint(), frozen)

    def test_adapter_caches_observations_and_preserves_done_flags_before_reset(self):
        _, _, norm = self.make_policy()
        for flags in ((True, False), (False, True), (True, True)):
            with self.subTest(flags=flags):
                env, episodes = PlayEnv(flags=flags), []
                adapter = PlaybackEnv(env, norm, episodes.append)
                with self.assertRaisesRegex(RuntimeError, "reset playback"):
                    adapter.get_observations()
                adapter.reset(seed=42)
                self.assertIs(adapter.unwrapped, env)
                for _ in range(8):
                    self.assertIs(adapter.get_observations(), adapter._observations)
                self.assertEqual(env.reset_count, 1)
                for _ in range(3):
                    result = adapter.step(torch.tensor([[2., -3.]]))
                self.assertEqual(adapter.completed_episodes, 1)
                self.assertEqual(adapter.manual_resets, 0)
                self.assertEqual(env.reset_count, 2)
                self.assertEqual(adapter.episode_steps, 0)
                self.assertEqual((result[2].item(), result[3].item()), flags)
                self.assertFalse((env.terminated | env.truncated).any())
                env.reward.fill_(999)
                self.assertAlmostEqual(result[1].item(), .29, places=6)
                self.assertAlmostEqual(episodes[0]["return"], .57, places=6)

    def test_partial_keyboard_reset_does_not_invent_a_completed_episode(self):
        _, _, norm = self.make_policy()
        env = PlayEnv(done_after=10)
        adapter = PlaybackEnv(env, norm)
        adapter.reset(seed=42)
        adapter.step(torch.tensor([[2., -3.]]))
        adapter.reset()
        self.assertEqual(adapter.total_steps, 1)
        self.assertEqual(adapter.manual_resets, 1)
        self.assertEqual(adapter.completed_episodes, 0)
        self.assertEqual(adapter.episode_steps, 0)

    def test_viewer_loop_preserves_models_stats_and_closes_window(self):
        actor, critic, norm = self.make_policy()
        frozen = norm.checkpoint()
        env, viewers, episodes = PlayEnv(), [], []

        def factory(adapted, policy):
            viewer = FakeViewer(adapted, policy)
            viewers.append(viewer)
            return viewer

        report = run_native_playback(env, actor, critic, norm, steps=6,
                                      viewer_factory=factory, on_episode=episodes.append)
        self.assertEqual(report["steps"], 6)
        self.assertEqual(report["mean_policy_calls"], 6)
        self.assertEqual(report["completed_episodes"], 2)
        self.assertEqual(report["manual_resets"], 0)
        self.assertTrue(viewers[0].closed)
        self.assertEqual(len(episodes), 2)
        assert_same_state(norm.checkpoint(), frozen)

    def test_viewer_error_is_propagated_and_closed_instead_of_reported_as_success(self):
        actor, critic, norm = self.make_policy()
        viewers = []

        class ErrorViewer(FakeViewer):
            def run(self, num_steps=None):
                self.status.last_error = "policy failed"

        def factory(env, policy):
            viewer = ErrorViewer(env, policy)
            viewers.append(viewer)
            return viewer

        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            run_native_playback(PlayEnv(), actor, critic, norm, viewer_factory=factory)
        self.assertTrue(viewers[0].closed)

    def test_native_close_waits_for_render_thread_and_is_idempotent(self):
        class NativeBackend:
            def __init__(self, *args, **kwargs):
                self.viewer = None
                self.request_close, self.finished = threading.Event(), threading.Event()
                self.close_calls = 0

            def setup(self):
                def render_loop():
                    self.request_close.wait(timeout=2)
                    time.sleep(.02)  # Model the async GLFW/window cleanup.
                    self.finished.set()

                self.thread = threading.Thread(target=render_loop, name="test (_launch_internal)", daemon=True)
                self.thread.start()
                self.viewer = object()

            def close(self):
                self.close_calls += 1
                self.viewer = None
                self.request_close.set()

        module = SimpleNamespace(NativeMujocoViewer=NativeBackend, VerbosityLevel=SimpleNamespace(INFO=1))
        with patch.dict(sys.modules, {"mjlab": SimpleNamespace(), "mjlab.viewer": module}):
            viewer = _make_native_viewer(None, None)
            viewer.setup()
            viewer.close()
            self.assertTrue(viewer.finished.is_set())
            self.assertFalse(viewer.thread.is_alive())
            viewer.close()
            self.assertEqual(viewer.close_calls, 1)

    def test_invalid_limits_and_unfrozen_modes_do_not_launch_viewer(self):
        actor, critic, norm = self.make_policy()
        env, factory = PlayEnv(), Mock(side_effect=AssertionError("must not launch"))
        for kwargs in ({"steps": 0}, {"seed": -1}):
            with self.assertRaises(ValueError):
                run_native_playback(env, actor, critic, norm, viewer_factory=factory, **kwargs)
        norm.stats["critic"].train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            run_native_playback(env, actor, critic, norm, viewer_factory=factory)
        self.assertEqual(env.reset_count, 0)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
