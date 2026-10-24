"""Native WSLg playback of our actor means with frozen checkpoint normalization."""

import argparse
import math
import os
from pathlib import Path
import threading

import torch

from training.evaluate import _check_observations
from training.rollout import _check_policy_result, load_policy
from training.train import assert_same_state, cpu_snapshot


def configure_native_gpu(environ=None):
    """Match scripts/play_native_wsl.sh without modifying or invoking its RSL runner.

    Call before importing mjlab/mujoco in a fresh CLI process. These settings affect
    only this process and its children; importing this module has no env side effects.
    """
    settings = {"GALLIUM_DRIVER": "d3d12", "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
                "MUJOCO_GL": "glfw"}
    (os.environ if environ is None else environ).update(settings)
    return settings


class MeanPolicy:
    def __init__(self, actor, normalizer, *, action_dim, device):
        if actor.training or normalizer.training or any(x.training for x in normalizer.stats.values()):
            raise ValueError("actor must be in eval mode and normalizer must be frozen")
        self.actor, self.normalizer = actor, normalizer
        self.action_dim, self.device = action_dim, device
        self.calls = 0

    @torch.no_grad()
    def __call__(self, observations):
        _check_observations(observations, self.normalizer, self.device)
        action = self.actor(self.normalizer(observations)["actor"])
        _check_policy_result("mean actions", action, (1, self.action_dim), self.device)
        self.calls += 1
        return action  # No sample(), clipping, EMA or second normalization.


class PlaybackEnv:
    """Adapt raw mjlab observations to the native viewer's cached-observation API.

    Keep auto_reset=False so end flags/rewards can be copied before a manual reset.
    The returned done transition's next observation is the NEW episode's input.
    Native rendering uses unwrapped to reach the original GPU simulation directly.
    """

    def __init__(self, env, normalizer, on_episode=None):
        if env.num_envs != 1 or env.cfg.auto_reset:
            raise ValueError("playback requires one env with auto_reset=False")
        self.env, self.normalizer, self.on_episode = env, normalizer, on_episode
        self._observations = None
        self.total_steps = self.completed_episodes = self.manual_resets = 0
        self.episode_steps, self.episode_return = 0, 0.0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        if self._observations is None:
            raise RuntimeError("reset playback before requesting observations")
        return self._observations  # Do not recompute noise/delay buffers at render rate.

    @torch.no_grad()
    def reset(self, *, seed=None):
        if self.episode_steps:
            self.manual_resets += 1  # Keyboard reset interrupts a segment, not a done.
        self._observations, extras = self.env.reset(seed=seed)
        _check_observations(self._observations, self.normalizer, self.env.device)
        self.episode_steps, self.episode_return = 0, 0.0
        return self._observations, extras

    @torch.no_grad()
    def step(self, action):
        observations, reward, terminated, truncated, extras = self.env.step(action)
        _check_observations(observations, self.normalizer, self.env.device)
        _check_policy_result("reward", reward, (1,), self.env.device)
        for name, flag in (("terminated", terminated), ("truncated", truncated)):
            if (not isinstance(flag, torch.Tensor) or flag.shape != (1,)
                    or flag.dtype != torch.bool or flag.device != torch.device(self.env.device)):
                raise ValueError(f"{name} must be boolean [1] on the env device")
        if "nan_state" in self.env.termination_manager.active_terms:
            if self.env.termination_manager.get_term("nan_state").any().item():
                raise RuntimeError("nan_state termination during playback")
        # reset() can overwrite reward/flag buffers; preserve this transition first.
        reward, terminated, truncated = (x.detach().clone() for x in (reward, terminated, truncated))
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_return += reward.item()
        if terminated.item() or truncated.item():
            self.completed_episodes += 1
            summary = {"episode": self.completed_episodes, "return": self.episode_return,
                       "steps": self.episode_steps, "duration_s": self.episode_steps * self.env.step_dt,
                       "terminated": terminated.item(), "truncated": truncated.item(),
                       "end_terms": [name for name in self.env.termination_manager.active_terms
                                     if self.env.termination_manager.get_term(name).any().item()]}
            if self.on_episode is not None:
                self.on_episode(summary)
            # This is an episode reset, not a viewer/manual reset.
            observations, extras = self.env.reset()
            _check_observations(observations, self.normalizer, self.env.device)
            self.episode_steps, self.episode_return = 0, 0.0
        self._observations = observations
        return observations, reward, terminated, truncated, extras


def _make_native_viewer(env, policy):
    # Import only AFTER configure_native_gpu(), since MuJoCo chooses its GL backend
    # at import time. Use upstream rendering, timing, camera and keyboard behavior.
    from mjlab.viewer import NativeMujocoViewer, VerbosityLevel

    class CheckedNativeViewer(NativeMujocoViewer):
        def setup(self):
            before = set(threading.enumerate())
            try:
                super().setup()
            finally:
                # MuJoCo's passive handle.close() only requests shutdown. Retain
                # its render thread so GLFW teardown cannot race interpreter exit.
                self._render_threads = [thread for thread in threading.enumerate()
                                        if thread not in before and "_launch_internal" in thread.name]

        def close(self):
            if self.viewer is not None:
                super().close()
            for thread in getattr(self, "_render_threads", ()):
                thread.join(timeout=5.0)
                if thread.is_alive():
                    raise RuntimeError("MuJoCo render thread did not finish closing")

        def tick(self):
            rendered = super().tick()
            error = self.get_status().last_error
            if error:
                # Upstream pauses on policy errors. Fail visibly rather than leave
                # an unattended bounded smoke test hanging in a paused window.
                raise RuntimeError(f"playback step failed:\n{error}")
            return rendered

    return CheckedNativeViewer(env, policy, frame_rate=60.0, verbosity=VerbosityLevel.INFO)


def run_native_playback(env, actor, critic, normalizer, *, seed=42, steps=None,
                        start_paused=False, viewer_factory=None, on_episode=None):
    """Run a native window, verify read-only policy state on exit; caller closes env.

    steps is the upstream viewer's optional step limit since its last UI reset.
    Without a limit, close the window or press Ctrl+C to end playback.
    """
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1):
        raise ValueError("steps must be positive or None")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if critic.training:
        raise ValueError("critic must be in eval mode")
    policy = MeanPolicy(actor, normalizer, action_dim=env.action_manager.total_action_dim, device=env.device)
    frozen = {"actor": cpu_snapshot(actor.state_dict()), "critic": cpu_snapshot(critic.state_dict()),
              "normalizer": normalizer.checkpoint()}
    playback, viewer = PlaybackEnv(env, normalizer, on_episode=on_episode), None
    try:
        playback.reset(seed=seed)
        viewer = (viewer_factory or _make_native_viewer)(playback, policy)
        if start_paused:
            viewer.pause()
        viewer.run(num_steps=steps)
        status = viewer.get_status()
        if status.last_error:
            raise RuntimeError(f"native viewer reported a step error:\n{status.last_error}")
    finally:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            assert_same_state(actor.state_dict(), frozen["actor"])
            assert_same_state(critic.state_dict(), frozen["critic"])
            assert_same_state(normalizer.checkpoint(), frozen["normalizer"])
            if actor.training or critic.training:
                raise RuntimeError("model mode changed during playback")
    return {"steps": playback.total_steps, "completed_episodes": playback.completed_episodes,
            "manual_resets": playback.manual_resets, "mean_policy_calls": policy.calls,
            "policy_unchanged": True, "normalizer_frozen": True,
            "normalizer_counts": {name: group.count.item() for name, group in normalizer.stats.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from training.paths import LATEST_CHECKPOINT
    parser.add_argument("--checkpoint", type=Path, default=LATEST_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Optional viewer step limit for a short check")
    parser.add_argument("--start-paused", action="store_true", help="Press Space in the viewer to begin")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("--checkpoint does not exist")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must fit in [0, 2**32)")
    settings = configure_native_gpu()
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("WSLg/native display is unavailable; run from a local WSLg session")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU simulation fallback")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from bam.mjlab import BamActuator
    from training.inspect_env import ACTION_DIM, TASK_ID

    configure_torch_backends(allow_tf32=False)
    actor, critic, normalizer = load_policy(args.checkpoint, "cuda:0")
    cfg = load_env_cfg(TASK_ID, play=False)  # Match evaluate.py's noise/DR/reward conditions.
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    print(f"[play] checkpoint={args.checkpoint.resolve()} seed={args.seed} viewer=native "
          f"actor=mean envs=1 cfg=train GPU_settings={settings}", flush=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        actuators = env.scene["robot"].actuators
        if not actuators or any(not isinstance(a, BamActuator) or
                                (a.cfg.motor_name, a.cfg.model) != ("xl330", "m6") for a in actuators):
            raise RuntimeError("expected original XL330 M6 BAM actuators")
        if env.action_manager.total_action_dim != ACTION_DIM or not math.isclose(env.step_dt, .02):
            raise RuntimeError("expected 14 actions at 50 Hz")

        def log_episode(summary):
            print(f"[play] episode={summary['episode']} return={summary['return']:.6f} "
                  f"duration_s={summary['duration_s']:.3f} terminated={summary['terminated']} "
                  f"truncated={summary['truncated']} terms={summary['end_terms']}", flush=True)

        print("[play] Space: pause/resume | Enter: reset | -/=: speed | close window: exit", flush=True)
        report = run_native_playback(env, actor, critic, normalizer, seed=args.seed, steps=args.steps,
                                     start_paused=args.start_paused, on_episode=log_episode)
        print(f"[play] PASS: {report}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
