"""Check the original training environment with one CUDA env and zero actions."""

import argparse
import math

import torch

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
NUM_ENVS = 1
OBS_DIMS = {"actor": 61, "critic": 76}
ACTION_DIM = 14


def check_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise RuntimeError(f"{name}: expected tensor shape {shape}, got {value!r}")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name}: contains NaN or Inf")


def check_observations(observations: dict[str, torch.Tensor]) -> None:
    for name, width in OBS_DIMS.items():
        check_tensor(name, observations[name], (NUM_ENVS, width))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this probe does not fall back to CPU")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from bam.mjlab import BamActuator

    configure_torch_backends()
    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = args.seed
    print(f"[inspect] task={TASK_ID} mode=train envs={NUM_ENVS} device=cuda:0 "
          f"seed={args.seed} steps={args.steps}", flush=True)
    print(f"[inspect] GPU={torch.cuda.get_device_name(0)}", flush=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        robot = env.scene["robot"]
        if not robot.actuators or not all(isinstance(a, BamActuator) for a in robot.actuators):
            raise RuntimeError("The robot is not using BAM actuators")
        for actuator in robot.actuators:
            if (actuator.cfg.motor_name, actuator.cfg.model) != ("xl330", "m6"):
                raise RuntimeError("Expected the original XL330 M6 actuator model")
            print(f"[inspect] BAM={type(actuator).__name__} motor=xl330 model=m6")
        if env.action_manager.total_action_dim != ACTION_DIM:
            raise RuntimeError("Expected 14 action dimensions")
        if not math.isclose(env.step_dt, 0.02):
            raise RuntimeError(f"Expected a 0.02 s control period, got {env.step_dt}")
        print(f"[inspect] physics_dt={env.physics_dt} decimation={cfg.decimation} "
              f"step_dt={env.step_dt} auto_reset={cfg.auto_reset}")
        # Zero action means zero offset from HOME, not zero torque or motor shutdown.
        action = torch.zeros((NUM_ENVS, ACTION_DIM), device=env.device)
        ended = 0
        with torch.inference_mode():
            observations, extras = env.reset()
            check_observations(observations)
            if not isinstance(extras, dict):
                raise RuntimeError("reset() extras must be a dict")
            shapes = {name: tuple(observations[name].shape) for name in OBS_DIMS}
            print(f"[inspect] reset() -> 2 items; obs={shapes}; action={tuple(action.shape)}")
            for step in range(1, args.steps + 1):
                observations, reward, terminated, truncated, extras = env.step(action)
                check_observations(observations)
                check_tensor("reward", reward, (NUM_ENVS,))
                for name, flag in (("terminated", terminated), ("truncated", truncated)):
                    check_tensor(name, flag, (NUM_ENVS,))
                    if flag.dtype != torch.bool:
                        raise RuntimeError(f"{name}: expected torch.bool")
                if not isinstance(extras, dict):
                    raise RuntimeError("step() extras must be a dict")
                if env.termination_manager.get_term("nan_state").any().item():
                    raise RuntimeError("The environment triggered its nan_state termination")
                ended += int((terminated | truncated).sum().item())
                print(f"[inspect] step={step:02d} reward={reward.item():.6f} "
                      f"terminated={terminated.item()} truncated={truncated.item()} finite=True")
        print(f"[inspect] PASS: reset=2 items, step=5 items, shapes/finite/BAM checked; "
              f"{args.steps * env.step_dt:.3f} s simulated, ended={ended}")
        print("[inspect] Interface check only; no policy training or standing assessment.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
