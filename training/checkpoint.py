"""Training resume with explicit simulator reset; no claim of trajectory replay."""

import random

import numpy as np
import torch


def capture_training_state(record, env):
    numpy_rng = np.random.get_state()
    return {"version": 1, "iteration": record["iteration"], "env_steps": record["env_steps"],
            "vector_steps": record["vector_steps"], "num_envs": env.num_envs,
            "common_step_counter": env.common_step_counter,
            "normalization_mode": record["normalization"]["mode"],
            "rng": {"torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "python": random.getstate(),
                    "numpy": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:])},
            "simulator_state_saved": False,
            "unfinished_episodes_discarded_on_resume": record["episodes"]["unfinished_count"]}


def restore_training_state(source, env, actor, critic, normalizer, optimizer, *, normalization_mode):
    from training.train import assert_same_state

    state = source.get("training_state")
    if not state or state["version"] != 1:
        raise ValueError("--resume requires a version-1 training_state checkpoint")
    if env.num_envs != state["num_envs"] or normalization_mode != state["normalization_mode"]:
        raise ValueError("resume must retain num_envs and normalization mode")
    for key in ("iteration", "env_steps", "vector_steps", "common_step_counter"):
        if not isinstance(state[key], int) or state[key] < 0:
            raise ValueError("invalid resume counter: " + key)
    assert_same_state(actor.state_dict(), source["actor_state_dict"])
    assert_same_state(critic.state_dict(), source["critic_state_dict"])
    assert_same_state(normalizer.checkpoint(), source["normalizer"])
    assert_same_state(optimizer.state_dict(), source["optimizer_state_dict"])
    # Original reset invokes curriculum.compute, using this restored counter.
    env.common_step_counter = state["common_step_counter"]
    rng = state["rng"]
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if rng["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in rng["torch_cuda"]])
    random.setstate(rng["python"])
    numpy_rng = rng["numpy"]
    np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    return {key: state[key] for key in ("iteration", "env_steps", "vector_steps")}
