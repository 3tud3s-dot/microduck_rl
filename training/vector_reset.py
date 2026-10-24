"""Keep continuing rows intact across mjlab 1.3's full-batch reset observation pass.

Only training code uses this adapter; mjlab and the original task are unchanged.
The public reset still runs all normal reset events/managers for the selected IDs.
"""

from contextlib import contextmanager

import torch


def _row_tensors(owner, num_envs):
    """Snapshot per-env tensors in an object's attributes and nested dictionaries."""
    def visit(container, items, is_dict):
        for name, value in items:
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == num_envs:
                yield container, name, is_dict, value.clone()
            elif isinstance(value, dict):
                yield from visit(value, value.items(), True)

    return list(visit(owner, vars(owner).items(), False))


def _restore_rows(snapshots, keep):
    for owner, name, is_dict, saved in snapshots:
        current = owner[name] if is_dict else getattr(owner, name)
        if current.shape != saved.shape:
            raise RuntimeError(f"reset changed per-env state shape: {name}")
        current[keep] = saved[keep]


def _temporal_buffers(manager):
    for terms in manager._group_obs_term_delay_buffer.values():
        for delay in terms.values():
            yield delay._buffer, delay
    for terms in manager._group_obs_term_history_buffer.values():
        for history in terms.values():
            yield history, None


def _save_temporal(circular, delay, n):
    if circular._buffer is None:
        raise RuntimeError("history must be initialized before partial reset")
    return (circular, circular._pointer, circular._buffer.clone(),
            circular._num_pushes.clone(), _row_tensors(delay, n) if delay is not None else [])


def _restore_temporal(snapshot, ids, keep):
    circular, pointer, storage, pushes, delay_rows = snapshot
    shifted = torch.roll(circular._buffer[:, ids], shifts=pointer - circular._pointer, dims=0)
    circular._buffer.copy_(storage)
    circular._buffer[:, ids] = shifted
    circular._pointer = pointer
    circular._num_pushes[keep] = pushes[keep]
    _restore_rows(delay_rows, keep)


@contextmanager
def _isolate_actuator_writes(env, ids, keep):
    """Undo the extra all-row actuator compute BEFORE reset's sim.forward().

    Public reset calls scene.write_data_to_sim(), which otherwise advances motor
    delay buffers and BAM voltage/friction state for continuing robots as well.
    The override is local to this scene instance and removed even on failure.
    """
    scene = getattr(env, "scene", None)
    if scene is None:
        yield
        return
    original = scene.write_data_to_sim
    had_override = "write_data_to_sim" in vars(scene)
    old_override = vars(scene).get("write_data_to_sim")
    ctrl = env.sim.data.ctrl.clone()
    fields = {name: getattr(env.sim.model, name).clone()
              for name in ("dof_frictionloss", "dof_damping")}
    owners, temporal, seen = [], [], set()
    for entity in scene.entities.values():
        for actuator in entity.actuators:
            owners.append(actuator)
            bam = getattr(actuator, "_bam_model", None)
            if bam is not None:
                owners.append(bam.actuator)
            delay = getattr(actuator, "_delay_buffer", None)
            if delay is not None and id(delay) not in seen:
                seen.add(id(delay))
                temporal.append(_save_temporal(delay._buffer, delay, env.num_envs))
    rows = [s for owner in owners for s in _row_tensors(owner, env.num_envs)]

    def write_selected():
        original()
        env.sim.data.ctrl[keep] = ctrl[keep]
        for name, saved in fields.items():
            current = getattr(env.sim.model, name)
            if current.shape[0] == env.num_envs:
                current[keep] = saved[keep]
            elif keep.any() and not torch.equal(current, saved):
                raise RuntimeError(f"actuator reset changed shared model field: {name}")
        _restore_rows(rows, keep)
        for saved in temporal:
            _restore_temporal(saved, ids, keep)

    scene.write_data_to_sim = write_selected
    try:
        yield
    finally:
        if had_override:
            scene.write_data_to_sim = old_override
        else:
            del scene.write_data_to_sim


@torch.no_grad()
def reset_done_envs(env, env_ids, transition_observations):
    """Reset done rows, preserving other rows' observations and temporal state.

    mjlab.reset(env_ids) computes observations for ALL rows, advancing circular
    pointers and delay counters even for continuing episodes. Save their state,
    then undo that extra append. Reset rows retain their newly backfilled history.
    Circular storage is rotated to the old pointer before merging rows; simply
    restoring tensor rows while leaving the new global pointer is incorrect.

    Reset consumes the usual shared RNG stream; independent RNG streams per env
    are not promised. This does not roll back reset events or curriculum updates.
    """
    manager = getattr(env, "observation_manager", None)
    if manager is None:  # Simple environments without managed delay/history.
        groups = getattr(env.cfg, "observations", {}).values()
        if any((getattr(g, "history_length", 0) or 0) > 0 or any(
                getattr(t, "delay_max_lag", 0) > 0 or getattr(t, "history_length", 0) > 0
                for t in g.terms.values()) for g in groups):
            raise ValueError("delayed/history observations require a supported observation manager")
        reset_obs, _ = env.reset(env_ids=env_ids)
    else:
        required = ("_group_obs_term_delay_buffer", "_group_obs_term_history_buffer",
                    "_group_obs_class_instances", "_group_obs_class_term_cfgs", "_obs_buffer")
        if any(not hasattr(manager, name) for name in required):
            raise RuntimeError("unsupported mjlab observation manager reset layout")
        keep = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        keep[env_ids] = False
        temporal = [_save_temporal(circular, delay, env.num_envs)
                    for circular, delay in _temporal_buffers(manager)]
        owners = list(getattr(getattr(env, "command_manager", None), "_terms", {}).values())
        owners += list(manager._group_obs_class_instances.values())
        for configs in manager._group_obs_class_term_cfgs.values():
            owners += [cfg.func for cfg in configs]
        rows = [snapshot for owner in owners for snapshot in _row_tensors(owner, env.num_envs)]

        with _isolate_actuator_writes(env, env_ids, keep):
            reset_obs, _ = env.reset(env_ids=env_ids)
        for saved in temporal:
            _restore_temporal(saved, env_ids, keep)
        _restore_rows(rows, keep)

    observations = {name: value.clone() for name, value in transition_observations.items()}
    for name in observations:
        observations[name][env_ids] = reset_obs[name][env_ids]
    if manager is not None:
        manager._obs_buffer = observations
        env.obs_buf = observations
    return observations
