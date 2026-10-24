"""GAE advantages and critic targets from a rollout with true next-state values."""

import argparse
import math
from pathlib import Path

import torch


@torch.no_grad()
def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                bootstrap_values: torch.Tensor, terminated: torch.Tensor,
                truncated: torch.Tensor, *, gamma: float = 0.99,
                gae_lambda: float = 0.95) -> dict[str, torch.Tensor]:
    """All inputs are [time, env]; outputs are detached, unnormalized targets.

    bootstrap_values[t] estimates the true post-action state BEFORE reset, including
    the last collected transition. Termination removes its bootstrap contribution.
    Both termination and time truncation stop GAE recursion across episode resets;
    truncation alone still bootstraps from its own terminal observation.
    If both flags are true, termination wins. A collection horizon is not a done.
    """
    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if (not isinstance(rewards, torch.Tensor) or not rewards.is_floating_point()
            or rewards.ndim != 2 or rewards.numel() == 0):
        raise ValueError("rewards must be a nonempty floating [time, env] tensor")
    for name, value in (("rewards", rewards), ("values", values),
                        ("bootstrap_values", bootstrap_values),
                        ("terminated", terminated), ("truncated", truncated)):
        dtype = torch.bool if name in ("terminated", "truncated") else rewards.dtype
        if (not isinstance(value, torch.Tensor) or value.shape != rewards.shape
                or value.device != rewards.device or value.dtype != dtype):
            raise ValueError(f"{name}: shape/device/dtype must match [time, env] contract")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name}: contains NaN or Inf")

    next_value = torch.where(terminated, 0.0, bootstrap_values)
    deltas = rewards + gamma * next_value - values
    continuation = ~(terminated | truncated)
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        carry = deltas[step] + gamma * gae_lambda * torch.where(
            continuation[step], carry, 0.0)
        advantages[step] = carry
    returns = advantages + values
    if not torch.isfinite(advantages).all().item() or not torch.isfinite(returns).all().item():
        raise ValueError("GAE overflowed; targets contain NaN or Inf")
    return {"advantages": advantages, "returns": returns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new file")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the source rollout")
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    batch = source["rollout"]
    if "bootstrap_values" not in batch or "transition_next_observations" not in batch:
        raise ValueError("rollout lacks pre-reset bootstrap data; recollect with --agent policy")
    targets = compute_gae(*(batch[name] for name in (
        "rewards", "values", "bootstrap_values", "terminated", "truncated")),
        gamma=args.gamma, gae_lambda=args.gae_lambda)
    torch.testing.assert_close(targets["returns"], targets["advantages"] + batch["values"])
    # At episode boundaries and the collection horizon there is no recursive tail.
    boundary = (batch["terminated"] | batch["truncated"]).clone()
    boundary[-1] = True
    deltas = (batch["rewards"] + args.gamma * torch.where(
        batch["terminated"], 0.0, batch["bootstrap_values"]) - batch["values"])
    torch.testing.assert_close(targets["advantages"][boundary], deltas[boundary])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": {"source_rollout": str(args.input.resolve()),
                             "gamma": args.gamma, "gae_lambda": args.gae_lambda,
                             "advantage_normalization": False}, **targets}, args.output)
    restored = torch.load(args.output, map_location="cpu", weights_only=True)
    for name, value in targets.items():
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
        print(f"[advantages] {name}={tuple(value.shape)} finite=True "
              f"range=({value.min().item():.5f}, {value.max().item():.5f})")
    print(f"[advantages] gamma={args.gamma} lambda={args.gae_lambda} "
          f"terminated={batch['terminated'].sum().item()} "
          f"truncated={batch['truncated'].sum().item()} boundary_checks=PASS")
    print(f"[advantages] saved={args.output}")
    print("[advantages] PASS: targets only; no normalization or policy/value updates")


if __name__ == "__main__":
    main()
