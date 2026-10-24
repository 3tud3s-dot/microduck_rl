"""Actor/critic observation statistics, explicit boundary updates and diagnostics."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Population moments over all leading axes; last axis holds features.

    update() explicitly changes statistics in train mode. forward() only applies
    (x - mean) / (sqrt(var) + eps). Before the first update, mean=0 and var=1.
    Accumulation uses float64; output retains the input shape, dtype and device.
    """

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.dim, self.eps = dim, eps
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))

    def _validate(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("observations must be floating-point tensors")
        if x.ndim < 1 or x.shape[-1] != self.dim or x.numel() == 0:
            raise ValueError(f"expected nonempty [..., {self.dim}] observations")
        if not torch.isfinite(x).all().item():
            raise ValueError("observations contain NaN or Inf")

    def _updated_moments(self, x: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        self._validate(x)
        samples = x.detach().reshape(-1, self.dim).to(self.mean)
        batch_var, batch_mean = torch.var_mean(samples, dim=0, correction=0)
        old_count, batch_count = self.count.item(), samples.shape[0]
        total = old_count + batch_count
        if old_count == 0:
            mean, var = batch_mean, batch_var
        else:
            delta = batch_mean - self.mean
            mean = self.mean + delta * (batch_count / total)
            m2 = self.var * old_count + batch_var * batch_count
            m2 = m2 + delta.square() * (old_count * batch_count / total)
            var = m2 / total
        if not torch.isfinite(mean).all().item() or not torch.isfinite(var).all().item():
            raise ValueError("running moments overflowed; statistics were not changed")
        return total, mean, var.clamp_min(0)

    def _commit(self, state: tuple[int, torch.Tensor, torch.Tensor]) -> None:
        count, mean, var = state
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.count.fill_(count)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if self.training:
            self._commit(self._updated_moments(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        mean = self.mean.to(device=x.device)
        std = self.var.clamp_min(0).sqrt().to(device=x.device)
        return ((x.to(dtype=mean.dtype) - mean) / (std + self.eps)).to(dtype=x.dtype)


class ObservationNormalizer(nn.Module):
    """Keep separate statistics for each group, including distinct feature widths."""

    def __init__(self, dims: dict[str, int] | None = None, eps: float = 1e-2):
        super().__init__()
        dims = {"actor": 61, "critic": 76} if dims is None else dict(dims)
        if not dims:
            raise ValueError("at least one observation group is required")
        self.stats = nn.ModuleDict({name: RunningMeanStd(dim, eps) for name, dim in dims.items()})

    def _validate_groups(self, observations: dict[str, torch.Tensor]) -> None:
        if set(observations) != set(self.stats):
            raise ValueError("observation groups must match the configured groups")
        for name, stats in self.stats.items():
            stats._validate(observations[name])
        if len({tuple(x.shape[:-1]) for x in observations.values()}) != 1:
            raise ValueError("observation groups must have the same sample axes")

    @torch.no_grad()
    def update(self, observations: dict[str, torch.Tensor]) -> None:
        if not self.training:
            return
        self._validate_groups(observations)
        # Prepare every group first: an invalid critic batch must not update actor.
        pending = {name: stats._updated_moments(observations[name])
                   for name, stats in self.stats.items() if stats.training}
        for name, state in pending.items():
            self.stats[name]._commit(state)

    def forward(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._validate_groups(observations)
        return {name: stats(observations[name]) for name, stats in self.stats.items()}

    def checkpoint(self) -> dict:
        return {"dims": {name: stats.dim for name, stats in self.stats.items()},
                "eps": {name: stats.eps for name, stats in self.stats.items()},
                "training": self.training,
                "group_training": {name: stats.training for name, stats in self.stats.items()},
                "state_dict": {name: value.detach().cpu().clone()
                               for name, value in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        result = cls(checkpoint["dims"])
        for name, eps in checkpoint["eps"].items():
            if not math.isfinite(eps) or eps <= 0:
                raise ValueError("checkpoint eps must be finite and positive")
            result.stats[name].eps = eps
        result.load_state_dict(checkpoint["state_dict"], strict=True)
        result.train(checkpoint["training"])
        for name, mode in checkpoint["group_training"].items():
            result.stats[name].train(mode)
        return result


@torch.no_grad()
def observation_diagnostics(normalizer, observations):
    """Describe the CURRENT frozen transform; forward never collects statistics."""
    normalized = normalizer(observations)
    result = {}
    for name, values in normalized.items():
        stats = normalizer.stats[name]
        std = stats.var.sqrt() + stats.eps
        result[name] = {"count": stats.count.item(), "effective_std_min": std.min().item(),
                        "effective_std_max": std.max().item(),
                        "normalized_mean": values.double().mean().item(),
                        "normalized_std": values.double().std(correction=0).item(),
                        "normalized_max_abs": values.abs().max().item(),
                        "fraction_abs_gt_5": (values.abs() > 5).double().mean().item()}
    return result


@torch.no_grad()
def update_between_iterations(normalizer, observations):
    """Explicit running-mode boundary update, AFTER PPO; return frozen again.

    Count each raw policy-input observation once, not normalized inputs, terminal
    next observations or repeated PPO epochs. This changes the raw-observation
    policy between iterations; log the input shift rather than hiding the change.
    """
    if normalizer.training or any(stats.training for stats in normalizer.stats.values()):
        raise ValueError("boundary update requires a frozen normalizer")
    before = normalizer(observations)
    counts = {name: stats.count.item() for name, stats in normalizer.stats.items()}
    normalizer.train()
    try:
        normalizer.update(observations)
    finally:
        normalizer.eval()
    after = normalizer(observations)
    result = {}
    for name, stats in normalizer.stats.items():
        samples = observations[name].numel() // stats.dim
        if stats.count.item() != counts[name] + samples:
            raise RuntimeError("normalizer did not count every [time, env] sample exactly once")
        result[name] = {"count_before": counts[name], "count_after": stats.count.item(),
                        "samples_added": samples,
                        "normalized_input_shift_rms": (after[name].double() - before[name]).square().mean().sqrt().item()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new file")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the source rollout")
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    batch = source["rollout"]
    observations = batch["observations"]
    normalizer = ObservationNormalizer(eps=args.eps)
    normalizer._validate_groups(observations)
    time_steps = observations["actor"].shape[0]
    for start, stop in ((0, 1), (1, 8), (8, time_steps)):
        if start < time_steps:
            normalizer.update({name: x[start:min(stop, time_steps)]
                               for name, x in observations.items()})
    for name, stats in normalizer.stats.items():
        samples = observations[name].double().reshape(-1, stats.dim)
        torch.testing.assert_close(stats.mean, samples.mean(0), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(stats.var, samples.var(0, correction=0), rtol=1e-10, atol=1e-12)
        if stats.count.item() != samples.shape[0]:
            raise RuntimeError(f"{name}: incorrect sample count")
        print(f"[normalization] {name}: input={tuple(observations[name].shape)} "
              f"count={stats.count.item()} features={stats.dim} batch_reference=PASS")

    normalizer.eval()
    frozen = {name: value.clone() for name, value in normalizer.state_dict().items()}
    normalized = normalizer(observations)
    normalized_next = normalizer(batch["next_observations"])
    normalizer.update(observations)  # Eval must ignore explicit updates too.
    if not all(torch.equal(value, normalizer.state_dict()[name]) for name, value in frozen.items()):
        raise RuntimeError("statistics changed while frozen")
    for name, output in normalized.items():
        stats = normalizer.stats[name]
        reference = ((observations[name].double() - stats.mean) /
                     (stats.var.sqrt() + stats.eps)).to(observations[name].dtype)
        torch.testing.assert_close(output, reference)
        if not torch.isfinite(output).all().item() or not torch.isfinite(normalized_next[name]).all().item():
            raise RuntimeError(f"{name}: non-finite normalized observations")
        print(f"[normalization] {name}: output={tuple(output.shape)} dtype={output.dtype} "
              f"finite=True max_abs={output.abs().max().item():.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"normalizer": normalizer.checkpoint(), "source_rollout": str(args.input),
                "fit_source": "observations only", "normalized_observations": normalized,
                "normalized_next_observations": normalized_next}, args.output)
    restored = ObservationNormalizer.from_checkpoint(
        torch.load(args.output, map_location="cpu", weights_only=True)["normalizer"])
    for name, output in restored(observations).items():
        if not torch.equal(output, normalized[name]):
            raise RuntimeError(f"{name}: reload output differs")
    if restored.training or any(stats.training for stats in restored.stats.values()):
        raise RuntimeError("reload did not restore eval mode")
    # The last returned observation is one new sample per env beyond the fitted inputs.
    continuation = {name: x[-1:] for name, x in batch["next_observations"].items()}
    for module in (normalizer, restored):
        module.train()
        module.update(continuation)
    for name, value in normalizer.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise RuntimeError("resumed statistics differ")
    counts = {name: stats.count.item() for name, stats in restored.stats.items()}
    print(f"[normalization] frozen=PASS reload=PASS continuation_counts={counts} resume=PASS")
    print(f"[normalization] saved={args.output} (frozen state before continuation)")
    print("[normalization] PASS: offline CPU probe only; no reward/action normalization or training")


if __name__ == "__main__":
    main()
