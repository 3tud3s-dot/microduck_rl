"""Actor/critic networks for already-normalized observations; offline sampling probe."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal


def make_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 1
           for dim in (input_dim, *hidden_dims, output_dim)):
        raise ValueError("network dimensions must be positive integers")
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def network_input(x: torch.Tensor, dim: int, weight: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("normalized observations must be floating-point tensors")
    if x.ndim < 1 or x.shape[-1] != dim or x.numel() == 0:
        raise ValueError(f"expected nonempty [..., {dim}] observations")
    if x.device != weight.device:
        raise ValueError("move the network and observations to the same device")
    if not torch.isfinite(x).all().item():
        raise ValueError("normalized observations contain NaN or Inf")
    return x.to(dtype=weight.dtype)


class GaussianActor(nn.Module):
    """Normalized actor obs -> linear mean + diagonal Gaussian; no action clipping.

    The effective standard deviation is exp(clamp(log_std, -5, 2)). Sampling is
    explicit: eval() does not turn sample() into deterministic mean actions.
    """

    def __init__(self, obs_dim: int = 61, action_dim: int = 14,
                 hidden_dims: tuple[int, ...] = (64, 64), init_std: float = 0.05):
        super().__init__()
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        self.config = {"obs_dim": obs_dim, "action_dim": action_dim,
                       "hidden_dims": tuple(hidden_dims), "init_std": init_std}
        self.mean_net = make_mlp(obs_dim, action_dim, hidden_dims)
        nn.init.normal_(self.mean_net[-1].weight, std=0.01)
        nn.init.zeros_(self.mean_net[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std)))

    def forward(self, normalized_actor_obs: torch.Tensor) -> torch.Tensor:
        x = network_input(normalized_actor_obs, self.config["obs_dim"], self.mean_net[0].weight)
        return self.mean_net(x)

    def distribution(self, normalized_actor_obs: torch.Tensor) -> Normal:
        mean = self(normalized_actor_obs)
        if not torch.isfinite(self.log_std).all().item():
            raise ValueError("actor log_std contains NaN or Inf")
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        return Normal(mean, std, validate_args=True)

    def sample(self, normalized_actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(normalized_actor_obs)
        actions = distribution.sample()  # Fixed samples for the score-function gradient.
        return actions, distribution.log_prob(actions).sum(dim=-1)

    def evaluate_actions(self, normalized_actor_obs: torch.Tensor,
                         actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute density of these exact actions; keep gradients for later PPO."""
        distribution = self.distribution(normalized_actor_obs)
        if actions.shape != distribution.mean.shape:
            raise ValueError("actions must match the full mean shape; no broadcasting")
        if actions.dtype != distribution.mean.dtype or actions.device != distribution.mean.device:
            raise ValueError("actions must have the same dtype and device as the distribution")
        return distribution.log_prob(actions).sum(dim=-1), distribution.entropy().sum(dim=-1)


class Critic(nn.Module):
    """Normalized critic obs -> scalar value, preserving every leading batch axis."""

    def __init__(self, obs_dim: int = 76, hidden_dims: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.config = {"obs_dim": obs_dim, "hidden_dims": tuple(hidden_dims)}
        self.value_net = make_mlp(obs_dim, 1, hidden_dims)

    def forward(self, normalized_critic_obs: torch.Tensor) -> torch.Tensor:
        x = network_input(normalized_critic_obs, self.config["obs_dim"], self.value_net[0].weight)
        return self.value_net(x).squeeze(-1)


def main() -> None:
    from training.normalization import ObservationNormalizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-std", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new file")
    if args.output.resolve() in {args.rollout.resolve(), args.normalization.resolve()}:
        parser.error("--output must not overwrite either input artifact")
    torch.manual_seed(args.seed)
    raw = torch.load(args.rollout, map_location="cpu", weights_only=True)["rollout"]["observations"]
    state = torch.load(args.normalization, map_location="cpu", weights_only=True)["normalizer"]
    normalizer = ObservationNormalizer.from_checkpoint(state)
    normalizer.eval()
    frozen = {name: value.clone() for name, value in normalizer.state_dict().items()}
    actor, critic = GaussianActor(init_std=args.init_std).eval(), Critic().eval()
    with torch.no_grad():
        normalized = normalizer(raw)  # Normalize raw observations exactly once.
        mean = actor(normalized["actor"])
        distribution = actor.distribution(normalized["actor"])
        actions, log_prob = actor.sample(normalized["actor"])
        recomputed, entropy = actor.evaluate_actions(normalized["actor"], actions)
        values = critic(normalized["critic"])
        leading = raw["actor"].shape[:-1]
        for name, tensor in {"mean": mean, "std": distribution.stddev, "actions": actions,
                             "log_prob": log_prob, "entropy": entropy, "values": values}.items():
            expected = (*leading, 14) if name in {"mean", "std", "actions"} else leading
            if tensor.shape != expected or not torch.isfinite(tensor).all().item():
                raise RuntimeError(f"{name}: expected finite tensor shaped {expected}")
            print(f"[models] {name}={tuple(tensor.shape)} dtype={tensor.dtype} finite=True")
        manual = (-0.5 * ((actions - mean) / distribution.stddev).square()
                  - distribution.stddev.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        torch.testing.assert_close(log_prob, manual)
        torch.testing.assert_close(log_prob, recomputed, rtol=0, atol=0)
        torch.testing.assert_close(mean, actor(normalized["actor"]), rtol=0, atol=0)
        if not distribution.stddev.gt(0).all().item():
            raise RuntimeError("standard deviations must be positive")
        if not all(torch.equal(value, normalizer.state_dict()[name]) for name, value in frozen.items()):
            raise RuntimeError("network probe changed normalization statistics")

        checkpoint = {"actor_config": actor.config, "critic_config": critic.config,
                      "actor_state_dict": {name: value.detach().cpu().clone()
                                           for name, value in actor.state_dict().items()},
                      "critic_state_dict": {name: value.detach().cpu().clone()
                                            for name, value in critic.state_dict().items()},
                      "normalizer": normalizer.checkpoint(),
                      "actor_training": actor.training, "critic_training": critic.training,
                      "seed": args.seed, "actions": actions, "log_prob": log_prob}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.output)
        loaded = torch.load(args.output, map_location="cpu", weights_only=True)
        restored_actor = GaussianActor(**loaded["actor_config"])
        restored_critic = Critic(**loaded["critic_config"])
        restored_actor.load_state_dict(loaded["actor_state_dict"], strict=True)
        restored_critic.load_state_dict(loaded["critic_state_dict"], strict=True)
        restored_actor.train(loaded["actor_training"])
        restored_critic.train(loaded["critic_training"])
        restored_norm = ObservationNormalizer.from_checkpoint(loaded["normalizer"])
        restored_obs = restored_norm(raw)
        torch.testing.assert_close(restored_actor(restored_obs["actor"]), mean, rtol=0, atol=0)
        torch.testing.assert_close(restored_critic(restored_obs["critic"]), values, rtol=0, atol=0)
        restored_log_prob, _ = restored_actor.evaluate_actions(restored_obs["actor"], loaded["actions"])
        torch.testing.assert_close(restored_log_prob, loaded["log_prob"], rtol=0, atol=0)
    counts = {name: stats.count.item() for name, stats in normalizer.stats.items()}
    print(f"[models] CPU seed={args.seed} hidden_dims=(64, 64) "
          f"std_range=({distribution.stddev.min().item():.5f}, {distribution.stddev.max().item():.5f})")
    print(f"[models] actions_range=({actions.min().item():.5f}, {actions.max().item():.5f}) "
          f"joint_log_prob_range=({log_prob.min().item():.5f}, {log_prob.max().item():.5f})")
    print(f"[models] density_reference=PASS same_action_recompute=PASS "
          f"normalizer_counts={counts} frozen=PASS reload=PASS")
    print(f"[models] saved={args.output}")
    print("[models] PASS: forward/sampling only; no parameter update, RNG restore or behavior claim")


if __name__ == "__main__":
    main()
