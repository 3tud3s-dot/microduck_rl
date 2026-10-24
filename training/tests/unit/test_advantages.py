"""Hand-calculated GAE cases, including asynchronous reset and horizon boundaries."""

import unittest

import torch

from training.advantages import compute_gae


class AdvantageTests(unittest.TestCase):
    def test_ordinary_steps_and_unfinished_horizon_use_true_next_value(self):
        rewards = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
        values = torch.tensor([[.5], [1.], [1.5]], requires_grad=True)
        next_values = torch.tensor([[1.], [1.5], [2.]], requires_grad=True)
        flags = torch.zeros(3, 1, dtype=torch.bool)
        result = compute_gae(rewards, values, next_values, flags, flags,
                             gamma=.9, gae_lambda=.8)
        torch.testing.assert_close(result["advantages"], torch.tensor([[4.80272], [4.726], [3.3]]))
        torch.testing.assert_close(result["returns"], torch.tensor([[5.30272], [5.726], [4.8]]))
        for x in result.values():
            self.assertFalse(x.requires_grad)
            self.assertFalse(torch.is_inference(x))
        torch.testing.assert_close(values.detach(), torch.tensor([[.5], [1.], [1.5]]))

    def test_termination_truncation_and_both_flags_do_not_leak_next_episode(self):
        # Three envs end at t=1 for different reasons; reward=100 at t=2 belongs
        # to new episodes and must not change their preceding advantages.
        rewards = torch.tensor([[1.], [2.], [100.]]).expand(3, 3).clone()
        values = torch.tensor([[.5], [1.], [1.5]]).expand(3, 3).clone()
        next_values = torch.tensor([[1.], [4.], [2.]]).expand(3, 3).clone()
        terminated = torch.zeros(3, 3, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        terminated[1, [0, 2]] = True
        truncated[1, [1, 2]] = True
        result = compute_gae(rewards, values, next_values, terminated, truncated,
                             gamma=.9, gae_lambda=.8)
        expected = torch.tensor([[2.12, 4.712, 2.12], [1., 4.6, 1.], [100.3, 100.3, 100.3]])
        torch.testing.assert_close(result["advantages"], expected)
        rewards[-1].fill_(100000)
        again = compute_gae(rewards, values, next_values, terminated, truncated,
                            gamma=.9, gae_lambda=.8)
        torch.testing.assert_close(again["advantages"][:-1], expected[:-1])

    def test_lambda_zero_gamma_zero_monte_carlo_and_single_step(self):
        r = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
        v, n = torch.tensor([[.5], [1.], [1.5]], dtype=r.dtype), torch.tensor([[1.], [1.5], [0.]], dtype=r.dtype)
        terminated = torch.tensor([[False], [False], [True]])
        truncated = torch.zeros_like(terminated)
        result = compute_gae(r, v, n, terminated, truncated, gamma=1, gae_lambda=1)
        torch.testing.assert_close(result["returns"], torch.tensor([[6.], [5.], [3.]], dtype=r.dtype))
        result = compute_gae(r, v, n, terminated, truncated, gamma=.9, gae_lambda=0)
        torch.testing.assert_close(result["advantages"], r + .9 * n - v)
        result = compute_gae(r, v, n, terminated, truncated, gamma=0)
        torch.testing.assert_close(result["returns"], r)
        result = compute_gae(r[:1], v[:1], n[:1], terminated[:1], truncated[:1], gamma=.9)
        torch.testing.assert_close(result["advantages"], torch.tensor([[1.4]], dtype=r.dtype))

    def test_validation_rejects_broadcasting_nonfinite_and_overflow(self):
        r = torch.ones(2, 1)
        flag = torch.zeros_like(r, dtype=torch.bool)
        for kwargs in ({"gamma": -1}, {"gamma": float("nan")}, {"gae_lambda": 1.01}):
            with self.assertRaises(ValueError):
                compute_gae(r, r, r, flag, flag, **kwargs)
        bad_cases = [
            (r, r[:, 0], r, flag, flag),
            (r, r.double(), r, flag, flag),
            (r, r, r, flag.float(), flag),
            (r, r, r * float("nan"), flag, flag),
            (r[:0], r[:0], r[:0], flag[:0], flag[:0]),
            (r[:, 0], r[:, 0], r[:, 0], flag[:, 0], flag[:, 0]),
        ]
        for inputs in bad_cases:
            with self.assertRaises(ValueError):
                compute_gae(*inputs)
        huge = torch.full((2, 1), 3e38)
        with self.assertRaisesRegex(ValueError, "overflowed"):
            compute_gae(huge, -huge, huge, flag, flag)


if __name__ == "__main__":
    unittest.main()
