"""CPU checks for body-frame diagnostic metrics and honest episode endpoints."""

import copy
import math
import unittest

import torch

from training.diagnose import aggregate_episode, compute_measurements


class MeasurementTests(unittest.TestCase):
    def test_body_frame_errors_preserve_sign_and_command_has_no_vertical_velocity(self):
        linear = torch.tensor([[2., -1., 9.]], dtype=torch.float64)
        angular = torch.tensor([[.2, .3, .75]], dtype=torch.float64)
        command = torch.tensor([[1., .5, -.25]], dtype=torch.float64)
        gravity = torch.tensor([[0., 0., -1.]], dtype=torch.float64)
        result = compute_measurements(linear, angular, gravity, command)
        torch.testing.assert_close(result["linear_error_b"], torch.tensor([[1., -1.5, 9.]], dtype=torch.float64))
        torch.testing.assert_close(result["angular_error_b"], torch.tensor([[.2, .3, 1.]], dtype=torch.float64))
        torch.testing.assert_close(result["linear_xy_error_mps"], torch.tensor([math.sqrt(3.25)], dtype=torch.float64))
        torch.testing.assert_close(result["yaw_error_radps"], torch.tensor([1.], dtype=torch.float64))
        torch.testing.assert_close(result["tilt_deg"], torch.zeros(1, dtype=torch.float64))

    def test_upright_sideways_inverted_and_scaled_gravity_with_arbitrary_batch_axes(self):
        gravity = torch.tensor([[[0., 0., -2.], [2., 0., 0.], [0., 0., 2.]],
                                [[0., 0., -1.], [0., -3., 0.], [0., 0., 1.]]], dtype=torch.float64)
        zeros = torch.zeros_like(gravity)
        result = compute_measurements(zeros, zeros, gravity, zeros)
        torch.testing.assert_close(result["tilt_deg"], torch.tensor([[0., 90., 180.], [0., 90., 180.]], dtype=torch.float64))
        self.assertEqual(result["linear_error_b"].shape, (2, 3, 3))
        self.assertEqual(result["linear_xy_error_mps"].shape, (2, 3))

    def test_turning_error_is_signed_in_both_directions(self):
        zero = torch.zeros(2, 3)
        angular = zero.clone()
        angular[:, 2] = torch.tensor([-.25, .25])
        command = zero.clone()
        command[:, 2] = torch.tensor([.5, -.5])
        gravity = zero.clone()
        gravity[:, 2] = -1
        result = compute_measurements(zero, angular, gravity, command)
        torch.testing.assert_close(result["yaw_error_radps"], torch.tensor([-.75, .75]))

    def test_measurement_input_shape_nonfinite_and_zero_gravity_are_rejected(self):
        valid = [torch.zeros(2, 3) for _ in range(4)]
        valid[2][:, 2] = -1
        for position in range(4):
            for invalid in (torch.zeros(2, 2), torch.zeros(1, 3), torch.full((2, 3), float("nan"))):
                with self.subTest(position=position, shape=invalid.shape):
                    args = list(valid)
                    args[position] = invalid
                    with self.assertRaises(ValueError):
                        compute_measurements(*args)
        with self.assertRaises(ValueError):
            compute_measurements(valid[0], valid[1], torch.zeros(2, 3), valid[3])


class EpisodeMetricTests(unittest.TestCase):
    def trace(self):
        return {
            "commands": torch.tensor([[1., 0., .5]] * 3, dtype=torch.float64),
            "linear_velocity_b": torch.tensor([[2., 0., 9.], [1., 2., -3.], [1., 0., 2.]], dtype=torch.float64),
            "angular_velocity_b": torch.tensor([[0., 0., 1.5], [0., 0., -.5], [0., 0., .5]], dtype=torch.float64),
            "projected_gravity_b": torch.tensor([[0., 0., -1.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64),
            "actions": torch.tensor([[.1, -.1], [.2, -.2], [.3, -.3]], dtype=torch.float64),
            "action_std": torch.tensor([[.1, .3], [.2, .4], [.3, .5]], dtype=torch.float64),
            "rewards": torch.tensor([1., 2., -.5], dtype=torch.float64),
            "terminated": torch.tensor([False, False, True]),
            "truncated": torch.tensor([False, False, False]),
        }

    def test_rmse_counts_both_linear_axes_and_duration_includes_terminal_transition(self):
        trace = self.trace()
        before = copy.deepcopy(trace)
        result = aggregate_episode(trace, step_dt=.02, end_terms=["fell_over"])
        self.assertEqual(result["steps"], 3)
        self.assertAlmostEqual(result["duration_s"], .06)
        self.assertAlmostEqual(result["episode_return"], 2.5)
        self.assertAlmostEqual(result["linear_xy_rmse_mps"], math.sqrt(5 / 3))
        self.assertAlmostEqual(result["yaw_rmse_radps"], math.sqrt(2 / 3))
        self.assertAlmostEqual(result["tilt_mean_deg"], 90.)
        self.assertAlmostEqual(result["tilt_max_deg"], 180.)
        self.assertAlmostEqual(result["mean_action_std"], .3)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertFalse(result["evaluation_cutoff"])
        self.assertTrue(result["fall_detected"])
        self.assertAlmostEqual(result["fall_time_s"], .06)
        self.assertEqual(result["active_end_terms"], ["fell_over"])
        for name, value in before.items():
            torch.testing.assert_close(trace[name], value, rtol=0, atol=0)

    def test_fall_timeout_other_termination_and_cap_are_distinguished(self):
        cases = ((True, False, ["fell_over"], True, False),
                 (True, True, ["fell_over", "time_out"], True, False),
                 (False, True, ["time_out"], False, False),
                 (True, False, ["joint_limit"], False, False),
                 (False, False, [], False, True))
        for terminated, truncated, terms, fall, cutoff in cases:
            with self.subTest(terms=terms, terminated=terminated, truncated=truncated):
                trace = self.trace()
                trace["terminated"][-1] = terminated
                trace["truncated"][-1] = truncated
                result = aggregate_episode(trace, step_dt=.02, end_terms=terms)
                self.assertEqual(result["terminated"], terminated)
                self.assertEqual(result["truncated"], truncated)
                self.assertEqual(result["fall_detected"], fall)
                self.assertEqual(result["evaluation_cutoff"], cutoff)
                self.assertEqual(result["fall_time_s"], .06 if fall else None)
                self.assertEqual(result["active_end_terms"], terms)

    def test_inconsistent_fall_term_and_post_done_steps_are_rejected(self):
        trace = self.trace()
        trace["terminated"].zero_()
        with self.assertRaises(ValueError):
            aggregate_episode(trace, step_dt=.02, end_terms=["fell_over"])
        for flag in ("terminated", "truncated"):
            with self.subTest(flag=flag):
                trace = self.trace()
                trace[flag][0] = True
                with self.assertRaises(ValueError):
                    aggregate_episode(trace, step_dt=.02, end_terms=["fell_over"])

    def test_invalid_timesteps_shapes_values_flags_and_empty_episodes_are_rejected(self):
        for step_dt in (0, -1, float("nan"), float("inf")):
            with self.subTest(step_dt=step_dt):
                with self.assertRaises(ValueError):
                    aggregate_episode(self.trace(), step_dt=step_dt, end_terms=["fell_over"])
        for name in self.trace():
            with self.subTest(field=name, issue="length"):
                trace = self.trace()
                trace[name] = trace[name][:-1]
                with self.assertRaises(ValueError):
                    aggregate_episode(trace, step_dt=.02, end_terms=["fell_over"])
            if self.trace()[name].dtype == torch.bool:
                with self.subTest(field=name, issue="flag dtype"):
                    trace = self.trace()
                    trace[name] = trace[name].float()
                    with self.assertRaises(ValueError):
                        aggregate_episode(trace, step_dt=.02, end_terms=["fell_over"])
            else:
                with self.subTest(field=name, issue="nonfinite"):
                    trace = self.trace()
                    trace[name].flatten()[0] = float("inf")
                    with self.assertRaises(ValueError):
                        aggregate_episode(trace, step_dt=.02, end_terms=["fell_over"])
        empty = {name: value[:0] for name, value in self.trace().items()}
        with self.assertRaises(ValueError):
            aggregate_episode(empty, step_dt=.02, end_terms=[])


if __name__ == "__main__":
    unittest.main()
