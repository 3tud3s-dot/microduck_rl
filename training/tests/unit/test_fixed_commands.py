"""Instance-local fixed-command hooks preserve samplers and their RNG stream."""

from types import SimpleNamespace
import unittest

import torch

from training.diagnose import install_fixed_commands


class FakeCommandTerm:
    def __init__(self, width, *, twist=False, init_velocity_prob=0.):
        self.command = torch.full((1, width), -9.)
        self.cfg = SimpleNamespace(init_velocity_prob=init_velocity_prob)
        self.resample_calls = []
        self.update_calls = 0
        self.random_draws = []
        if twist:
            self.vel_command_w = torch.full_like(self.command, -8.)

    def _resample_command(self, ids):
        self.resample_calls.append(ids.clone())
        draw = torch.rand((len(ids), self.command.shape[-1]))
        self.random_draws.append(draw.clone())
        self.command[ids] = draw
        if hasattr(self, "vel_command_w"):
            self.vel_command_w.copy_(self.command + 4.)

    def _update_command(self):
        self.update_calls += 1
        draw = torch.rand_like(self.command)
        self.random_draws.append(draw.clone())
        self.command.copy_(draw + 2.)
        if hasattr(self, "vel_command_w"):
            self.vel_command_w.copy_(self.command + 5.)


def fake_env(*, init_velocity_prob=0.):
    terms = {
        "twist": FakeCommandTerm(3, twist=True, init_velocity_prob=init_velocity_prob),
        "head_pose": FakeCommandTerm(4),
        "body_pose": FakeCommandTerm(6),
    }
    manager = SimpleNamespace(get_term=terms.__getitem__, get_command=lambda name: terms[name].command)
    return SimpleNamespace(device="cpu", command_manager=manager, terms=terms)


class FixedCommandTests(unittest.TestCase):
    def assert_targets(self, env, twist):
        for name, expected in (("twist", torch.tensor([twist])),
                               ("head_pose", torch.zeros(1, 4)),
                               ("body_pose", torch.zeros(1, 6))):
            torch.testing.assert_close(env.terms[name].command, expected, rtol=0, atol=0)
        torch.testing.assert_close(env.terms["twist"].vel_command_w,
                                   torch.tensor([twist]), rtol=0, atol=0)

    def test_resample_and_update_repeatedly_pin_distinct_closure_targets(self):
        env = fake_env()
        twist = (.2, -.1, .5)
        targets = install_fixed_commands(env, twist)
        self.assertEqual(set(targets), {"twist", "head_pose", "body_pose"})
        ids = torch.tensor([0])
        for _ in range(4):
            for term in env.terms.values():
                term.command.fill_(99.)
                term._resample_command(ids)
            self.assert_targets(env, twist)
            for term in env.terms.values():
                term.command.fill_(88.)
                term._update_command()
            self.assert_targets(env, twist)
        for name, term in env.terms.items():
            self.assertEqual(len(term.resample_calls), 4)
            self.assertEqual(term.update_calls, 4)
            self.assertEqual(len(term.random_draws), 8)
            for ids_seen in term.resample_calls:
                torch.testing.assert_close(ids_seen, ids, rtol=0, atol=0)
            torch.testing.assert_close(targets[name], term.command, rtol=0, atol=0)

    def test_original_methods_and_rng_draws_match_same_seed_unwrapped_terms(self):
        def run(env):
            for _ in range(3):
                for term in env.terms.values():
                    term._resample_command(torch.tensor([0]))
                for term in env.terms.values():
                    term._update_command()
            return torch.get_rng_state().clone(), torch.rand(7)

        baseline, wrapped = fake_env(), fake_env()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731)
            before_install = torch.get_rng_state().clone()
            install_fixed_commands(wrapped, (.2, 0., -.5))
            torch.testing.assert_close(torch.get_rng_state(), before_install, rtol=0, atol=0)
            wrapped_state, wrapped_next = run(wrapped)
            torch.manual_seed(731)
            baseline_state, baseline_next = run(baseline)
        torch.testing.assert_close(wrapped_state, baseline_state, rtol=0, atol=0)
        torch.testing.assert_close(wrapped_next, baseline_next, rtol=0, atol=0)
        for name in baseline.terms:
            original, hooked = baseline.terms[name], wrapped.terms[name]
            self.assertEqual(original.update_calls, hooked.update_calls)
            self.assertEqual(len(original.resample_calls), len(hooked.resample_calls))
            for original_draw, hooked_draw in zip(original.random_draws, hooked.random_draws):
                torch.testing.assert_close(original_draw, hooked_draw, rtol=0, atol=0)
        self.assert_targets(wrapped, (.2, 0., -.5))

    def test_hooks_only_modify_the_selected_environment_instances(self):
        env, other = fake_env(), fake_env()
        original_resample = FakeCommandTerm._resample_command
        original_update = FakeCommandTerm._update_command
        install_fixed_commands(env, (0., 0., .5))
        for name, term in other.terms.items():
            self.assertNotIn("_resample_command", vars(term))
            self.assertNotIn("_update_command", vars(term))
            self.assertIn("_resample_command", vars(env.terms[name]))
            self.assertIn("_update_command", vars(env.terms[name]))
            term._resample_command(torch.tensor([0]))
            self.assertTrue(torch.all((term.command >= 0.) & (term.command < 1.)).item())
            term._update_command()
            self.assertTrue(torch.all((term.command >= 2.) & (term.command < 3.)).item())
        self.assertIs(FakeCommandTerm._resample_command, original_resample)
        self.assertIs(FakeCommandTerm._update_command, original_update)
        for term in env.terms.values():
            term._resample_command(torch.tensor([0]))
        self.assert_targets(env, (0., 0., .5))

    def test_initial_velocity_sampling_is_rejected_before_any_mutation(self):
        env = fake_env(init_velocity_prob=.1)
        snapshots = {name: term.command.clone() for name, term in env.terms.items()}
        velocity_before = env.terms["twist"].vel_command_w.clone()
        rng_before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "init_velocity_prob=0"):
            install_fixed_commands(env, (.2, 0., .5))
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        for name, term in env.terms.items():
            torch.testing.assert_close(term.command, snapshots[name], rtol=0, atol=0)
            self.assertNotIn("_resample_command", vars(term))
            self.assertNotIn("_update_command", vars(term))
            self.assertEqual(term.resample_calls, [])
            self.assertEqual(term.update_calls, 0)
        torch.testing.assert_close(env.terms["twist"].vel_command_w, velocity_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
