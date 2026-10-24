"""Numerical, freeze/resume and independent actor/critic regression checks."""

import io
import unittest

import torch

from training.normalization import (ObservationNormalizer, RunningMeanStd,
                                    observation_diagnostics, update_between_iterations)


def snapshot(module):
    return {name: value.clone() for name, value in module.state_dict().items()}


class NormalizationTests(unittest.TestCase):
    def test_boundary_update_counts_raw_samples_once_and_refreezes(self):
        norm = ObservationNormalizer({"actor": 2, "critic": 3}).eval()
        first = {"actor": torch.arange(16).float().reshape(4, 2, 2),
                 "critic": torch.arange(24).float().reshape(4, 2, 3)}
        report = update_between_iterations(norm, first)
        self.assertFalse(norm.training)
        self.assertTrue(all(not s.training for s in norm.stats.values()))
        for name in first:
            self.assertEqual(report[name]["samples_added"], 8)
            self.assertEqual(report[name]["count_after"], 8)
            self.assertGreater(report[name]["normalized_input_shift_rms"], 0)
        before = snapshot(norm)
        diagnostics = observation_diagnostics(norm, first)
        self.assert_state_equal(norm, before)
        self.assertEqual(diagnostics["actor"]["count"], 8)
        update_between_iterations(norm, {name: x + 10 for name, x in first.items()})
        for name, data in first.items():
            reference = torch.cat((data, data + 10)).double().reshape(-1, data.shape[-1])
            self.assertEqual(norm.stats[name].count, 16)
            torch.testing.assert_close(norm.stats[name].mean, reference.mean(0))
            torch.testing.assert_close(norm.stats[name].var, reference.var(0, correction=0))
        before = snapshot(norm)
        with self.assertRaises(ValueError):
            update_between_iterations(norm, {"actor": first["actor"], "critic": first["critic"] * float("nan")})
        self.assert_state_equal(norm, before)
        self.assertFalse(norm.training)

    def assert_state_equal(self, module, expected):
        for name, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)

    def test_chunked_population_statistics_and_feature_axes(self):
        obs = {"actor": torch.arange(24, dtype=torch.float64).reshape(4, 3, 2),
               "critic": torch.arange(36, dtype=torch.float64).reshape(4, 3, 3) * 7 - 100}
        norm = ObservationNormalizer({"actor": 2, "critic": 3})
        for start, stop in ((0, 1), (1, 3), (3, 4)):
            norm.update({name: x[start:stop] for name, x in obs.items()})
        before = snapshot(norm)
        for name, output in norm(obs).items():
            stats = norm.stats[name]
            samples = obs[name].reshape(-1, stats.dim)
            self.assertEqual(stats.count.item(), 12)
            torch.testing.assert_close(stats.mean, samples.mean(0))
            torch.testing.assert_close(stats.var, samples.var(0, correction=0))
            expected = (obs[name] - samples.mean(0)) / (samples.std(0, correction=0) + stats.eps)
            torch.testing.assert_close(output, expected)
            self.assertEqual(output.shape, obs[name].shape)
        self.assert_state_equal(norm, before)  # forward() never updates, even in train.
        critic_before = snapshot(norm.stats["critic"])
        norm.stats["actor"].update(torch.ones(2))
        self.assertEqual(norm.stats["actor"].count.item(), 13)
        self.assert_state_equal(norm.stats["critic"], critic_before)

    def test_freeze_serialization_and_continued_update(self):
        obs = {"actor": torch.tensor([[1., 3.], [5., 9.]]),
               "critic": torch.tensor([[100., -7., 2.], [200., 1., -4.]])}
        original = {name: x.clone() for name, x in obs.items()}
        norm = ObservationNormalizer({"actor": 2, "critic": 3}, eps=0.03)
        norm.update(obs)
        norm.eval()
        frozen = snapshot(norm)
        norm.update({name: x + 99 for name, x in obs.items()})
        outputs = norm(obs)
        self.assert_state_equal(norm, frozen)
        checkpoint = norm.checkpoint()
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        restored = ObservationNormalizer.from_checkpoint(torch.load(stream, weights_only=True))
        self.assertFalse(restored.training)
        self.assertTrue(all(not stats.training for stats in restored.stats.values()))
        for name, output in restored(obs).items():
            self.assertTrue(torch.equal(output, outputs[name]))
            self.assertEqual(restored.stats[name].eps, 0.03)
            self.assertEqual(output.dtype, obs[name].dtype)
            self.assertEqual(output.device, obs[name].device)
            self.assertTrue(torch.equal(obs[name], original[name]))
        following = {name: x[:1] + 7 for name, x in obs.items()}
        for module in (norm, restored):
            module.train()
            module.update(following)
        self.assert_state_equal(restored, snapshot(norm))
        for name, stats in restored.stats.items():
            self.assertEqual(stats.count.item(), 3)
            samples = torch.cat((obs[name], following[name])).double()
            torch.testing.assert_close(stats.mean, samples.mean(0))
            torch.testing.assert_close(stats.var, samples.var(0, correction=0))
        # The exported checkpoint is a snapshot, not a live view of the module.
        self.assertTrue(torch.equal(checkpoint["state_dict"]["stats.actor.count"], torch.tensor(2)))

    def test_single_constant_near_constant_and_input_gradient(self):
        for data in (torch.tensor([3., -2.]), torch.full((5, 2), 7.),
                     torch.tensor([[1., 1.], [1. + 1e-12, 1. - 1e-12]], dtype=torch.float64)):
            stats = RunningMeanStd(2)
            before_update = stats(data)
            torch.testing.assert_close(before_update, data / 1.01)
            self.assertEqual(stats.count.item(), 0)
            stats.update(data)
            output = stats(data)
            self.assertTrue(torch.isfinite(output).all().item())
            torch.testing.assert_close(stats.var, data.double().reshape(-1, 2).var(0, correction=0))
            self.assertFalse(torch.is_inference(output))
        x = torch.tensor([[1., 2.]], requires_grad=True)
        stats(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_invalid_inputs_do_not_modify_either_group(self):
        norm = ObservationNormalizer({"actor": 2, "critic": 3})
        valid = {"actor": torch.ones(2, 2), "critic": torch.ones(2, 3)}
        norm.update(valid)
        frozen = snapshot(norm)
        invalid = [torch.ones(2, 4), torch.empty(0, 3), torch.ones(2, 3, dtype=torch.long),
                   torch.full((2, 3), float("nan")), torch.full((2, 3), float("inf")),
                   torch.tensor([[1e308, 1e308, 1e308], [-1e308, -1e308, -1e308]],
                                dtype=torch.float64)]
        for critic in invalid:
            with self.assertRaises((ValueError, TypeError)):
                norm.update({"actor": valid["actor"] + 8, "critic": critic})
            self.assert_state_equal(norm, frozen)
        with self.assertRaisesRegex(ValueError, "groups"):
            norm.update({"actor": valid["actor"]})
        with self.assertRaisesRegex(ValueError, "sample axes"):
            norm.update({"actor": torch.ones(1, 2), "critic": valid["critic"]})
        self.assert_state_equal(norm, frozen)
        for eps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RunningMeanStd(2, eps)


if __name__ == "__main__":
    unittest.main()
