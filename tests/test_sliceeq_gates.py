import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_gates import (  # noqa: E402
    PearsonAccumulator,
    acquisition_residual,
    normalized_weighted_soft_cross_entropy,
    per_sample_gradient_cosine,
)


class SliceEqGateUtilitiesTest(unittest.TestCase):
    def test_acquisition_residual_is_total_variation_from_center(self):
        occupancy = torch.tensor(
            [[[[1.0, 0.7]], [[0.0, 0.3]]]])
        center = torch.tensor([[[0, 0]]])
        residual = acquisition_residual(occupancy, center)
        self.assertTrue(torch.allclose(
            residual, torch.tensor([[[0.0, 0.3]]])))
        self.assertFalse(residual.requires_grad)

    def test_weighted_ce_is_normalized_per_sample_and_zero_when_inactive(self):
        logits = torch.zeros(2, 2, 1, 2, requires_grad=True)
        occupancy = torch.tensor([
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[1.0, 1.0]], [[0.0, 0.0]]],
        ])
        weights = torch.tensor([[[1.0, 3.0]], [[0.0, 0.0]]])
        loss = normalized_weighted_soft_cross_entropy(
            logits, occupancy, weights)
        self.assertTrue(torch.allclose(
            loss, 0.5 * torch.log(torch.tensor(2.0))))
        loss.backward()
        self.assertTrue(torch.allclose(
            logits.grad[1], torch.zeros_like(logits.grad[1])))

    def test_fractional_and_binary_measures_can_have_different_gradients(self):
        logits = torch.tensor(
            [[[[1.0, -0.5]], [[-0.25, 0.75]]]], requires_grad=True)
        occupancy = torch.tensor(
            [[[[0.8, 0.3]], [[0.2, 0.7]]]])
        fractional = torch.tensor([[[0.1, 0.9]]])
        binary = torch.ones_like(fractional)
        fractional_loss = normalized_weighted_soft_cross_entropy(
            logits, occupancy, fractional)
        binary_loss = normalized_weighted_soft_cross_entropy(
            logits, occupancy, binary)
        first = torch.autograd.grad(
            fractional_loss, logits, retain_graph=True)[0]
        second = torch.autograd.grad(binary_loss, logits)[0]
        cosine, distance = per_sample_gradient_cosine(first, second)
        self.assertEqual(len(cosine), 1)
        self.assertLess(cosine[0], 0.98)
        self.assertGreater(distance[0], 0.2)

    def test_pearson_accumulator_matches_perfect_linear_relation(self):
        accumulator = PearsonAccumulator()
        accumulator.update(
            torch.tensor([1.0, 2.0]), torch.tensor([3.0, 5.0]))
        accumulator.update(
            torch.tensor([3.0, 4.0]), torch.tensor([7.0, 9.0]))
        self.assertAlmostEqual(accumulator.correlation(), 1.0)


if __name__ == '__main__':
    unittest.main()
