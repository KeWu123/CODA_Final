import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from utils.sliceeq_arcp import (  # noqa: E402
    PARENT_CENTER_WEIGHT_MAX, PARENT_CENTER_WEIGHT_MIN,
    calibrate_profile_weights, calibrate_profile_weights_numpy,
    response_gram)


class SliceEqARCPTest(unittest.TestCase):
    def _stack(self):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 9),
            torch.linspace(-1.0, 1.0, 11), indexing='ij')
        center = xx.square() + 0.4 * yy + 0.2 * xx * yy
        previous = center - 0.3 * xx + 0.08 * yy.square()
        following = center + 0.2 * xx + 0.15 * xx.square()
        return torch.stack((previous, center, following), dim=0).view(
            1, 3, 1, 9, 11)

    def test_exact_residual_decomposition(self):
        stack = self._stack()
        weights = torch.tensor([[0.17, 0.61, 0.22]])
        residual = (stack * weights.view(1, 3, 1, 1, 1)).sum(1) - stack[:, 1]
        first = 0.5 * (stack[:, 2] - stack[:, 0])
        second = stack[:, 0] - 2.0 * stack[:, 1] + stack[:, 2]
        expected = (weights[:, 2] - weights[:, 0]).view(1, 1, 1, 1) * first
        expected = expected + 0.5 * (
            weights[:, 0] + weights[:, 2]).view(1, 1, 1, 1) * second
        self.assertTrue(torch.allclose(residual, expected, atol=1e-7, rtol=1e-7))

    def test_reference_identity_reproduces_parent(self):
        stack = self._stack()
        matrix, valid = response_gram(stack)
        self.assertTrue(valid.item())
        weights = torch.tensor([[0.20, 0.60, 0.20]])
        calibrated, metadata = calibrate_profile_weights(
            stack, weights, matrix[0])
        self.assertTrue(torch.allclose(calibrated, weights, atol=1e-6))
        self.assertAlmostEqual(metadata['arcp_alpha_mean'].item(), 1.0, places=6)

    def test_duplicate_support_is_parent_identity(self):
        stack = self._stack().repeat(2, 1, 1, 1, 1)
        stack[1, 0] = stack[1, 1]
        weights = torch.tensor([
            [0.20, 0.60, 0.20], [0.10, 0.70, 0.20]])
        reference = torch.tensor([[2.0, 0.1], [0.1, 1.0]])
        calibrated, metadata = calibrate_profile_weights(
            stack, weights, reference)
        self.assertTrue(torch.equal(calibrated[1], weights[1]))
        self.assertAlmostEqual(
            metadata['arcp_duplicate_support_fraction'].item(), 0.5)

    def test_convexity_parent_bounds_and_rng_identity(self):
        stack = self._stack().repeat(3, 1, 1, 1, 1)
        stack[1] *= 0.02
        stack[2] *= 20.0
        weights = torch.tensor([
            [0.20, 0.60, 0.20],
            [0.03, 0.85, 0.12],
            [0.25, 0.50, 0.25],
        ])
        reference = torch.tensor([[1.4, 0.2], [0.2, 0.9]])
        rng_before = torch.random.get_rng_state().clone()
        calibrated, _ = calibrate_profile_weights(
            stack, weights, reference)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue((calibrated >= 0.0).all())
        self.assertTrue(torch.allclose(
            calibrated.sum(1), torch.ones(3), atol=1e-6, rtol=1e-6))
        self.assertTrue((calibrated[:, 1] >= PARENT_CENTER_WEIGHT_MIN).all())
        self.assertTrue((calibrated[:, 1] <= PARENT_CENTER_WEIGHT_MAX).all())

    def test_numpy_and_torch_calibration_match(self):
        stack = self._stack()
        weights = torch.tensor([[0.18, 0.62, 0.20]])
        reference = torch.tensor([[1.1, -0.1], [-0.1, 0.7]])
        torch_weights, torch_metadata = calibrate_profile_weights(
            stack, weights, reference)
        numpy_weights, numpy_metadata = calibrate_profile_weights_numpy(
            stack[0, :, 0].numpy(), weights[0].numpy(), reference.numpy())
        self.assertTrue(np.allclose(
            torch_weights[0].numpy(), numpy_weights, atol=1e-6, rtol=1e-6))
        self.assertAlmostEqual(
            torch_metadata['arcp_alpha_mean'].item(),
            numpy_metadata['alpha'], places=6)


if __name__ == '__main__':
    unittest.main()

