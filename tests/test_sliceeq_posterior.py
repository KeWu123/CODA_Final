import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_posterior import (  # noqa: E402
    distribution_residual,
    occupancy_brier_map,
    profile_weighted_distribution,
    topology_gate_binary_posterior,
)


class SliceEqPosteriorUtilitiesTest(unittest.TestCase):
    def test_profile_weighted_distribution_preserves_simplex(self):
        stack = torch.tensor([[
            [[[0.9]], [[0.1]]],
            [[[0.4]], [[0.6]]],
            [[[0.2]], [[0.8]]],
        ]])
        weights = torch.tensor([[0.2, 0.5, 0.3]])
        output = profile_weighted_distribution(stack, weights)
        expected = torch.tensor([[[[0.44]], [[0.56]]]])
        self.assertTrue(torch.allclose(output, expected))
        self.assertTrue(torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0])))

    def test_topology_gate_removes_foreground_outside_lcc_only(self):
        posterior = torch.tensor([[[
            [[0.1, 0.7]],
            [[0.9, 0.3]],
        ]]])
        lcc = torch.tensor([[[[1, 0]]]])
        gated = topology_gate_binary_posterior(posterior, lcc)
        expected = torch.tensor([[[
            [[0.1, 1.0]],
            [[0.9, 0.0]],
        ]]])
        self.assertTrue(torch.allclose(gated, expected))

    def test_distribution_residual_uses_matching_center_distribution(self):
        reacquired = torch.tensor([[[[0.6]], [[0.4]]]])
        center = torch.tensor([[[[0.8]], [[0.2]]]])
        residual = distribution_residual(reacquired, center)
        self.assertTrue(torch.allclose(
            residual, torch.tensor([[[0.2]]])))

    def test_brier_map_is_class_mean_squared_error(self):
        candidate = torch.tensor([[[[0.7]], [[0.3]]]])
        exact = torch.tensor([[[[1.0]], [[0.0]]]])
        brier = occupancy_brier_map(candidate, exact)
        self.assertTrue(torch.allclose(
            brier, torch.tensor([[[0.09]]])))

    def test_invalid_profile_weights_fail_loudly(self):
        stack = torch.tensor([[
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
        ]])
        with self.assertRaises(ValueError):
            profile_weighted_distribution(
                stack, torch.tensor([[0.2, 0.2]]))


if __name__ == '__main__':
    unittest.main()
