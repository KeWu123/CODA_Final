import math
import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_saq import (  # noqa: E402
    gauss_legendre_profile_nodes,
    quadrature_assignment_diagnostics,
    sample_stratified_slice_profiles,
)


class SliceEqSAQUtilitiesTest(unittest.TestCase):
    def test_nodes_match_two_dimensional_gauss_legendre_rule(self):
        nodes = gauss_legendre_profile_nodes(
            (0.45, 0.85), (-0.25, 0.25))
        delta_sigma = 0.20 / math.sqrt(3.0)
        delta_phase = 0.25 / math.sqrt(3.0)
        expected = (
            (0.65 - delta_sigma, -delta_phase),
            (0.65 - delta_sigma, delta_phase),
            (0.65 + delta_sigma, -delta_phase),
            (0.65 + delta_sigma, delta_phase),
        )
        for observed, target in zip(nodes, expected):
            self.assertAlmostEqual(observed[0], target[0], places=7)
            self.assertAlmostEqual(observed[1], target[1], places=7)

    def test_every_node_occurs_three_times_in_twelve_samples(self):
        generator = torch.Generator(device='cpu').manual_seed(1337)
        weights, sigma, phase, node_ids = \
            sample_stratified_slice_profiles(
                12, (-1, 0, 1), (0.45, 0.85), (-0.25, 0.25),
                device=torch.device('cpu'), generator=generator)
        self.assertEqual(tuple(weights.shape), (12, 3))
        self.assertEqual(tuple(sigma.shape), (12,))
        self.assertEqual(tuple(phase.shape), (12,))
        self.assertTrue(torch.allclose(
            weights.sum(dim=1), torch.ones(12)))
        self.assertEqual(torch.bincount(node_ids).tolist(), [3, 3, 3, 3])
        metadata = quadrature_assignment_diagnostics(node_ids)
        self.assertEqual(float(metadata['quadrature_node_coverage']), 1.0)
        self.assertEqual(
            float(metadata['quadrature_max_count_deviation']), 0.0)

    def test_seeded_permutation_is_reproducible(self):
        def draw(seed):
            generator = torch.Generator(device='cpu').manual_seed(seed)
            return sample_stratified_slice_profiles(
                12, (-1, 0, 1), (0.45, 0.85), (-0.25, 0.25),
                device=torch.device('cpu'), generator=generator)

        first = draw(1337)
        second = draw(1337)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

    def test_unbalanced_branch_size_fails_loudly(self):
        with self.assertRaises(ValueError):
            sample_stratified_slice_profiles(
                10, (-1, 0, 1), (0.45, 0.85), (-0.25, 0.25),
                device=torch.device('cpu'))

    def test_invalid_profile_ranges_fail_loudly(self):
        with self.assertRaises(ValueError):
            gauss_legendre_profile_nodes((0.0, 0.85), (-0.25, 0.25))
        with self.assertRaises(ValueError):
            gauss_legendre_profile_nodes((0.45, 0.85), (0.25, -0.25))


if __name__ == '__main__':
    unittest.main()
