import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from utils.sliceeq_mpd import (  # noqa: E402
    FrozenProfileSampler, design_robust_distribution,
    midpoint_profile_grid, normalized_axial_gram,
    normalized_profile_residuals, occupancy_metrics_from_patterns,
    pattern_counts, profile_moments)


class SliceEqMPDTest(unittest.TestCase):
    def test_grid_is_convex_symmetric_and_has_locked_support(self):
        sigmas, phases, weights, parent = midpoint_profile_grid()
        self.assertEqual(weights.shape, (441, 3))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.allclose(weights.sum(1), 1.0, atol=1e-12))
        self.assertTrue(np.allclose(
            weights.reshape(21, 21, 3)[:, :, 0],
            weights.reshape(21, 21, 3)[:, ::-1, 2], atol=1e-12))
        self.assertAlmostEqual(parent.sum(), 1.0, places=12)
        self.assertGreater(sigmas.min(), 0.45)
        self.assertLess(sigmas.max(), 0.85)
        self.assertGreater(phases.min(), -0.25)
        self.assertLess(phases.max(), 0.25)

    def test_pattern_utility_retains_only_semantic_identity(self):
        previous = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
        center = np.asarray([[0, 1], [0, 1]], dtype=np.uint8)
        following = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        weights = np.asarray([
            [0.2, 0.6, 0.2],
            [0.8, 0.10, 0.10],
        ], dtype=np.float64)
        metrics = occupancy_metrics_from_patterns(
            pattern_counts(previous, center, following), weights)
        self.assertEqual(metrics['opportunity_pixels'], 4)
        # RFI is entropy retained only where the profile preserves the center
        # semantic class. Its magnitude need not rank profiles by hard-change
        # rate because the retained fractional pixels can have different
        # entropies; hard-change is checked independently below.
        self.assertTrue(np.isfinite(metrics['utility']).all())
        self.assertTrue(np.all(metrics['utility'] >= 0.0))
        self.assertLess(metrics['hard_change'][0], metrics['hard_change'][1])

    def test_gram_residual_matches_materialized_operator(self):
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, 9),
            np.linspace(-1.0, 1.0, 11), indexing='ij')
        center = xx ** 2 + 0.3 * yy
        stack = np.stack((
            center - 0.2 * xx + 0.1 * yy ** 2,
            center,
            center + 0.3 * xx + 0.05 * xx ** 2), axis=0)
        weights = np.asarray([
            [0.2, 0.6, 0.2], [0.1, 0.7, 0.2]], dtype=np.float64)
        predicted = normalized_profile_residuals(
            weights, normalized_axial_gram(stack))
        scale = np.sqrt(np.mean((center - center.mean()) ** 2))
        actual = []
        for profile in weights:
            mixed = np.sum(stack * profile[:, None, None], axis=0)
            actual.append(np.sqrt(np.mean((mixed - center) ** 2)) / scale)
        self.assertTrue(np.allclose(predicted, actual, atol=1e-10))

    def test_two_stage_design_respects_all_locked_constraints(self):
        sigmas, phases, weights, parent = midpoint_profile_grid()
        b, _, features = profile_moments(weights)
        symmetric_score = 1.0 - np.abs(phases) / 0.3
        utilities = np.stack((
            0.3 + 0.2 * symmetric_score + 0.05 * b,
            0.3 + 0.1 * symmetric_score + 0.08 * b,
            0.3 + 0.15 * symmetric_score + 0.04 * b,
        ), axis=0)
        residuals = np.stack((
            0.1 + features[:, 0],
            0.2 + features[:, 0] + 0.1 * features[:, 2],
            0.15 + 0.8 * features[:, 0],
        ), axis=0)
        result = design_robust_distribution(
            utilities, residuals, weights, sigmas, phases, parent)
        q = result['probabilities']
        self.assertTrue(result['diagnostics']['all_pass'])
        self.assertTrue(np.allclose(q.sum(), 1.0, atol=1e-8))
        self.assertTrue(np.allclose(
            q.reshape(21, 21), q.reshape(21, 21)[:, ::-1], atol=1e-10))
        self.assertLessEqual(np.max(q / parent), 3.0 + 1e-7)

    def test_utility_and_residual_strata_may_have_different_counts(self):
        sigmas, phases, weights, parent = midpoint_profile_grid()
        b, _, features = profile_moments(weights)
        utilities = np.stack((
            0.2 + 0.1 * b,
            0.25 + 0.05 * b,
        ), axis=0)
        # Empty-opportunity RFI strata are excluded from utilities, but their
        # image residual budgets remain represented by these four rows.
        residuals = np.stack((
            0.1 + b,
            0.2 + 0.8 * b,
            0.15 + features[:, 2],
            0.12 + b + 0.1 * features[:, 2],
        ), axis=0)
        result = design_robust_distribution(
            utilities, residuals, weights, sigmas, phases, parent)
        self.assertTrue(result['diagnostics']['all_pass'])
        self.assertEqual(
            len(result['diagnostics']['image_residuals']), 4)
        self.assertEqual(len(result['diagnostics']['utilities']), 2)

    def test_frozen_sampler_uses_only_private_generator(self):
        sigmas, phases, weights, parent = midpoint_profile_grid()
        sampler = FrozenProfileSampler({
            'sigmas': sigmas,
            'phases': phases,
            'weights': weights,
            'probabilities': parent,
            'distribution_sha256': 'unit-test',
        })
        global_before = torch.random.get_rng_state().clone()
        g1 = torch.Generator(device='cpu').manual_seed(1337)
        g2 = torch.Generator(device='cpu').manual_seed(1337)
        first = sampler(
            64, (-1, 0, 1), (0.45, 0.85), (-0.25, 0.25),
            torch.device('cpu'), g1)
        second = sampler(
            64, (-1, 0, 1), (0.45, 0.85), (-0.25, 0.25),
            torch.device('cpu'), g2)
        self.assertTrue(all(
            torch.equal(left, right) for left, right in zip(first, second)))
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
