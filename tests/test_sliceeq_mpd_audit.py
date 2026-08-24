import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402

from utils import sliceeq_mpd as mpd  # noqa: E402
from utils.sliceeq_mpd_audit import (  # noqa: E402
    aggregate_statistics, design_distribution, midpoint_profile_grid)


class SliceEqMPDAuditTest(unittest.TestCase):
    def test_variable_grid_is_convex_and_phase_symmetric(self):
        for side in (5, 11):
            sigmas, phases, weights, parent = midpoint_profile_grid(side)
            self.assertEqual(weights.shape, (side * side, 3))
            self.assertTrue(np.allclose(weights.sum(1), 1.0, atol=1e-12))
            self.assertTrue(np.allclose(
                weights.reshape(side, side, 3)[:, :, 0],
                weights.reshape(side, side, 3)[:, ::-1, 2], atol=1e-12))
            self.assertTrue(np.allclose(parent, 1.0 / (side * side)))
            self.assertEqual(sigmas.shape, phases.shape)

    def test_generalized_design_respects_audit_constraints(self):
        sigmas, phases, weights, parent = midpoint_profile_grid(5)
        neighbor_mass, _, features = mpd.profile_moments(weights)
        utilities = np.stack((
            0.30 + 0.20 * neighbor_mass,
            0.35 + 0.10 * neighbor_mass,
            0.32 + 0.08 * neighbor_mass + 0.02 * features[:, 2],
        ))
        residuals = np.stack((
            0.10 + neighbor_mass,
            0.20 + 0.80 * neighbor_mass,
        ))
        result = design_distribution(
            utilities, residuals, weights, sigmas, phases, parent)
        self.assertTrue(all(result['checks'].values()))
        q = result['probabilities']
        self.assertAlmostEqual(float(q.sum()), 1.0, places=10)
        self.assertTrue(np.allclose(
            q.reshape(5, 5), q.reshape(5, 5)[:, ::-1], atol=1e-10))
        self.assertLessEqual(np.max(q / parent), 3.0 + 1e-7)

    def test_dynamic_axial_binning_keeps_patient_boundaries(self):
        _, _, weights, parent = midpoint_profile_grid(5)
        profile_count = weights.shape[0]
        case_names = np.asarray(['CaseA'] * 6 + ['CaseB'] * 6)
        positions = np.asarray(list(range(6)) * 2)
        lengths = np.full(12, 6)
        base = np.arange(12, dtype=np.float64)[:, None]
        tiled = np.repeat(base, profile_count, axis=1)
        slice_statistics = {
            'weights': weights,
            'parent': parent,
            'utilities': tiled + 1.0,
            'residuals': tiled + 2.0,
            'hard_change': tiled + 3.0,
            'fractional_support': tiled + 4.0,
            'foreground_mass_error': tiled + 5.0,
            'active': np.ones(12, dtype=bool),
            'opportunity_pixels': np.ones(12, dtype=np.int64),
            'clamped': np.zeros(12, dtype=bool),
            'case_names': case_names,
            'case_positions': positions,
            'case_lengths': lengths,
            'case_order': ['CaseA', 'CaseB'],
        }
        aggregated = aggregate_statistics(slice_statistics, axial_bins=3)
        self.assertEqual(aggregated['utilities'].shape, (6, profile_count))
        self.assertEqual(aggregated['slice_count'].tolist(), [2] * 6)
        self.assertEqual(
            aggregated['strata'],
            [('CaseA', 0), ('CaseA', 1), ('CaseA', 2),
             ('CaseB', 0), ('CaseB', 1), ('CaseB', 2)])
        self.assertAlmostEqual(aggregated['utilities'][0, 0], 1.5)
        self.assertAlmostEqual(aggregated['utilities'][3, 0], 7.5)

    def test_locked_configuration_matches_original_solver(self):
        sigmas, phases, weights, parent = mpd.midpoint_profile_grid()
        neighbor_mass, _, features = mpd.profile_moments(weights)
        utilities = np.stack((
            0.30 + 0.20 * neighbor_mass,
            0.35 + 0.10 * neighbor_mass,
        ))
        residuals = np.stack((
            0.10 + neighbor_mass,
            0.20 + 0.80 * neighbor_mass + 0.01 * features[:, 2],
        ))
        original = mpd.design_robust_distribution(
            utilities, residuals, weights, sigmas, phases, parent)
        audited = design_distribution(
            utilities, residuals, weights, sigmas, phases, parent)
        self.assertLess(
            np.max(np.abs(
                original['probabilities'] - audited['probabilities'])),
            1e-6)
        self.assertLess(
            mpd.js_divergence(
                original['probabilities'], audited['probabilities']),
            1e-9)


if __name__ == '__main__':
    unittest.main()
