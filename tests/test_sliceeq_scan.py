import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_scan import (  # noqa: E402
    build_scan_protocol_table, scan_coherence_diagnostics,
    scan_coherent_slice_profiles,
)


class SliceEqScanUtilitiesTest(unittest.TestCase):
    def test_stratified_protocol_is_reproducible_and_order_invariant(self):
        cases = ['Case03', 'Case01', 'Case00', 'Case02']
        first = build_scan_protocol_table(
            cases, (0.45, 0.85), (-0.25, 0.25),
            refresh_id=4, base_seed=1337)
        second = build_scan_protocol_table(
            list(reversed(cases)), (0.45, 0.85), (-0.25, 0.25),
            refresh_id=4, base_seed=1337)
        self.assertEqual(first, second)

    def test_randomized_strata_cover_full_continuous_marginals(self):
        cases = ['Case{:02d}'.format(index) for index in range(7)]
        table = build_scan_protocol_table(
            cases, (0.45, 0.85), (-0.25, 0.25),
            refresh_id=0, base_seed=1337)
        sigma_unit = sorted(
            (table[name][0] - 0.45) / 0.40 for name in cases)
        phase_unit = sorted(
            (table[name][1] + 0.25) / 0.50 for name in cases)
        for index, value in enumerate(sigma_unit):
            self.assertGreaterEqual(value, index / 7.0)
            self.assertLess(value, (index + 1) / 7.0)
        for index, value in enumerate(phase_unit):
            self.assertGreaterEqual(value, index / 7.0)
            self.assertLess(value, (index + 1) / 7.0)
        # Jitter keeps the schedule continuous rather than reducing it to
        # deterministic bin centers or the old four SAQ nodes.
        self.assertFalse(all(
            abs(value * 7.0 - (int(value * 7.0) + 0.5)) < 1e-12
            for value in sigma_unit))

    def test_refresh_changes_protocol_but_preserves_cases(self):
        cases = ['Case00', 'Case01', 'Case02']
        first = build_scan_protocol_table(
            cases, (0.45, 0.85), (-0.25, 0.25), 0, 101)
        second = build_scan_protocol_table(
            cases, (0.45, 0.85), (-0.25, 0.25), 1, 101)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)

    def test_repeated_case_receives_identical_profile(self):
        table = build_scan_protocol_table(
            ['Case00', 'Case01'], (0.45, 0.85), (-0.25, 0.25),
            refresh_id=0, base_seed=1337)
        batch_cases = ['Case00', 'Case01', 'Case00', 'Case00']
        weights, sigma, phase = scan_coherent_slice_profiles(
            batch_cases, table, (-1, 0, 1), device=torch.device('cpu'))
        self.assertTrue(torch.equal(sigma[[0, 2, 3]], sigma[[0, 0, 0]]))
        self.assertTrue(torch.equal(phase[[0, 2, 3]], phase[[0, 0, 0]]))
        self.assertTrue(torch.equal(weights[0], weights[2]))
        self.assertTrue(torch.equal(weights[0], weights[3]))
        self.assertTrue(torch.allclose(
            weights.sum(dim=1), torch.ones(len(batch_cases))))
        diagnostics = scan_coherence_diagnostics(
            batch_cases, sigma, phase)
        self.assertEqual(
            float(diagnostics['within_case_sigma_range_max']), 0.0)
        self.assertEqual(
            float(diagnostics['within_case_phase_range_max']), 0.0)
        self.assertAlmostEqual(
            float(diagnostics['unique_case_fraction']), 0.5)

    def test_invalid_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            build_scan_protocol_table(
                [], (0.45, 0.85), (-0.25, 0.25), 0, 1)
        with self.assertRaises(ValueError):
            build_scan_protocol_table(
                ['Case00', 'Case00'], (0.45, 0.85), (-0.25, 0.25), 0, 1)
        with self.assertRaises(ValueError):
            build_scan_protocol_table(
                ['Case00'], (0.0, 0.85), (-0.25, 0.25), 0, 1)
        with self.assertRaises(ValueError):
            build_scan_protocol_table(
                ['Case00'], (0.45, 0.85), (0.25, -0.25), 0, 1)
        with self.assertRaises(KeyError):
            scan_coherent_slice_profiles(
                ['Missing'], {'Case00': (0.65, 0.0)}, (-1, 0, 1),
                device=torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()

