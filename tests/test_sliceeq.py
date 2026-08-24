import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from dataloaders.sliceeq_dataset import build_neighbor_table  # noqa: E402
from utils.sliceeq import (  # noqa: E402
    paired_slice_reacquisition,
    reacquisition_diagnostics,
    sample_slice_profiles,
)


class SliceEqUtilitiesTest(unittest.TestCase):
    def test_neighbor_table_is_case_local_and_endpoint_clamped(self):
        names = [
            'Case00_slice_0', 'Case00_slice_1', 'Case00_slice_2',
            'Case01_slice_7', 'Case01_slice_8',
        ]
        table = build_neighbor_table(names, radius=1)
        self.assertEqual(
            table[0][0],
            ('Case00_slice_0', 'Case00_slice_0', 'Case00_slice_1'))
        self.assertTrue(table[0][1])
        self.assertEqual(
            table[1][0],
            ('Case00_slice_0', 'Case00_slice_1', 'Case00_slice_2'))
        self.assertFalse(table[1][1])
        self.assertEqual(
            table[3][0],
            ('Case01_slice_7', 'Case01_slice_7', 'Case01_slice_8'))

    def test_neighbor_table_rejects_internal_gaps_and_duplicates(self):
        with self.assertRaises(ValueError):
            build_neighbor_table(
                ['Case00_slice_0', 'Case00_slice_2'], radius=1)
        with self.assertRaises(ValueError):
            build_neighbor_table(
                ['Case00_slice_0', 'Case00_slice_0'], radius=1)

    def test_profile_sampling_is_normalized_and_deterministic(self):
        first = sample_slice_profiles(
            5, (-1, 0, 1), (0.45, 0.85), (-0.25, 0.25),
            device='cpu', generator=torch.Generator().manual_seed(7))
        second = sample_slice_profiles(
            5, (-1, 0, 1), (0.45, 0.85), (-0.25, 0.25),
            device='cpu', generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.all(first[0] >= 0.0))
        self.assertTrue(torch.allclose(
            first[0].sum(dim=1), torch.ones(5)))

    def test_paired_operator_uses_identical_profile(self):
        images = torch.tensor([0.0, 1.0, 3.0]).view(1, 3, 1, 1, 1)
        targets = torch.tensor([0, 1, 1]).view(1, 3, 1, 1)
        weights = torch.tensor([[0.2, 0.3, 0.5]])
        image, target, occupancy = paired_slice_reacquisition(
            images, targets, weights, num_classes=2)
        self.assertTrue(torch.allclose(image, torch.tensor([[[[1.8]]]])))
        self.assertEqual(int(target.item()), 1)
        self.assertTrue(torch.allclose(
            occupancy[:, :, 0, 0], torch.tensor([[0.2, 0.8]])))

    def test_identity_profile_recovers_center_exactly(self):
        images = torch.randn(4, 3, 1, 8, 8)
        targets = torch.randint(0, 2, (4, 3, 8, 8))
        weights = torch.tensor([[0.0, 1.0, 0.0]]).repeat(4, 1)
        image, target, _ = paired_slice_reacquisition(
            images, targets, weights, num_classes=2)
        self.assertTrue(torch.equal(image, images[:, 1]))
        self.assertTrue(torch.equal(target, targets[:, 1]))

    def test_diagnostics_are_finite_and_report_target_change(self):
        images = torch.tensor([0.0, 1.0, 3.0]).view(1, 3, 1, 1, 1)
        targets = torch.tensor([0, 0, 1]).view(1, 3, 1, 1)
        weights = torch.tensor([[0.0, 0.4, 0.6]])
        image, target, _ = paired_slice_reacquisition(
            images, targets, weights, num_classes=2)
        metadata = reacquisition_diagnostics(
            images, targets, image, target, weights,
            torch.tensor([0.7]), torch.tensor([0.2]))
        for value in metadata.values():
            self.assertTrue(torch.isfinite(value))
        self.assertEqual(float(metadata['target_changed_fraction']), 1.0)

    def test_invalid_weights_fail_loudly(self):
        images = torch.zeros(1, 3, 1, 4, 4)
        targets = torch.zeros(1, 3, 4, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            paired_slice_reacquisition(
                images, targets, torch.tensor([[0.2, 0.2, 0.2]]), 2)
        with self.assertRaises(ValueError):
            paired_slice_reacquisition(
                images, targets, torch.tensor([[-0.1, 0.5, 0.6]]), 2)


if __name__ == '__main__':
    unittest.main()

