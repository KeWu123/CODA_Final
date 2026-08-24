import unittest
import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_scpo import (  # noqa: E402
    scpo_diagnostics, slab_largest_connected_component)


class SliceEqSCPOTest(unittest.TestCase):
    def test_volume_component_beats_larger_single_slice_distractor(self):
        stack = torch.zeros(1, 3, 10, 10)
        stack[:, :, 1:3, 1:3] = 1
        stack[:, 1, 5:8, 5:8] = 1
        coherent, counts = slab_largest_connected_component(stack)
        self.assertEqual(counts.item(), 2)
        self.assertEqual(coherent.sum().item(), 12)
        self.assertTrue(torch.all(coherent[:, :, 1:3, 1:3] == 1))
        self.assertEqual(coherent[:, 1, 5:8, 5:8].sum().item(), 0)

    def test_26_connectivity_links_diagonal_axial_voxels(self):
        stack = torch.zeros(1, 3, 5, 5)
        stack[0, 0, 1, 1] = 1
        stack[0, 1, 2, 2] = 1
        stack[0, 2, 3, 3] = 1
        coherent, counts = slab_largest_connected_component(stack)
        self.assertEqual(counts.item(), 1)
        self.assertTrue(torch.equal(coherent, stack))

    def test_empty_slabs_stay_empty(self):
        stack = torch.zeros(2, 3, 6, 6)
        coherent, counts = slab_largest_connected_component(stack)
        self.assertTrue(torch.equal(coherent, stack))
        self.assertTrue(torch.equal(counts, torch.zeros_like(counts)))

    def test_diagnostics_separate_removed_and_added_mass(self):
        raw = torch.zeros(1, 3, 4, 4)
        parent = raw.clone()
        coherent = raw.clone()
        parent[0, 0, 0, 0] = 1
        coherent[0, 1, 1, 1] = 1
        diagnostics = scpo_diagnostics(
            raw, parent, coherent, torch.tensor([2.0]))
        self.assertAlmostEqual(
            diagnostics['changed_pixel_fraction'].item(), 2 / 48)
        self.assertAlmostEqual(
            diagnostics['removed_parent_foreground_fraction'].item(), 1 / 48)
        self.assertAlmostEqual(
            diagnostics['added_parent_foreground_fraction'].item(), 1 / 48)
        self.assertEqual(diagnostics['active_slab_fraction'].item(), 1.0)

    def test_shape_and_binary_contracts_are_strict(self):
        with self.assertRaises(ValueError):
            slab_largest_connected_component(torch.zeros(1, 5, 4, 4))
        invalid = torch.zeros(1, 3, 4, 4)
        invalid[0, 0, 0, 0] = 0.5
        with self.assertRaises(ValueError):
            slab_largest_connected_component(invalid)


if __name__ == '__main__':
    unittest.main()
