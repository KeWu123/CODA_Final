import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.bmer import (  # noqa: E402
    BoundaryProfileBank,
    build_position_bin_map,
    extract_profile_field,
    lookup_position_bins,
    resynthesize_labeled_images,
)


def make_mask(size=48):
    mask = torch.zeros(size, size, dtype=torch.long)
    mask[10:38, 12:36] = 1
    return mask


def make_step_image(mask, inside=0.8, outside=0.2):
    image = torch.full(mask.shape, outside, dtype=torch.float32)
    image[mask.bool()] = inside
    return image.unsqueeze(0)


def make_smooth_image(mask):
    mask_np = mask.numpy().astype(bool)
    signed = (distance_transform_edt(mask_np) -
              distance_transform_edt(~mask_np))
    smooth = 0.2 + 0.6 / (1.0 + np.exp(-signed / 2.0))
    grid_x = np.linspace(-0.04, 0.04, mask.shape[1], dtype=np.float32)
    smooth = smooth.astype(np.float32) + grid_x[None, :]
    return torch.from_numpy(smooth).unsqueeze(0)


class BMERUtilitiesTest(unittest.TestCase):
    def setUp(self):
        self.mask = make_mask()
        self.recipient = make_step_image(self.mask)
        self.donor = make_smooth_image(self.mask)

    def make_bank(self, image=None, position_bin=1):
        if image is None:
            image = self.donor
        bank = BoundaryProfileBank(
            radius=6, sectors=8, position_bins=3,
            min_foreground_pixels=16)
        added = bank.add(image.unsqueeze(0), self.mask.unsqueeze(0),
                         [position_bin], ['donor_slice_0'])
        self.assertEqual(added, 1)
        return bank.freeze()

    def test_extract_profile_contract(self):
        result = extract_profile_field(
            self.donor[0].numpy(), self.mask.numpy(),
            radius=6, sectors=8, min_foreground_pixels=16)
        self.assertIsNotNone(result)
        self.assertEqual(tuple(result['field'].shape), (8, 13))
        self.assertTrue(torch.isfinite(result['field']).all())
        self.assertEqual(result['geometry']['taper'].shape, self.mask.shape)

    def test_invalid_masks_are_skipped(self):
        bank = BoundaryProfileBank(
            radius=4, sectors=4, position_bins=1,
            min_foreground_pixels=16)
        empty = torch.zeros_like(self.mask)
        added = bank.add(self.recipient.unsqueeze(0), empty.unsqueeze(0),
                         [0], ['empty'])
        self.assertEqual(added, 0)
        self.assertEqual(bank.skipped, 1)
        with self.assertRaises(RuntimeError):
            bank.freeze()

    def test_sampling_is_deterministic(self):
        bank = self.make_bank()
        first_generator = torch.Generator().manual_seed(17)
        second_generator = torch.Generator().manual_seed(17)
        first = bank.sample([1, 0, 2], generator=first_generator)
        second = bank.sample([1, 0, 2], generator=second_generator)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1:], second[1:])

    def test_same_profile_is_identity(self):
        bank = self.make_bank(image=self.recipient)
        augmented, metadata = resynthesize_labeled_images(
            self.recipient.unsqueeze(0), self.mask.unsqueeze(0), bank, [1],
            probability=1.0, strength=(1.0, 1.0),
            generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(augmented, self.recipient.unsqueeze(0)))
        self.assertEqual(float(metadata['mean_absolute_change']), 0.0)
        self.assertFalse(augmented.requires_grad)

    def test_donor_changes_only_boundary_ribbon(self):
        bank = self.make_bank()
        augmented, metadata = resynthesize_labeled_images(
            self.recipient.unsqueeze(0), self.mask.unsqueeze(0), bank, [1],
            probability=1.0, strength=(1.0, 1.0),
            generator=torch.Generator().manual_seed(5))
        self.assertEqual(augmented.shape, self.recipient.unsqueeze(0).shape)
        self.assertEqual(augmented.dtype, self.recipient.dtype)
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertGreater(float(metadata['mean_absolute_change']), 0.0)

        mask_np = self.mask.numpy().astype(bool)
        signed = (distance_transform_edt(mask_np) -
                  distance_transform_edt(~mask_np))
        outside = torch.from_numpy(np.abs(signed) > bank.radius)
        original = self.recipient.unsqueeze(0)[0, 0]
        rendered = augmented[0, 0]
        self.assertTrue(torch.equal(original[outside], rendered[outside]))

    def test_probability_zero_is_exact_identity(self):
        bank = self.make_bank()
        augmented, metadata = resynthesize_labeled_images(
            self.recipient.unsqueeze(0), self.mask.unsqueeze(0), bank, [1],
            probability=0.0,
            generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(augmented, self.recipient.unsqueeze(0)))
        self.assertEqual(float(metadata['applied_fraction']), 0.0)

    def test_position_mapping_and_lookup(self):
        names = ['case_a_slice_0', 'case_a_slice_1', 'case_a_slice_2',
                 'case_b_slice_8', 'unparsed']
        mapping = build_position_bin_map(names, position_bins=3)
        self.assertEqual(mapping['case_a_slice_0'], 0)
        self.assertEqual(mapping['case_a_slice_1'], 1)
        self.assertEqual(mapping['case_a_slice_2'], 2)
        self.assertEqual(mapping['case_b_slice_8'], 1)
        self.assertEqual(mapping['unparsed'], 1)
        self.assertEqual(lookup_position_bins(names[:2], mapping), [0, 1])

    def test_bank_save_load_round_trip(self):
        bank = self.make_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bank.pt'
            bank.save(str(path))
            loaded = BoundaryProfileBank.load(str(path))
            self.assertEqual(bank.summary(), loaded.summary())
            generator_a = torch.Generator().manual_seed(19)
            generator_b = torch.Generator().manual_seed(19)
            self.assertTrue(torch.equal(
                bank.sample([1], generator=generator_a)[0],
                loaded.sample([1], generator=generator_b)[0]))

    def test_invalid_arguments_fail_loudly(self):
        with self.assertRaises(ValueError):
            BoundaryProfileBank(radius=0)
        bank = self.make_bank()
        with self.assertRaises(ValueError):
            resynthesize_labeled_images(
                self.recipient.unsqueeze(0), self.mask.unsqueeze(0), bank,
                [1], probability=1.5)


if __name__ == '__main__':
    unittest.main()

