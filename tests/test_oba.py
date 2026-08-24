import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.oba import (  # noqa: E402
    gaussian_noise_pair,
    log_gamma_pair,
    orbit_balanced_augment,
    smooth_bias_pair,
)


class OBAUtilitiesTest(unittest.TestCase):
    def setUp(self):
        row = torch.linspace(-0.4, 1.0, 32).view(1, 1, 1, 32)
        column = torch.linspace(0.0, 0.5, 32).view(1, 1, 32, 1)
        self.images = (row + column).repeat(6, 1, 1, 1)

    def assert_valid_pair(self, plus, minus, reference=None):
        if reference is None:
            reference = self.images
        self.assertEqual(plus.shape, reference.shape)
        self.assertEqual(minus.shape, reference.shape)
        self.assertEqual(plus.dtype, reference.dtype)
        self.assertEqual(minus.dtype, reference.dtype)
        self.assertTrue(torch.isfinite(plus).all())
        self.assertTrue(torch.isfinite(minus).all())
        self.assertFalse(plus.requires_grad)
        self.assertFalse(minus.requires_grad)

    def test_gaussian_pair_is_exactly_antithetic(self):
        plus, minus, metadata = gaussian_noise_pair(
            self.images, 0.1, 0.1,
            generator=torch.Generator().manual_seed(7))
        self.assert_valid_pair(plus, minus)
        self.assertTrue(torch.allclose(
            plus + minus, 2.0 * self.images, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(
            metadata['severity'], torch.full((6,), 0.1)))

    def test_log_gamma_uses_reciprocal_exponents(self):
        plus, minus, metadata = log_gamma_pair(
            self.images, 0.25, 0.25,
            generator=torch.Generator().manual_seed(9))
        self.assert_valid_pair(plus, minus)
        product = metadata['plus_exponent'] * metadata['minus_exponent']
        self.assertTrue(torch.allclose(product, torch.ones_like(product)))
        self.assertFalse(torch.equal(plus, minus))

    def test_smooth_bias_pair_is_finite_and_opposed(self):
        plus, minus, metadata = smooth_bias_pair(
            self.images, 0.2, 0.2, grid_size=4,
            generator=torch.Generator().manual_seed(11))
        self.assert_valid_pair(plus, minus)
        plus_delta = (plus - self.images).flatten(1)
        minus_delta = (minus - self.images).flatten(1)
        dot = (plus_delta * minus_delta).sum(dim=1)
        self.assertTrue((dot < 0.0).all())
        self.assertTrue(torch.isfinite(metadata['coordinate_rms']).all())

    def test_constant_images_are_identity_for_all_families(self):
        constant = torch.ones(3, 1, 16, 16)
        for function, kwargs in (
                (log_gamma_pair, {}),
                (smooth_bias_pair, {'grid_size': 4}),
                (gaussian_noise_pair, {})):
            plus, minus, _ = function(
                constant, generator=torch.Generator().manual_seed(13),
                **kwargs)
            self.assert_valid_pair(plus, minus, constant)
            self.assertTrue(torch.equal(plus, constant))
            self.assertTrue(torch.equal(minus, constant))

    def test_orbit_balanced_augment_is_deterministic(self):
        first = orbit_balanced_augment(
            self.images, bias_grid_size=4,
            generator=torch.Generator().manual_seed(17))
        second = orbit_balanced_augment(
            self.images, bias_grid_size=4,
            generator=torch.Generator().manual_seed(17))
        self.assert_valid_pair(first[0], first[1])
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(
            first[2]['family_ids'], second[2]['family_ids']))
        self.assertTrue(torch.equal(
            first[2]['severity'], second[2]['severity']))

    def test_sampling_and_diagnostic_contract(self):
        plus, minus, metadata = orbit_balanced_augment(
            self.images, bias_grid_size=4,
            generator=torch.Generator().manual_seed(19))
        self.assert_valid_pair(plus, minus)
        self.assertEqual(metadata['family_ids'].shape, (6,))
        self.assertEqual(metadata['severity'].shape, (6,))
        fractions = sum(
            metadata['family_fraction_{}'.format(name)]
            for name in ('log_gamma', 'smooth_bias', 'gaussian_noise'))
        self.assertTrue(torch.allclose(fractions, torch.tensor(1.0)))
        for name in ('plus_mean_absolute_change',
                     'minus_mean_absolute_change', 'displacement_cosine',
                     'midpoint_drift', 'pair_span'):
            self.assertTrue(torch.isfinite(metadata[name]))
        self.assertLess(float(metadata['displacement_cosine']), 0.0)

    def test_gaussian_only_reports_negative_one_cosine(self):
        _, _, metadata = orbit_balanced_augment(
            self.images, augmentations=('gaussian_noise',),
            noise_magnitude=(0.1, 0.1),
            generator=torch.Generator().manual_seed(23))
        self.assertTrue(torch.allclose(
            metadata['displacement_cosine'], torch.tensor(-1.0),
            atol=1e-5, rtol=1e-5))
        self.assertLess(float(metadata['midpoint_drift']), 1e-6)

    def test_invalid_arguments_fail_loudly(self):
        with self.assertRaises(ValueError):
            orbit_balanced_augment(self.images, augmentations=())
        with self.assertRaises(ValueError):
            orbit_balanced_augment(
                self.images,
                augmentations=('gaussian_noise', 'gaussian_noise'))
        with self.assertRaises(ValueError):
            orbit_balanced_augment(
                self.images, augmentations=('not_a_family',))
        with self.assertRaises(ValueError):
            orbit_balanced_augment(
                self.images, gamma_magnitude=(0.4, 0.1))
        with self.assertRaises(ValueError):
            smooth_bias_pair(self.images, grid_size=1)


if __name__ == '__main__':
    unittest.main()
