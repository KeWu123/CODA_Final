import unittest
import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_oaac import ordered_appearance_transform  # noqa: E402


class SliceEqOAACTensorTest(unittest.TestCase):
    def _generator(self, seed):
        return torch.Generator(device='cpu').manual_seed(seed)

    def test_shape_dtype_finite_and_activity(self):
        images = torch.linspace(-1.0, 2.0, 2 * 1 * 8 * 9).reshape(
            2, 1, 8, 9)
        transformed, metadata = ordered_appearance_transform(
            images, self._generator(1339))
        self.assertEqual(transformed.shape, images.shape)
        self.assertEqual(transformed.dtype, images.dtype)
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertGreater(
            float(metadata['appearance_normalized_absolute_change']), 0.0)
        self.assertEqual(
            float(metadata['appearance_active_sample_fraction']), 1.0)
        self.assertIn('appearance_below_source_min_fraction', metadata)
        self.assertIn('appearance_above_source_max_fraction', metadata)

    def test_constant_images_are_exact_identity(self):
        images = torch.full((3, 1, 7, 6), 4.25)
        transformed, metadata = ordered_appearance_transform(
            images, self._generator(1339))
        self.assertTrue(torch.equal(transformed, images))
        self.assertEqual(
            float(metadata['appearance_active_sample_fraction']), 0.0)

    def test_seed_is_reproducible_and_independent(self):
        images = torch.rand((4, 1, 8, 8), generator=self._generator(2))
        first, _ = ordered_appearance_transform(
            images, self._generator(1339))
        repeated, _ = ordered_appearance_transform(
            images, self._generator(1339))
        different, _ = ordered_appearance_transform(
            images, self._generator(1340))
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))

    def test_transform_is_strictly_monotonic_on_nonconstant_sample(self):
        images = torch.linspace(-2.0, 3.0, 100).reshape(1, 1, 10, 10)
        transformed, _ = ordered_appearance_transform(
            images, self._generator(1339))
        source_delta = images.flatten()[1:] - images.flatten()[:-1]
        target_delta = transformed.flatten()[1:] - transformed.flatten()[:-1]
        self.assertTrue((source_delta > 0).all())
        self.assertTrue((target_delta > 0).all())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ordered_appearance_transform(
                torch.zeros(1, 8, 8), self._generator(1))
        with self.assertRaises(ValueError):
            ordered_appearance_transform(
                torch.zeros(1, 0, 8, 8), self._generator(1))
        with self.assertRaises(TypeError):
            ordered_appearance_transform(
                torch.zeros(1, 1, 8, 8, dtype=torch.long),
                self._generator(1))
        invalid = torch.zeros(1, 1, 8, 8)
        invalid[0, 0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            ordered_appearance_transform(invalid, self._generator(1))


if __name__ == '__main__':
    unittest.main()
