import sys
import unittest
from pathlib import Path


try:
    import torch
except ModuleNotFoundError:  # The local audit machine intentionally lacks it.
    torch = None


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))


@unittest.skipIf(torch is None, 'PyTorch is unavailable in this environment')
class SliceEqAntitheticUtilitiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.sliceeq_antithetic import (
            antithetic_pair_diagnostics, antithetic_phase_weights,
            gaussian_profile_weights)
        cls.antithetic_pair_diagnostics = staticmethod(
            antithetic_pair_diagnostics)
        cls.antithetic_phase_weights = staticmethod(
            antithetic_phase_weights)
        cls.gaussian_profile_weights = staticmethod(
            gaussian_profile_weights)

    def test_reflection_reverses_taps_and_preserves_center_weight(self):
        sigma = torch.tensor([0.45, 0.65, 0.85])
        phase = torch.tensor([-0.25, 0.10, 0.25])
        primary = self.gaussian_profile_weights(
            (-1, 0, 1), sigma, phase)
        reflected = self.antithetic_phase_weights(
            (-1, 0, 1), sigma, phase)
        self.assertTrue(torch.allclose(reflected, primary.flip(dims=(1,))))
        self.assertTrue(torch.allclose(reflected[:, 1], primary[:, 1]))
        self.assertTrue(torch.allclose(
            reflected.sum(dim=1), torch.ones(3)))

    def test_two_marginals_preserve_the_original_symmetric_phase_domain(self):
        phase = torch.linspace(-0.25, 0.25, steps=101)
        paired = torch.cat((phase, -phase))
        self.assertAlmostEqual(float(paired.mean()), 0.0, places=7)
        self.assertAlmostEqual(
            float(paired.abs().mean()), float(phase.abs().mean()), places=7)
        self.assertEqual(float(paired.min()), -0.25)
        self.assertEqual(float(paired.max()), 0.25)

    def test_pair_diagnostics_detect_activity_and_exact_symmetry(self):
        sigma = torch.tensor([0.65])
        phase = torch.tensor([0.20])
        primary = self.gaussian_profile_weights(
            (-1, 0, 1), sigma, phase)
        reflected = self.antithetic_phase_weights(
            (-1, 0, 1), sigma, phase)
        stack = torch.tensor([[[[[0.0]]], [[[1.0]]], [[[2.0]]]]])
        primary_image = (stack * primary.view(1, 3, 1, 1, 1)).sum(1)
        reflected_image = (
            stack * reflected.view(1, 3, 1, 1, 1)).sum(1)
        primary_occ = torch.tensor([[[[0.7]], [[0.3]]]])
        reflected_occ = torch.tensor([[[[0.4]], [[0.6]]]])
        values = self.antithetic_pair_diagnostics(
            primary, reflected, primary_image, reflected_image,
            primary_occ, reflected_occ, phase)
        self.assertEqual(float(values['phase_pair_residual']), 0.0)
        self.assertGreater(float(values['weight_l1_separation']), 0.0)
        self.assertGreater(float(values['image_absolute_separation']), 0.0)
        self.assertEqual(float(values['hard_target_disagreement']), 1.0)

    def test_invalid_parameters_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.gaussian_profile_weights(
                (-1, 0, 1), torch.tensor([0.0]), torch.tensor([0.0]))
        with self.assertRaises(ValueError):
            self.gaussian_profile_weights(
                (-1, 0, 1), torch.ones(2), torch.zeros(1))


if __name__ == '__main__':
    unittest.main()
