import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_ROOT))

from utils.coda import (  # noqa: E402
    couple_pseudo_target,
    evidence_augment,
    gaussian_noise_degradation,
    lcc_preserve_teacher_probabilities,
    resolution_degradation,
    soft_cross_entropy,
    soft_dice_loss,
)


class CoDAUtilitiesTest(unittest.TestCase):
    def setUp(self):
        row = torch.linspace(0.0, 1.0, 32).view(1, 1, 1, 32)
        column = torch.linspace(0.0, 0.5, 32).view(1, 1, 32, 1)
        self.images = (row + column).repeat(2, 1, 1, 1)

    def assert_valid_degradation(self, strong, gamma):
        self.assertEqual(strong.shape, self.images.shape)
        self.assertEqual(gamma.shape, (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(strong).all())
        self.assertTrue(torch.isfinite(gamma).all())
        self.assertGreaterEqual(float(gamma.min()), 0.0)
        self.assertLessEqual(float(gamma.max()), 1.0)
        self.assertFalse(strong.requires_grad)
        self.assertFalse(gamma.requires_grad)

    def test_resolution_degradation_contract(self):
        generator = torch.Generator().manual_seed(11)
        strong, gamma, metadata = resolution_degradation(
            self.images, 0.5, 0.5, generator=generator)
        self.assert_valid_degradation(strong, gamma)
        self.assertEqual(metadata["family"], "resolution")
        self.assertTrue(torch.equal(metadata["scale"], torch.tensor(0.5)))

    def test_resolution_detects_lost_high_frequency_evidence(self):
        stripe = (torch.arange(32) % 4 < 2).float()
        stripe = stripe.view(1, 1, 1, 32).repeat(1, 1, 32, 1)
        _, gamma, _ = resolution_degradation(stripe, 0.25, 0.25)
        self.assertGreater(float(gamma.mean()), 0.5)

    def test_gaussian_degradation_is_deterministic(self):
        first_generator = torch.Generator().manual_seed(17)
        second_generator = torch.Generator().manual_seed(17)
        first = gaussian_noise_degradation(
            self.images, 0.1, 0.1, generator=first_generator)
        second = gaussian_noise_degradation(
            self.images, 0.1, 0.1, generator=second_generator)
        self.assert_valid_degradation(first[0], first[1])
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[2]["family"], "gaussian_noise")

    def test_constant_background_has_no_false_evidence_loss(self):
        constant = torch.ones(2, 1, 32, 32)
        generator = torch.Generator().manual_seed(19)
        strong, gamma, _ = gaussian_noise_degradation(
            constant, 0.2, 0.2, generator=generator)
        self.assertTrue(torch.equal(strong, constant))
        self.assertTrue(torch.equal(gamma, torch.zeros_like(gamma)))

    def test_evidence_augment_fixed_seed(self):
        first_generator = torch.Generator().manual_seed(23)
        second_generator = torch.Generator().manual_seed(23)
        first = evidence_augment(self.images, generator=first_generator)
        second = evidence_augment(self.images, generator=second_generator)
        self.assertEqual(first[2]["family"], second[2]["family"])
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_lcc_prior_preserves_only_connected_foreground_probability(self):
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 1] = 2.0
        lcc = torch.zeros(1, 4, 4)
        lcc[:, 1:3, 1:3] = 1
        target = lcc_preserve_teacher_probabilities(logits, lcc)
        original = torch.softmax(logits, dim=1)
        self.assertTrue(torch.allclose(target[:, 1, 1:3, 1:3],
                                       original[:, 1, 1:3, 1:3]))
        self.assertTrue(torch.equal(target[:, 1, 0, 0], torch.zeros(1)))
        self.assertTrue(torch.equal(target[:, 0, 0, 0], torch.ones(1)))
        self.assertTrue(torch.allclose(target.sum(dim=1), torch.ones(1, 4, 4)))

    def test_coupling_limit_cases(self):
        target = torch.tensor([[[[0.8]], [[0.2]]]])
        zero = couple_pseudo_target(target, torch.zeros(1, 1, 1, 1))
        one = couple_pseudo_target(target, torch.ones(1, 1, 1, 1))
        self.assertTrue(torch.allclose(zero, target))
        self.assertTrue(torch.allclose(one, torch.full_like(target, 0.5)))
        self.assertTrue(torch.allclose(one.sum(dim=1), torch.ones(1, 1, 1)))

    def test_soft_losses_are_finite_and_backpropagate(self):
        logits = torch.randn(2, 2, 16, 16, requires_grad=True)
        teacher_logits = torch.randn(2, 2, 16, 16)
        lcc = teacher_logits.argmax(dim=1)
        pseudo_target = lcc_preserve_teacher_probabilities(teacher_logits, lcc)
        gamma = torch.rand(2, 1, 16, 16)
        target = couple_pseudo_target(pseudo_target, gamma)
        probabilities = torch.softmax(logits, dim=1)
        loss = 0.5 * (soft_cross_entropy(logits, target) +
                      soft_dice_loss(probabilities, target))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertFalse(target.requires_grad)

    def test_soft_cross_entropy_matches_hard_ce_for_one_hot_target(self):
        logits = torch.randn(2, 2, 8, 8)
        labels = torch.randint(0, 2, (2, 8, 8))
        one_hot = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).float()
        self.assertTrue(torch.allclose(
            soft_cross_entropy(logits, one_hot),
            F.cross_entropy(logits, labels)))


if __name__ == "__main__":
    unittest.main()
