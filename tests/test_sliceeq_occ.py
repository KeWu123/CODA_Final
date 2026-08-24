import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils import losses  # noqa: E402
from utils.sliceeq_occ import (  # noqa: E402
    occupancy_diagnostics,
    soft_cross_entropy,
    soft_dice_loss,
    soft_segmentation_loss,
    validate_occupancy,
)


class SliceEqOccupancyUtilitiesTest(unittest.TestCase):
    def test_one_hot_soft_losses_match_locked_hard_losses(self):
        logits = torch.tensor(
            [[[[2.0, -1.0]], [[-0.5, 1.5]]]], requires_grad=True)
        labels = torch.tensor([[[0, 1]]])
        occupancy = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2)
        occupancy = occupancy.float()

        expected_ce = F.cross_entropy(logits, labels)
        expected_dice = losses.DiceLoss(2)(
            torch.softmax(logits, dim=1), labels.unsqueeze(1))
        self.assertTrue(torch.allclose(
            soft_cross_entropy(logits, occupancy), expected_ce))
        self.assertTrue(torch.allclose(
            soft_dice_loss(torch.softmax(logits, dim=1), occupancy),
            expected_dice))

    def test_fractional_cross_entropy_is_exact_distributional_target(self):
        logits = torch.tensor([[[[0.0]], [[0.0]]]], requires_grad=True)
        occupancy = torch.tensor([[[[0.25]], [[0.75]]]])
        loss = soft_cross_entropy(logits, occupancy)
        self.assertTrue(torch.allclose(loss, torch.log(torch.tensor(2.0))))

    def test_soft_segmentation_loss_has_finite_gradient_and_detached_target(self):
        logits = torch.randn(2, 2, 4, 4, requires_grad=True)
        foreground = torch.rand(2, 1, 4, 4, requires_grad=True)
        occupancy = torch.cat((1.0 - foreground, foreground), dim=1)
        total, loss_ce, loss_dice = soft_segmentation_loss(logits, occupancy)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(loss_ce))
        self.assertTrue(torch.isfinite(loss_dice))
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNone(foreground.grad)

    def test_diagnostics_distinguish_fractional_from_hard_change(self):
        occupancy = torch.tensor(
            [[[[1.0, 0.7]], [[0.0, 0.3]]]])
        center = torch.tensor([[[0, 0]]])
        metadata = occupancy_diagnostics(occupancy, center)
        self.assertAlmostEqual(
            float(metadata['fractional_pixel_fraction']), 0.5)
        self.assertAlmostEqual(
            float(metadata['hard_target_changed_fraction']), 0.0)
        self.assertGreater(
            float(metadata['normalized_occupancy_entropy']), 0.0)
        self.assertGreater(
            float(metadata['occupancy_deviation_from_center']), 0.0)

    def test_invalid_occupancy_fails_loudly(self):
        with self.assertRaises(ValueError):
            validate_occupancy(torch.ones(1, 2, 2))
        with self.assertRaises(ValueError):
            validate_occupancy(torch.full((1, 2, 2, 2), 0.6))
        invalid = torch.tensor([[[[float('nan')]], [[0.0]]]])
        with self.assertRaises(FloatingPointError):
            validate_occupancy(invalid)


if __name__ == '__main__':
    unittest.main()

