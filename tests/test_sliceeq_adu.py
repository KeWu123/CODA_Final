import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_adu import (  # noqa: E402
    acquisition_aligned_reliability,
    isolated_stochastic_teacher_forward,
    reliability_weighted_soft_segmentation_loss,
)
from utils.sliceeq_occ import soft_segmentation_loss  # noqa: E402
from utils.sliceeq_reliability import snapshot_buffers  # noqa: E402


class SliceEqADUUtilitiesTest(unittest.TestCase):
    def test_agreement_recovers_parent_loss_and_gradient(self):
        occupancy = torch.tensor([
            [[[0.5, 0.75]], [[0.5, 0.25]]],
            [[[1.0, 0.2]], [[0.0, 0.8]]],
        ])
        mean, js, reliability = acquisition_aligned_reliability(
            occupancy, occupancy)
        self.assertTrue(torch.equal(mean, occupancy))
        self.assertTrue(torch.allclose(js, torch.zeros_like(js)))
        self.assertTrue(torch.allclose(
            reliability, torch.ones_like(reliability)))
        self.assertFalse(mean.requires_grad)
        self.assertFalse(reliability.requires_grad)

        parent_logits = torch.tensor([
            [[[0.2, -0.5]], [[-0.3, 0.6]]],
            [[[0.7, -0.1]], [[-0.2, 0.4]]],
        ], requires_grad=True)
        adu_logits = parent_logits.detach().clone().requires_grad_(True)
        parent = soft_segmentation_loss(parent_logits, occupancy)
        adu = reliability_weighted_soft_segmentation_loss(
            adu_logits, mean, reliability)
        for parent_value, adu_value in zip(parent, adu):
            self.assertTrue(torch.allclose(
                parent_value, adu_value, atol=1e-7, rtol=1e-6))
        parent[0].backward()
        adu[0].backward()
        self.assertTrue(torch.allclose(
            parent_logits.grad, adu_logits.grad, atol=1e-7, rtol=1e-6))

    def test_maximal_disagreement_has_zero_weight_and_gradient(self):
        first = torch.tensor([[[[1.0]], [[0.0]]]])
        second = torch.tensor([[[[0.0]], [[1.0]]]])
        mean, _, reliability = acquisition_aligned_reliability(first, second)
        self.assertAlmostEqual(float(reliability.item()), 0.0, places=7)
        logits = torch.randn(1, 2, 1, 1, requires_grad=True)
        loss, loss_ce, loss_dice = \
            reliability_weighted_soft_segmentation_loss(
                logits, mean, reliability)
        self.assertAlmostEqual(float(loss.item()), 0.0)
        self.assertAlmostEqual(float(loss_ce.item()), 0.0)
        self.assertAlmostEqual(float(loss_dice.item()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_weighting_is_global_not_per_slice(self):
        occupancy = torch.tensor([
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
        ])
        logits = torch.zeros(2, 2, 1, 1, requires_grad=True)
        reliability = torch.tensor([[[1.0]], [[0.0]]])
        _, ce, _ = reliability_weighted_soft_segmentation_loss(
            logits, occupancy, reliability)
        self.assertAlmostEqual(
            float(ce.item()), float(torch.log(torch.tensor(2.0))), places=6)
        ce.backward()
        self.assertTrue(torch.equal(
            logits.grad[1], torch.zeros_like(logits.grad[1])))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_isolated_teacher_forward_restores_buffers_and_rng(self):
        device = torch.device('cuda')
        teacher = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.Dropout(p=0.5),
            nn.Conv2d(4, 2, 1),
        ).to(device).train()
        reference = copy.deepcopy(teacher)
        inputs = torch.randn(6, 1, 8, 8, device=device)

        torch.cuda.manual_seed(41)
        with torch.no_grad():
            first = teacher(inputs)
        buffers_after_first = snapshot_buffers(teacher)
        rng_after_first = torch.cuda.get_rng_state(device).clone()
        parameters_before = [
            value.detach().clone() for value in teacher.parameters()]
        second = isolated_stochastic_teacher_forward(
            teacher, inputs, seed=7000044)

        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(
            rng_after_first, torch.cuda.get_rng_state(device)))
        for name, value in teacher.named_buffers():
            self.assertTrue(torch.equal(value, buffers_after_first[name]), name)
        for before, after in zip(parameters_before, teacher.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertTrue(teacher.training)

        reference.load_state_dict(teacher.state_dict())
        reference.train()
        torch.cuda.set_rng_state(rng_after_first, device)
        with torch.no_grad():
            expected_next = reference(inputs)
        torch.cuda.set_rng_state(rng_after_first, device)
        with torch.no_grad():
            actual_next = teacher(inputs)
        self.assertTrue(torch.equal(expected_next, actual_next))


if __name__ == '__main__':
    unittest.main()
