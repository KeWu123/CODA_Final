import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from utils.sliceeq_reliability import (  # noqa: E402
    INDEPENDENT_DROPOUT,
    STACK_SHARED_DROPOUT,
    binary_dice_per_sample,
    jensen_shannon_map,
    normalized_weighted_mean,
    occupancy_brier_map,
    reliability_from_js,
    spearman_correlation,
    structured_stack_dropout,
    temporary_stack_dropout,
    top_fraction_error_ratio,
)


class SliceEqReliabilityUtilitiesTest(unittest.TestCase):
    def test_stack_shared_dropout_reuses_center_mask_and_rng_budget(self):
        inputs = torch.ones(6, 4, 8, 8)
        torch.manual_seed(17)
        independent = structured_stack_dropout(
            inputs, 0.4, 3, INDEPENDENT_DROPOUT)
        independent_state = torch.random.get_rng_state().clone()
        torch.manual_seed(17)
        shared = structured_stack_dropout(
            inputs, 0.4, 3, STACK_SHARED_DROPOUT)
        shared_state = torch.random.get_rng_state().clone()

        independent_keep = independent.reshape(2, 3, 4, 8, 8) > 0
        shared_keep = shared.reshape(2, 3, 4, 8, 8) > 0
        self.assertTrue(torch.equal(shared_keep[:, 0], shared_keep[:, 1]))
        self.assertTrue(torch.equal(shared_keep[:, 1], shared_keep[:, 2]))
        self.assertTrue(torch.equal(
            shared_keep[:, 1], independent_keep[:, 1]))
        self.assertFalse(torch.equal(
            independent_keep[:, 0], independent_keep[:, 1]))
        self.assertTrue(torch.equal(independent_state, shared_state))

    def test_dropout_marginal_probability_and_scale_are_preserved(self):
        inputs = torch.ones(300, 20, 20)
        torch.manual_seed(23)
        output = structured_stack_dropout(
            inputs, 0.25, 3, STACK_SHARED_DROPOUT)
        zero_rate = float((output == 0).float().mean())
        self.assertAlmostEqual(zero_rate, 0.25, delta=0.02)
        self.assertAlmostEqual(float(output.mean()), 1.0, delta=0.04)
        kept = output[output > 0]
        self.assertTrue(torch.allclose(
            kept, torch.full_like(kept, 1.0 / 0.75)))

    def test_dropout_identity_and_temporary_patch_restoration(self):
        inputs = torch.randn(3, 2, 2)
        self.assertTrue(torch.equal(
            structured_stack_dropout(
                inputs, 0.5, 3, STACK_SHARED_DROPOUT, training=False),
            inputs))
        self.assertTrue(torch.equal(
            structured_stack_dropout(
                inputs, 0.0, 3, STACK_SHARED_DROPOUT), inputs))

        module = nn.Sequential(nn.Dropout(p=0.5))
        dropout = module[0]
        self.assertNotIn('forward', dropout.__dict__)
        with temporary_stack_dropout(
                module, STACK_SHARED_DROPOUT, 3) as patched:
            self.assertEqual(patched, 1)
            self.assertIn('forward', dropout.__dict__)
        self.assertNotIn('forward', dropout.__dict__)

    def test_js_distinguishes_disagreement_from_fractional_entropy(self):
        agreed_fractional = torch.tensor(
            [[[[0.5]], [[0.5]]]], dtype=torch.float32)
        zero_js = jensen_shannon_map(
            agreed_fractional, agreed_fractional)
        zero_weight = reliability_from_js(zero_js, num_classes=2)
        self.assertEqual(zero_js.dtype, torch.float64)
        self.assertTrue(torch.allclose(zero_js, torch.zeros_like(zero_js)))
        self.assertTrue(torch.allclose(
            zero_weight, torch.ones_like(zero_weight)))

        foreground = torch.tensor(
            [[[[0.0]], [[1.0]]]], dtype=torch.float32)
        background = torch.tensor(
            [[[[1.0]], [[0.0]]]], dtype=torch.float32)
        maximal_js = jensen_shannon_map(foreground, background)
        maximal_weight = reliability_from_js(maximal_js, num_classes=2)
        self.assertAlmostEqual(
            float(maximal_js.item()), math.log(2.0), places=7)
        self.assertAlmostEqual(float(maximal_weight.item()), 0.0, places=7)

    def test_brier_weighted_mean_and_binary_dice(self):
        exact = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        candidate = torch.tensor([[[[0.75, 0.25]], [[0.25, 0.75]]]])
        brier = occupancy_brier_map(candidate, exact)
        self.assertTrue(torch.allclose(
            brier, torch.tensor([[[0.0625, 0.0625]]])))
        weights = torch.tensor([[[1.0, 3.0]]])
        weighted = normalized_weighted_mean(brier, weights)
        self.assertAlmostEqual(float(weighted), 0.0625)
        dice = binary_dice_per_sample(
            torch.tensor([[[0, 1], [1, 0]]]),
            torch.tensor([[[0, 1], [1, 0]]]))
        self.assertAlmostEqual(float(dice.item()), 1.0)

    def test_rank_and_exact_top_fraction_statistics(self):
        self.assertAlmostEqual(
            spearman_correlation([0, 1, 1, 2], [0, 2, 2, 4]), 1.0)
        self.assertIsNone(spearman_correlation([1, 1], [1, 2]))
        ratio = top_fraction_error_ratio(
            np.asarray([4, 3, 2, 1, 0]),
            np.asarray([10, 1, 1, 1, 1]),
            fraction=0.20)
        self.assertAlmostEqual(ratio, 10.0)

    def test_invalid_incomplete_stack_is_rejected(self):
        with self.assertRaises(ValueError):
            structured_stack_dropout(
                torch.ones(5, 2), 0.5, 3, STACK_SHARED_DROPOUT)


if __name__ == '__main__':
    unittest.main()
