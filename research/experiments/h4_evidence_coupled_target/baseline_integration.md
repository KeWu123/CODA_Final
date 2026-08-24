# Baseline Integration Map

This document maps H4 to the audited `E:/Desktop/Baseline` implementation. It is a design record; the baseline folder has not been edited.

## Unchanged Path

- Keep `BaseDataSets`, `RandomGenerator`, and `TwoStreamBatchSampler` unchanged.
- Keep pretraining unchanged.
- Keep the labeled part of the self-training forward and supervised loss unchanged.
- Keep EMA initialization/update and consistency weighting unchanged.
- Keep validation and test lists unchanged.

## Replacement Boundary in `self_train`

Current logical block:

1. `ema_inputs = unlabeled_volume_batch`;
2. `outputs = model(volume_batch)`;
3. teacher `argmax + 2D LCC` pseudo-label;
4. unlabeled hard CE + hard Dice.

Proposed logical block:

1. `weak_u = unlabeled_volume_batch`;
2. `(strong_u, gamma) = evidence_augment(weak_u)`;
3. construct `student_inputs` by replacing only its unlabeled tail with `strong_u`;
4. teacher predicts `q` on `weak_u`, then retain the existing LCC topology prior without discarding all probabilities;
5. construct `q_aug = (1-gamma)*q_lcc + gamma/C`;
6. compute soft CE + soft Dice on the student's unlabeled output.

## Required Helper Interfaces

- `evidence_augment(x, generator) -> strong_x, gamma, metadata`
- `lcc_calibrated_probs(logits) -> q_lcc`
- `couple_target(q_lcc, gamma) -> q_aug`
- `soft_cross_entropy(logits, q_aug) -> scalar`
- `soft_dice_loss(probabilities, q_aug) -> scalar`

All helpers should live outside the data loader so the locked case list and sampling order are unaffected.

## Determinism

Use PyTorch RNG seeded by the existing `args.seed`; do not instantiate an unseeded NumPy generator. Log augmentation family, severity summary, and mean `gamma` to TensorBoard without changing sampling.

## Pre-Run Reproducibility Fixes

These do not change the method or data split:

- remove the final bare `vvvv...` expression in `test_baseline.py` or run an exact copied test script with that line removed;
- record SHA-256 hashes of the three list files on Linux;
- record the baseline code hashes from `baseline/audit.md`;
- state explicitly that current HD95/ASD are voxel-index distances.

