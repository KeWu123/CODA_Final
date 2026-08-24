# H3 Protocol: Strong Simple Baseline

## Purpose

Establish whether a clean, tuned weak-to-strong Mean Teacher already explains most potential gains.

## Required Baselines

- Independent standard Mean Teacher implementation.
- BCP-derived code with all Copy-Paste and Copy-Paste pretraining removed.
- FixMatch/UniMatch-style weak-to-strong variant with the same U-Net and preprocessing.

## Checks

- Same patient-level splits, backbone, optimizer, number of iterations, and labeled/unlabeled batch ratio.
- Report mean and confidence interval across folds/label draws.
- Sweep only a small predeclared set: confidence threshold, consistency weight, EMA decay, and strong augmentation magnitude.
- Record whether the BCP-derived implementation contains residual differences such as two-stage pretraining, non-maximum connected-component cleanup, mixed loss weighting, or data sampling.

If the BCP-derived no-mix code materially differs from standard Mean Teacher, it is not a clean baseline and must be separated in all tables.
