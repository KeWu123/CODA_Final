# Baseline Audit: E:/Desktop/Baseline

Audit date: 2026-08-10. The baseline folder was inspected read-only and was not modified.

## Integrity Fingerprints

| File | SHA-256 |
|---|---|
| `code/train_baseline.py` | `54393FCB977A3E4F199420885B6F6ACBD8B1D2B320C820979F355C003CD3EEC8` |
| `code/test_baseline.py` | `31CC57D26FB3476F55A593445E10BBC4EFA96E6DBFDE778BAA2D65C599491682` |
| `code/dataloaders/dataset.py` | `7A5B3C28EBEAF7AA2F64E5F111F88BD15F6075AFC53238B477BD1A8511B2206A` |

## Locked Data Contract

No proposal may alter any of the following:

- `train_slices.list`, `val.list`, or `test.list` contents or order;
- `patients_to_slices()` and the current `labelnum=7` selection;
- labeled indices `range(0, labeled_slice)` and unlabeled indices `range(labeled_slice, total_slices)`;
- `TwoStreamBatchSampler` ordering and `24 = 12 labeled + 12 unlabeled` batches;
- validation/test cases, patch size, seed, pretrain/self-train iterations, architecture, optimizer, learning rate, or EMA decay.

The copied folder does not contain the three list files or H5 data. Their contents cannot be independently verified on this Windows machine. Before Linux experiments, record their SHA-256 hashes without editing them.

## What the Baseline Actually Does

### Supervised pretraining

- 2D U-Net embedded directly in `train_baseline.py`.
- 10,000 iterations, SGD with constant learning rate `0.01`, momentum `0.9`, weight decay `1e-4`.
- Supervised loss is `0.5 * (CE + Dice)`.
- The batch sampler still returns labeled and unlabeled slices, but pretraining forwards only the first 12 labeled tensors.

### Self-training

- Student and EMA teacher are initialized from the same best pretraining checkpoint.
- The optimizer state, including momentum, is restored from pretraining.
- Student forwards the complete batch; teacher forwards the unlabeled tail.
- Both receive the exact same already-augmented unlabeled tensor. There is no separate weak and strong view.
- Teacher logits are converted by `argmax`, followed by per-slice largest-connected-component filtering, then one-hot encoding.
- The one-hot tensor `ema_output_soft` is computed but unused.
- After a 1,000-iteration hard delay, unlabeled loss is `0.5 * (CE + Dice)` against the hard pseudo-mask.
- Consistency weight ramps toward `5 * 0.1 = 0.5`; EMA decay is fixed at `0.99`.

The scientifically accurate name is therefore **EMA hard-pseudo-label self-training**, not canonical Mean Teacher consistency and not BCP.

## Existing Augmentation

`RandomGenerator` applies one shared sample-level transformation before teacher/student branching:

- random 90-degree rotation plus flip, or random rotation in approximately `[-20,20)` degrees;
- resize to `256x256` using nearest-neighbor interpolation for both MRI image and mask.

These transformations are part of the locked baseline. The proposed strong degradation must be applied only inside `self_train` after the existing loader returns the batch.

## Evaluation Audit

- Validation uses `val.list`; test uses `test.list`.
- Testing is slice-wise 2D inference followed by volume-level Dice, Jaccard, HD95, and ASD.
- Test NMS defaults to off, whereas teacher pseudo-label NMS is always on per 2D slice.
- `medpy` metrics are computed without voxel spacing, so HD95/ASD are voxel-index distances, not physical millimeters.
- `SetSpacing((1,1,10))` is applied only to saved NIfTI files and does not affect reported metrics.
- `test_baseline.py` ends with a bare identifier (`vvvv...`), causing a `NameError` after inference results have already been written. This should be treated as a reproducibility bug fix, not a method change.

## Available Logs

The folder contains sliding-window ablation logs for a different SRCLQT/SDAA method, not `train_baseline.py`. They cannot be used as the numerical baseline. No baseline checkpoint, `log.txt`, or `performance.txt` is present in the copied folder.

## Consequence for the Research Plan

The minimal legal intervention is confined to the unlabeled portion of `self_train`:

1. preserve the loader output as the teacher weak view;
2. derive a student strong view from the same tensor;
3. retain the existing teacher, EMA update, LCC prior, sampler, and schedules;
4. change only the dense pseudo-target construction and corresponding soft unsupervised loss.

