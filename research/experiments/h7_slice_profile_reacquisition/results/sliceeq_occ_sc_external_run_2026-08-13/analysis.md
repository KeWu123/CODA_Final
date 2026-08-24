# SliceEqOccSC exploratory result analysis

Date: 2026-08-13.  Status: negative; direction closed without tuning.

## Artifact identity

- User source log: `Z:\Downloads\log.txt`; SHA-256 `8aab816099fef816b70c8d233b52bf4ea253171a0bc772e16271ad65fd53bc52`.
- User performance file: `Z:\Downloads\performance.txt`; SHA-256 `a7695b192d74fd251614c9e77652bf172a03df6e3414bf7083f62d2d822fa5ad`.
- The local `training_log.txt` is a newline-normalized archival copy. The local `test_performance.txt` is byte-identical to the user performance file.
- Shared pretrain SHA-256 recorded in the log: `49e888ef8d40df423933a2790b71f8126001271dd59a2c3fe5bfb6aa247b9b9`.
- Seed 1337, 30,000 updates, loader batch 24, labeled batch 12, and effective student batch 36 are confirmed by the log.

## Outcome

| Method | Test Dice | Delta vs SliceEqOcc | Delta vs SliceEq |
|---|---:|---:|---:|
| SliceEq | 0.832603 | -0.011963 | 0 |
| SliceEqOcc | 0.844566 | 0 | +0.011963 |
| SliceEqOccSC | 0.836219 | -0.008347 | +0.003616 |

SliceEqOccSC's best validation Dice is 0.816251 at iteration 29,000; the final validation Dice is 0.815349. The run is healthy and convergent, so the negative test result is not attributable to a crash or an inactive operator.

Against SliceEqOcc, SC loses on 9/10 test cases. The paired Dice differences (SC minus Occ) are:

`+0.008763, -0.011679, -0.003951, -0.013411, -0.003809, -0.005869, -0.020055, -0.016969, -0.013021, -0.003475`.

This broad degradation is more consistent with reduced useful augmentation diversity than with one outlier case.

## Mechanistic reading

The implementation diagnostics show exactly zero within-case sigma and phase ranges, confirming that the intended scan-coherent constraint was active. Profile severity and fractional occupancy remained active. Therefore the result falsifies the useful part of H7.6a: sharing one synthetic profile across all slices of a case during an epoch does not improve ordinary PROMISE12 Dice under the current recipe.

The result does not falsify metadata-conditioned physical acquisition because no spacing/thickness metadata was used. It only closes metadata-free scan-level profile sharing. Do not tune the refresh interval or rename this as a protocol-conditioned result.

## Next hypothesis

Return to independent continuous SliceEqOcc sampling, but estimate the unlabeled acquisition risk conditionally on the same anatomy using a phase-antithetic pair. This directly addresses the limitation of SAQ: SAQ balanced four nodes across different anatomy samples, whereas the proposed pair evaluates `+phase` and `-phase` for the same pseudo-volume stack and the same sigma. It also retains, rather than truncates, the continuous profile tails that were lost by SAQ.

