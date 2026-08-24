# SliceEqOcc-OAAC-Strong-ARCP external result manifest

## Supplied artifacts

- training log: `Z:/Downloads/log.txt`
  - SHA-256: `a5e99b3ad2beff2e0ae0c8d74b01891949c429bcb1c33caf4f2b561aa3717b90`
  - 787 lines, 150 validation records, complete through 30000 iterations
- performance report: `Z:/Downloads/performance.txt`
  - SHA-256: `b58cdb9f3be7d2960bed1c73e9e6ed6d6e731e2f90fa225cc1f6d4f4e04fce43`
  - evaluates `unet_best_model.pth`

Raw logs, medical images and checkpoints are not copied into the repository.

## Run identity and integrity

- experiment: `SliceEqOccOAACStrongARCP_PROMISE12`
- seed: 1337
- shared pretrain SHA-256: `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- training length: 30000 iterations
- loader/student batch: 24/36
- SliceEq profile: sigma `[0.45,0.85]`, phase `[-0.25,0.25]`
- OAAC: Strong scale 1.25, application probability 1.0
- ARCP image-only reference: 35 training patients and 870 interior stacks
- periodic checkpoints: every 1000 iterations, 30 total
- CUDA smoke: both OAAC and ARCP passed
- terminal paired/diagnostic call counts: 58000/58000

No NaN, premature termination, batch/view drift, or periodic-checkpoint gap was found. The external zero-training gate JSON was not supplied with this result, so this manifest cannot independently recheck the gate metrics; it only confirms that the full run passed its startup CUDA contracts.

## Validation trajectory

- best validation Dice: `0.838425 @ iter29800`
- final validation Dice: `0.830195 @ iter30000`
- best-to-final change: `-0.008230`
- 27k--30k mean/SD: `0.828529 / 0.006839`
- change from final Strong parent best `0.836475`: `+0.001950`
- preregistered pass line: `0.839475`
- shortfall from pass line: `-0.001050`

The validation proxy is weakly positive, but it does not meet the preregistered `+0.003` material-effect margin.

## Validation-selected test result

- Dice: `0.851062`
- Jaccard: `0.743164`
- HD95: `6.217004` (legacy voxel-index distance)
- ASD: `2.123881` (legacy voxel-index distance)

Relative to final SliceEqOcc-OAAC-Strong:

- Dice: `-0.000898`
- Jaccard: `-0.002183`
- HD95: `+2.988140` (worse)
- ASD: `+0.816818` (worse)

The HD95 increase is dominated by `Case16 = 33.015148`. Excluding that case, the remaining nine-case ARCP HD95 is approximately `3.239433`. This localizes most of the mean HD95 failure, but Dice and Jaccard still do not exceed the parent.

## ARCP intervention activity

Means over 146 post-warmup diagnostics:

- alpha: `1.028110`
- `|alpha-1|`: `0.132262`
- active fraction: `0.694635`
- lower/upper center-weight bound hit: `0.004566 / 0.160388`
- parent/calibrated center weight: `0.621927 / 0.616106`
- mean absolute center shift: `0.014669`
- effect before/after/reference: `0.159782 / 0.157336 / 0.166532`
- batch-level absolute reference-mismatch reduction: about `5.95%`
- fraction of diagnostic records whose calibrated effect is closer to reference: `54.79%`

ARCP is not a near-identity intervention and its average center-weight shift stays below the protocol limit of 0.03. However, the observed reference-effect calibration is weak and does not consistently move each diagnostic batch closer to the reference.

## Decision

Evidence class: `single-seed validation-selected development result; weak validation positive but preregistered failure and no test replacement`.

ARCP is not promoted into the final method. `SliceEqOcc-OAAC-Strong` remains selected. Post-hoc rescue of alpha, epsilon, center support, reference aggregation, or the profile grid is closed. ARCP may be reported as a neutral/negative appendix design showing that H5-only axial-response equalization did not reliably improve final segmentation.
