# SliceEqOcc-OAAC scale1.50 external result manifest

## Supplied artifacts

- external training log: `Z:/Downloads/log.txt`
  - SHA-256: `86e93db1cfe6cec52b5c14b3bdc7dbf9237321c4679e3971c324a7901d6a6aa0`
  - complete 30k run with 150 validation records
- external performance report: `Z:/Downloads/performance.txt`
  - SHA-256: `c2d2ef1660e2ad369f4ea0549ab2a3cc4579b6978e1c1ea7d43268f1f7ab8d12`
  - evaluates `unet_best_model.pth`

Raw logs, checkpoints and medical images are not copied into the repository.

## Run identity

- experiment: `SliceEqOccOAACScale150_PROMISE12`
- seed: 1337
- shared pretrain SHA-256: `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- OAAC scale: `1.50`
- log-gamma/contrast/brightness bounds: `0.30/0.225/0.15`
- probability: `1.0`
- SliceEq profile, optimizer, EMA, ramp, batch, loss, validation and inference:
  unchanged
- periodic raw-student archive: every 1000 iterations

CUDA smoke passes and paired/diagnostic terminal counts are both 58000.

## Validation

- best validation Dice: `0.835796 @ iter29400`
- final validation Dice: `0.831386 @ iter30000`
- 27k--30k mean/SD: `0.828721/0.005638`

The locked scale1.25 comparator is `0.836475`. Scale1.50 is lower by
`0.000679` and therefore fails the predeclared replacement rule.

## Validation-best test

- Dice: `0.852059`
- Jaccard: `0.745145`
- HD95: `3.294424` (legacy voxel-index distance)
- ASD: `1.553528` (legacy voxel-index distance)

Versus scale1.25, Dice changes by only `+0.000099`; Jaccard decreases by
`0.000202`, HD95 worsens by `0.065560`, and ASD worsens by `0.246465`.
Scale1.50 wins only 2/10 cases and the median Dice difference is `-0.002196`.

## Activity

Across 146 appearance records, activity remains 1.0. Mean absolute
log-gamma/log-contrast/brightness are `0.149349/0.114236/0.075483`; mean
normalized absolute change is `0.082723`. The stronger intervention executes
as intended, so the neutral result is not an implementation inactivity issue.

## Decision

`negative/neutral outer bracket; retain scale1.25; close OAAC severity search`
