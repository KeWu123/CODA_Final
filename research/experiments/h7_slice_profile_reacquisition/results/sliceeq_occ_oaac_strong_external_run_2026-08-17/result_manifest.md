# SliceEqOcc-OAAC Strong-all external result manifest

## Supplied artifacts

- external training log: `Z:/Downloads/log.txt`
  - SHA-256: `e5f373bc257a93e88fe6f8951094e97e60d7247d84973056fbdd8b1cac8bb178`
  - complete 30k run with 150 validation records
- external performance report: `Z:/Downloads/performance.txt`
  - SHA-256: `30b885c0678f2cdaeaa22da6f91ca73a0706bff4b77a94b7e993c63e183a0a12`
  - evaluates `unet_best_model.pth`

Raw external logs, medical images and checkpoints are not copied into the
repository.

## Run identity

- experiment: `SliceEqOccOAACStrong_PROMISE12`
- seed: 1337
- shared pretrain SHA-256: `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- loader/effective student batch: 24/36
- training length: 30000
- SliceEq profile: sigma `[0.45,0.85]`, phase `[-0.25,0.25]`
- OAAC scale: `1.25`
- log-gamma: `[-0.25,0.25]`
- log-contrast: `[-0.1875,0.1875]`
- brightness/span: `[-0.125,0.125]`
- application probability: `1.0`
- periodic raw-student archive: every 1000 iterations
- validation and best-model rule: unchanged

CUDA smoke passes and the terminal paired/diagnostic call counts are both
58000.

## Validation trajectory

- best validation Dice: `0.836475 @ iter29400`
- final validation Dice: `0.828680 @ iter30000`
- best-to-final change: `-0.007795`
- mean/SD over 27k--30k: `0.821553 / 0.012075`

The previous OAAC best validation was `0.834863`; Strong-all improves this
selector-compatible proxy by `+0.001612`. This is positive but below the
predeclared `+0.003` material-effect margin.

## Validation-selected test result

- Dice: `0.851960`
- Jaccard: `0.745347`
- HD95: `3.228864` (legacy voxel-index distance)
- ASD: `1.307063` (legacy voxel-index distance)

Relative to the previous OAAC test-selected development maximum `0.849538`,
the numerical difference is `+0.002422`. That comparison uses different
checkpoint selectors and is descriptive only. Strong-all wins 5/10 cases and
has median case Dice difference about `-0.000822`; gains are heterogeneous.

## Mechanism activity

Across 146 post-warmup appearance diagnostics:

- mean `|log-gamma|`: `0.124457`;
- mean `|log-contrast|`: `0.095196`;
- mean `|brightness/span|`: `0.062902`;
- mean normalized absolute image change: `0.068954`;
- activity fraction: `1.0`;
- mean below/above-source-range pixel fraction: `0.189584/0.000348`.

The intervention magnitude increased by approximately 25% as intended and did
not collapse to identity.

## Evidence class

`positive single-seed validation-selected development result; local scale optimum not bracketed; fresh confirmatory evaluation pending`
