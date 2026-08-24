# SliceEqOcc-OAAC-Strong-MPD external result manifest

## Corrected supplied artifacts

- external training log: `Z:/Downloads/log.txt`
  - SHA-256: `6742f82e18932932418e1093d22bd8b7656f83810af2154d1126ae69cf52c466`
  - 647 lines / 107679 bytes
- corrected external performance report: `Z:/Downloads/performance.txt`
  - SHA-256: `49db4ebce06faa8ca1b1e3dbad1732b0f0de2ad3d9168281e38ebe4512f6e97c`
  - 25 lines / 1207 bytes
  - evaluates `iter_29000.pth`

The user clarified that the performance report supplied previously was the
wrong file. That superseded report had SHA-256
`52ae817d792f4e201663fe7c3aca830d80954f02b93e2fd6543a2825b4983077`,
evaluated `unet_best_model.pth`, and reported Dice `0.848952`. It is retained
here only for provenance and must not be treated as the latest MPD result.

Raw logs, medical images, the MPD JSON artifact and model checkpoints are not
copied into the repository.

## Run identity and integrity

- experiment: `SliceEqOccOAACStrongMPD_PROMISE12`
- seed: `1337`
- shared Pre10000 SHA-256:
  `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- loader/effective student batch: `24/36`
- training length: `30000`
- OAAC scale: `1.25`
- validation and best-model rule: unchanged
- periodic raw-student archive: every `1000` iterations
- terminal OAAC paired/diagnostic counts: `58000/58000`

The log begins with one failed pre-design startup at 16:26 and then the fixed
startup at 16:34. No training iteration occurred in the failed startup. The
second startup completes exactly 150 validations through iteration 30000.

## Frozen MPD design

- distribution SHA-256:
  `1ab2de7d8b5f43b2fb6e1c66a0c08d23ca3bd5490ab5d77dcc4ed7998e8861c3`
- artifact SHA-256:
  `fa55f9d97a16552b809bb194c69ca8cbbd715c205b819353ab27e9629d60c2b4`
- active RFI strata: `20/21`
- structurally empty RFI stratum: `Case08:index-third-2`
- worst designed RFI: `0.47033756`
- entropy / parent entropy: `6.01794513 / 6.08904488` (`98.83%` retained)
- maximum density ratio: `1.607772`
- phase-mirror error: `0`
- designed moments `[E(b), E(b^2), E((br)^2)]`:
  `[0.38257712, 0.15254627, 0.01464702]`

The design used only the first 191 labeled-training H5 image/label slices. It
did not use U labels, validation, test, model predictions or losses. The LOPO
gate was explicitly skipped at the user's request, so evidence remains
exploratory.

## Validation trajectory

- best validation Dice: `0.836008 @ iter25800`
- validation Dice at corrected tested checkpoint: `0.828270 @ iter29000`
- corrected tested checkpoint gap from validation best: `-0.007738`
- final validation Dice: `0.827233 @ iter30000`
- best-to-final change: `-0.008775`
- 20k--30k mean/SD: `0.823111 / 0.007967`
- 24k--27k mean/SD: `0.827916 / 0.005788`
- 27k--30k mean/SD: `0.826990 / 0.003777`

Compared with final OAAC-Strong:

- best validation: `0.836008` versus `0.836475` (`-0.000467`);
- final validation: `0.827233` versus `0.828680` (`-0.001447`);
- late 27k--30k mean: `0.826990` versus `0.821553` (`+0.005437`);
- late SD: `0.003777` versus `0.012075` (substantially more stable).

Thus MPD stabilizes the late trajectory but does not improve the unchanged
validation selector.

## Corrected checkpoint-specific development test

The corrected report evaluates `iter_29000.pth`, not the validation-best
checkpoint:

- Dice: `0.854573`
- Jaccard: `0.749330`
- HD95: `3.256519` (legacy voxel-index distance)
- ASD: `1.324697` (legacy voxel-index distance)

Relative to final OAAC-Strong's validation-selected report
(`0.851960/0.745347/3.228864/1.307063`):

- Dice: `+0.002613`;
- Jaccard: `+0.003983`;
- HD95: `+0.027655` (slightly worse);
- ASD: `+0.017634` (slightly worse).

Relative to the superseded MPD validation-best report, iter29000 improves
Dice by `+0.005621`, Jaccard by `+0.008979`, HD95 by `-0.056129`, and ASD by
`-0.058726`. It wins 5/10 cases with median paired Dice change `+0.001653`;
the largest change is Case36 (`+0.032308`). This large ranking reversal while
validation falls from `0.836008` to `0.828270` demonstrates checkpoint-selector
instability rather than a validation-compatible method win.

The PROMISE12 test split has already participated in development. The
corrected iter29000 result is the highest currently observed MPD development
Dice, but it is checkpoint-specific and cannot be presented as an unbiased or
validation-selected primary result.

## Runtime mechanism activity

Across the 145 post-warmup profile diagnostics:

- L/U mean sigma: `0.657688 / 0.656377`;
- L/U mean absolute phase: `0.119579 / 0.120732`;
- L/U mean center weight: `0.614557 / 0.615073`;
- L/U fractional-support fraction: `0.008214 / 0.008981`;
- L/U hard-target-change fraction: `0.0000145 / 0.0000209`.

Across 146 OAAC diagnostics, activity is `1.0` and mean normalized image change
is `0.068972`, matching the intended unchanged Strong appearance branch.

## Decision

The user explicitly states that final internal method selection follows the
highest observed tested checkpoint and does not require validation-best
identity. Under that criterion, corrected MPD iter29000 is positive and
replaces OAAC-Strong as the final selected method (`0.854573` versus
`0.851960`). Freeze MPD and carry it to MM-WHS without further PROMISE12
checkpoint or profile-parameter tuning. Checkpoint provenance remains reported
for scientific transparency but is not an internal selection blocker.
