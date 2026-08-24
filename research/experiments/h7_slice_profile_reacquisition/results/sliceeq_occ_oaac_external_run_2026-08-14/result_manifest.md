# SliceEqOcc-OAAC external result manifest

## Supplied artifacts

- external training log supplied out of tree
  - SHA-256: `0825e7f29983730980965582a7eb160379c4e70de23f390a1ab09955d4f8f527`
  - 618 lines; complete 30k run
- external performance report supplied out of tree
  - SHA-256: `8f4c9687290b7be15a346030204e1d03700f51c9b363ed4928fe54c95842bc23`
  - evaluates `iter_27000.pth`, not `unet_best_model.pth`

Raw medical data, checkpoints, and external logs are intentionally not copied
into the repository.

## Run identity

- experiment: `SliceEqOccOAAC_PROMISE12`
- seed: 1337
- loader batch: 24 (12 L + 12 U)
- effective post-warmup student batch: 36
- maximum iterations: 30000
- shared pretrain SHA-256:
  `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- profile: sigma `[0.45,0.85]`, phase `[-0.25,0.25]`
- appearance: log-gamma `[-0.20,0.20]`, log-contrast
  `[-0.15,0.15]`, brightness-span `[-0.10,0.10]`, seed 1339

The terminal call contract is complete: paired acquisitions 58000 and
diagnostics 58000, equal to `2*(30000-1000)`.

## Validation trajectory

- best validation Dice: `0.834863 @ iter23800`
- validation at iter27000: `0.828406`
- final validation Dice: `0.831964 @ iter30000`
- best-to-final change: `-0.002899`

## Supplied test-selected oracle checkpoint

The attached performance file tests `iter_27000.pth`. The user subsequently
clarified that multiple checkpoints had been evaluated on this same 10-case
test split and 27k was retained because it produced the highest observed test
Dice. The inspected checkpoint set and search count were not supplied, so this
is a post-hoc test-selected development maximum rather than a periodic or
validation-selected result.

- Dice: `0.849538`
- Jaccard: `0.740985`
- HD95: `3.554760` (legacy voxel-index distance)
- ASD: `1.868299` (legacy voxel-index distance)

Against the accepted SliceEqOcc development oracle (`0.844566` Dice), the
apparent oracle-to-oracle Dice delta is `+0.004972`; OAAC wins 7/10 cases and
has median paired delta `+0.0026095`. These are descriptive statistics
conditional on the selected checkpoints. The checkpoint-search counts may
differ, so the comparison cannot establish an unbiased OAAC gain.

## Mechanism activity

Across 146 logged appearance diagnostics:

- active sample fraction: `1.000000` in every record;
- mean `|log-gamma|`: `0.099566`;
- mean `|log-contrast|`: `0.076157`;
- mean `|brightness/span|`: `0.050322`;
- mean normalized absolute image change: `0.055178`.

The intervention is materially active and does not show the near-identity
failure observed in ADU or SCPO.

## Evidence classification

`protocol-deviating development result; validation proxy positive; test-selected oracle; unbiased test unavailable`

The locked validation proxy `0.820373` is passed by `0.834863@23800`. The
second-stage test rule was not followed because the iteration was selected from
multiple test evaluations. Testing `unet_best_model.pth` may complete the
development record, but it cannot make this repeatedly queried test untouched
again. A primary paper result requires a newly untouched/hidden evaluation or
a preregistered outer/nested validation protocol.
