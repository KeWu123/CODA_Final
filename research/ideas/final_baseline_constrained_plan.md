# Final Baseline-Constrained Plan: CoDA-MT

## Working Title

*Do Strong Views Deserve Hard Labels? Corruption-Coupled Dense Targets for Semi-Supervised Medical Image Segmentation*

## Central Observation

The audited baseline converts every EMA teacher prediction into an argmax + largest-component hard mask, and the teacher and student see the same unlabeled image. Introducing a genuinely stronger student view without changing the target would ask a degraded observation to satisfy an equally certain one-hot constraint.

## One-Sentence Method

**Every sampled evidence-destroying augmentation returns both a strong image and a spatial evidence-loss map; the latter controls how much the teacher pseudo-target is softened at the same pixels.**

## Frozen Components

The following stay bit-for-bit or semantically identical: all data lists and order, labeled/unlabeled indices, batch sampler, labeled fraction, supervised pretraining, U-Net, seed, patch size, optimizer, LR, iteration counts, validation frequency, EMA decay, consistency ramp, and test case list.

## Minimal Training Change

For the unlabeled tensor `x_u` returned by the current loader:

1. Teacher weak view: `x_w = x_u`.
2. Teacher probability: `q = softmax(f_ema(x_w))`.
3. Preserve topology prior: obtain the current LCC mask `M` from `argmax(q)` and suppress disconnected foreground probability outside `M`, then renormalize.
4. Sample an intensity-only evidence-degrading augmentation `A`, returning `(x_s, gamma)`.
5. Student input: replace only the unlabeled tail of its batch by `x_s`; labeled tensors remain untouched.
6. Dense target:

   `q_A(v) = (1 - gamma(v)) q_M(v) + gamma(v) / C`.

7. Replace only the unlabeled hard CE/Dice with soft cross-entropy plus soft Dice. Supervised loss and total consistency weight remain unchanged.

No second network, feature bank, generator, cross-patient mixing, or inference module is added.

## Augmentation Bank for Version 1

Use only two families whose severity can be measured and which do not require coordinate warping:

### Resolution degradation

- downsample then upsample the MRI slice;
- `gamma(v)` is derived from normalized local gradient-energy loss between `x_w` and `x_s`;
- motivation: simulates loss of boundary evidence and partial-volume effects.

### Noise degradation

- zero-mean Gaussian noise scaled by each slice's standard deviation; use Rician noise only if the stored intensities are verified as magnitude-MRI and non-negative;
- `gamma(v)` is a bounded local noise-to-signal ratio using local variance of `x_w`;
- motivation: scanner/SNR variation without cross-patient synthesis.

Do not add CutMix, Copy-Paste, Fourier style exchange, learned policies, or boundary-specific erasing in the main method. They either collide with prior work or introduce artificial anatomy.

## Why It Is Not Merely Label Smoothing

- Fixed label smoothing uses one coefficient unrelated to the sampled view.
- GeoLS uses static image/geodesic context around ground-truth boundaries.
- ESL derives soft targets from predictive class ambiguity.
- Supervised Mollification couples global degradation and label entropy for image classification.
- CoDA uses the **realized, spatially varying information loss of the exact strong view** to transform an unlabeled dense pseudo-target.

This distinction must be demonstrated, not asserted.

## Mandatory Ablation Chain

All variants use the exact same split, seed set, augmentations, sampler, and training budget.

| ID | Variant | Purpose |
|---|---|---|
| B0 | Audited baseline: same view + LCC hard target | Exact reproduction |
| B1 | Weak teacher / strong student + LCC hard target | Isolate strong augmentation |
| B2 | Weak/strong + raw teacher soft target | Isolate removal of argmax |
| B3 | Weak/strong + fixed label smoothing | Strongest simple objection |
| B4 | Weak/strong + image-aware static/GeoLS-style smoothing | Spatial smoothing objection |
| B5 | Weak/strong + global severity-coupled target | Classification mollification transfer |
| B6 | CoDA spatial realized-evidence coupling | Proposed method |

ViSA-style view selection is optional and should not enter the main method unless B6 fails.

## Primary Metrics Under the Existing Evaluation

- retain current Dice, Jaccard, HD95, and ASD exactly for comparability;
- label HD95/ASD as voxel distances unless real spacing is restored outside the locked comparison;
- add calibration metrics on the unchanged validation/test cases: foreground NLL, Brier score, ECE, and boundary-band ECE;
- report three or more seeds on the same split; this does not alter the split.

## Required Figure 1

On the unchanged validation cases, plot corruption severity against:

- Dice/foreground accuracy;
- mean foreground confidence;
- NLL/ECE;
- the gap between prediction correctness and pseudo-target certainty.

The paper story requires a visible regime where B1 hard targets become over-certain or harmful while B5/B6 remain stable.

## Go/No-Go Decision

Proceed with CoDA as the paper headline only if:

1. B1 exposes a reproducible target-overcertainty/negative-transfer regime;
2. B6 beats B3, B4, and B5 in at least two degradation families;
3. the Dice/HD95 gain is accompanied by improved calibration and boundary quality;
4. the same `gamma` construction transfers without redesign to at least one additional medical segmentation benchmark for a CVPR submission.

If B3 or B5 matches B6, simplify the contribution to global corruption-coupled pseudo-label calibration. If B1 never becomes harmful, reject H4 and return to pseudo-label quality rather than adding modules.

## Provisional Contributions

1. Identify and measure augmentation-induced target overcertainty in semi-supervised dense prediction.
2. Introduce a paired augmentation API that returns both a strong view and its spatial evidence-loss field.
3. Construct corruption-coupled dense pseudo-targets in an EMA self-training framework without additional parameters or inference cost.
4. Provide mechanistic ablations separating hard labels, generic soft labels, image-aware smoothing, global severity coupling, and spatial realized-evidence coupling.

