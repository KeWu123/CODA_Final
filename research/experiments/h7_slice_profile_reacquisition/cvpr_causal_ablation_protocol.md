# SliceEqOcc CVPR Causal Ablation Protocol

Status: preregistered design; no run is claimed by this file.  
Date: 2026-08-13  
Primary method: SliceEqOcc.

## Objective

Determine whether SliceEqOcc's gain is caused by paired acquisition semantics and fractional occupancy, rather than extra labeled supervision, a larger student batch, neighboring-slice smoothing, BatchNorm composition, teacher stochasticity, or initialization differences.

The user-confirmed Dice 0.844566 is accepted as the development result. This protocol does not attempt to reproduce or dispute it; it defines the evidence required for a publication claim.

## Locked hypotheses

- `C1`: With compute, batch composition, initialization, and teacher behavior matched, paired image--target re-acquisition outperforms image-only re-acquisition.
- `C2`: Retaining fractional occupancy outperforms immediate hard argmax under the identical paired profile.
- `C3`: The full labeled-plus-unlabeled occupancy design outperforms either branch alone.
- `C4`: Gains are larger under through-plane acquisition shift and at apex/base than under mild in-distribution conditions.

## Core methods

| ID | Student inputs after warmup | Target semantics | Effective batch | Role |
|---|---|---|---:|---|
| B0 | 12 L center + 12 U center | hard GT / hard pseudo | 24 | historical scaffold |
| B0-36 | 12 L center + 12 matched ordinary L view + 12 U center | hard GT / hard pseudo | 36 | compute, BN, and L-view control |
| ImgOnly-36 | 12 L center + 12 re-acquired L + 12 re-acquired U | center hard GT / pseudo retained | 36 | image smoothing with target mismatch |
| SliceHard-36 | same as Full | same profile applied to masks, then argmax | 36 | paired operator without fractional target |
| SliceEqOcc-36 | same as Full | full exact/pseudo fractional occupancy | 36 | primary method |

The extra B0-36 labeled view must be chosen before runs. Prefer the same geometric transform and an identity through-plane profile so that it matches view count and spatial augmentation without adding neighbor information. A duplicate-view variant may be retained as a secondary control.

## Factorial mechanism methods

- `Occ-L-only`: fractional exact-GT re-acquired L branch; U uses the matched hard target.
- `Occ-U-only`: L re-acquired view uses the matched hard target; U uses fractional pseudo occupancy.
- `TargetOnly`: re-acquired occupancy supervises the center image; diagnostic mismatch control, not an expected strong method.
- `BlurHard`: matched Gaussian/mean through-plane image blur with central hard target.

## Public comparators

- Original BCP, including its defining bidirectional Copy-Paste.
- UniMatch or the existing project implementation, reported under the same split and backbone.
- At least one recent SSMIS method with a public implementation and compatible protocol.

Do not label B0 as BCP. Use `BCP-derived EMA pseudo-label scaffold (Copy-Paste removed)`.

## Shared training contract

For every method within a seed:

1. load the exact same supervised pretrain network and optimizer state; record SHA-256;
2. reset the self-training data, augmentation, profile, dropout, and CUDA RNGs from the same declared seed policy;
3. use the same sampler, patient/slice order policy, number of optimizer updates, validation cadence, and augmentation outside the tested factor;
4. use one common EMA teacher policy; recommended: teacher in eval mode for pseudo-label generation, with EMA/buffer updates explicitly defined;
5. match student effective batch at 36 for causal methods and report student/teacher forward FLOPs;
6. never expose unlabeled GT to the dataset object used by training;
7. record the actual optimizer learning rate after checkpoint loading;
8. store student, teacher, optimizer, scheduler, RNG, iteration, configuration, environment, and data hashes.

Changing teacher mode or BatchNorm handling requires rerunning every core method; it may not be applied only to SliceEqOcc.

## Seed and label-budget plan

Stage 1:

- five core methods × three optimization seeds = 15 self-training runs;
- fixed labeled cases and split, so this estimates optimization variance only.

Stage 2:

- extend Full and the strongest matched control to five seeds;
- run L-only and U-only for three seeds;
- repeat B0-36 versus Full at one additional labeled budget, preferably approximately 10% cases.

Stage 3:

- one untouched anisotropic MRI dataset: B0-36, ImgOnly or SliceHard, and Full for at least three seeds;
- if feasible, repeat with more than one labeled-subset draw and analyze separately from optimization seeds.

## Checkpoint rule

The test set must not choose iteration, hyperparameters, or method variants. Before running Stage 1, choose one common rule:

- a fixed training iteration derived from validation-only pilot data; or
- fixed candidate checkpoints plus an earliest one-standard-error rule using per-case validation Dice; or
- a fixed end-window weight average applied identically to all methods.

Use each run's test set once after the rule has selected the model. Existing development observations, including iteration 23k, remain development evidence and do not dictate a method-specific rule.

## Metrics

Per patient:

- Dice and Jaccard;
- HD95 and ASD with physical voxel spacing;
- normalized surface Dice with a predeclared tolerance;
- apex, mid-gland, and base Dice/surface metrics;
- prediction volume and empty-mask events.

Mechanism diagnostics:

- fractional-pixel fraction and occupancy deviation;
- performance versus slice spacing/thickness strata;
- prediction continuity along the axial trajectory;
- train-only FLOPs, wall time, peak memory, and inference FLOPs.

## Statistics

- Primary contrast: Full versus the strongest compute-matched non-paired control, selected without test access.
- Mechanism contrast: Full versus SliceHard.
- Use patient-level paired permutation or Wilcoxon tests.
- Report mean and median paired differences, patient win rate, and 95% hierarchical bootstrap confidence intervals with patient and seed levels.
- Apply Holm correction to the two core contrasts if both are treated as confirmatory.
- Slice-level p-values are prohibited.

## Decision rule

Retain the acquisition-equivariant contribution only if Full is directionally positive over both B0-36 and the nearest image/target control across at least three seeds and two evaluation domains, with the primary paired confidence interval excluding zero.

If Full ties ImgOnly or SliceHard, the result does not support the claimed semantic mechanism. If it only beats B0 but not B0-36, the gain is attributable to batch/view composition. Either outcome triggers claim narrowing rather than post-hoc hyperparameter rescue.

