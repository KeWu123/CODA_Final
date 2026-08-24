# CoDA-MT: Evidence-Coupled Data and Target Augmentation

## Two-Sentence Pitch

Weak-to-strong semi-supervised segmentation makes the student see less evidence under strong degradation but still demands the same hard pseudo-label, producing target over-certainty and confirmation bias. CoDA-MT augments the pseudo-target together with the image, increasing local target entropy in proportion to augmentation-induced evidence loss while preserving exact equivariance for geometric transforms.

## Hidden Assumption Being Negated

Conventional assumption: every label-preserving augmentation deserves an equally certain target.

Negation: semantic identity can remain unchanged while observability decreases. The class may still be the same, but the correct predictive distribution given a blurred, erased, downsampled, or noisy observation should be less concentrated.

## Minimal Method

Let the EMA teacher predict a soft pseudo-label `q = p_t(y|a_w(x))`. After geometric alignment, construct the strong target

`q_a(v) = (1 - gamma_a(v)) T_a(q)(v) + gamma_a(v) u`,

where:

- `T_a` is the exact mask-coordinate transform;
- `u` is the maximum-entropy class distribution;
- `gamma_a(v)` is the evidence-loss map in `[0,1]`.

Use standard cross-entropy/KL from `q_a` to the student's strong-view prediction. No new network and no inference-time operation are required.

## Evidence-Loss Map: Start Simple

Do not begin with a learned controller.

- Erasing/masking: `gamma` is the known opacity mask.
- Gaussian/Rician noise: `gamma` is the normalized local noise-to-signal ratio.
- Blur/downsampling: `gamma` is the normalized loss of local gradient energy between aligned weak and strong views.
- Geometric flips/rotations/crops: `gamma=0`; only warp `q`.
- Mild monotone intensity transforms: `gamma=0` unless the transform clips or saturates information.

The first version should contain one formula and no trainable augmentation policy.

## Information-Theoretic Motivation

For a degradation channel `Y -> X -> A(X)`, data processing gives `I(Y; A(X)) <= I(Y; X)`. Equivalently, conditional uncertainty cannot decrease on average after information is discarded. This motivates an expected entropy constraint. It does not prove that every individual pixel must become less confident, so claims must remain empirical and distribution-level.

## Novelty Boundary

May claim, if the search remains clean:

- spatial evidence-coupled pseudo-target augmentation for semi-supervised dense prediction;
- a transformation taxonomy separating coordinate change, appearance invariance, and evidence destruction;
- empirical identification of target over-certainty under strong augmentation.

Must not claim:

- first label smoothing;
- first uncertainty-aware semi-supervised segmentation;
- first coupling of image degradation and label smoothing in all vision tasks;
- formal per-pixel uncertainty guarantees from the data-processing inequality.

## What Would Make This CVPR-Caliber

1. A striking failure plot shows that hard-target consistency becomes increasingly miscalibrated and eventually harmful as realistic corruption severity increases.
2. One fixed coupling rule improves Dice, HD95, calibration, and robustness across PROMISE12 plus at least two additional datasets/modalities.
3. It beats hard targets, fixed label smoothing, loss masking, confidence weighting, entropy minimization, and augmentation selection.
4. The method remains architecture-agnostic and has no inference overhead.
5. Gains persist over patient-level folds and multiple label draws.

## Strongest Reviewer Objection

"This is just label smoothing with an augmentation-dependent coefficient."

Required answer: show that fixed smoothing, teacher-confidence smoothing, and global severity smoothing underperform the spatial evidence-loss map; show localized boundary/calibration effects; and demonstrate that the rule transfers without retuning across augmentation families and datasets.

