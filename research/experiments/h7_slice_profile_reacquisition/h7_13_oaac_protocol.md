# H7.13 Ordered Acquisition-Appearance Consistency protocol

## Status

The method and ranges were locked after the neutral H7.12 result and before
implementation or training. Before training, code review added one semantic
clarification: unchanged L inputs/targets do not imply identical L activations
under a shared BatchNorm forward. No method, range, or decision rule changed.
This is the only authorized next full method experiment. It forks the original
SliceEqOcc parent and does not carry SCPO, ADU, native anchors, or additional
profile sampling forward.

## Problem and hypothesis

SliceEqOcc correctly treats through-plane re-acquisition as target-changing:
the same profile transforms image signal and tissue occupancy. Its unlabeled
student view, however, stops at that moderately changed re-acquired image and
does not use a true appearance weak-to-strong path. Recent successors targeted
rare pseudo-target exceptions and became almost identity interventions.

**Hypothesis:** after constructing the paired re-acquired image and occupancy,
applying an information-preserving, coordinate-preserving MRI appearance
operator to every unlabeled student view increases useful perturbation coverage
without corrupting fractional occupancy. The semantic order is

`teacher weak stack -> paired acquisition A_h(image, occupancy) -> appearance G_eta(image only)`.

The U objective is

`L_U = L_soft(f(G_eta(A_h(X_U))), A_h(onehot(Yhat_U)))`.

## Single method change

Only the 12 post-warmup `unlabeled_reacquired_images` are modified. For every
sample independently:

1. Normalize its finite nonconstant intensity range to `[0,1]`.
2. Draw signed log-gamma in `[-0.20,0.20]` and apply a strictly positive gamma
   exponent `exp(log_gamma)`.
3. Draw signed log-contrast in `[-0.15,0.15]` and apply positive contrast
   `exp(log_contrast)` around the transformed per-sample mean.
4. Draw an additive brightness offset uniformly in `[-0.10,0.10]` times the
   original per-sample intensity span.
5. Do not clamp, blur, downsample, mask, mix, Fourier-transform, add noise, or
   change coordinates. Constant images remain unchanged.

For nonconstant inputs this composition is monotonic and invertible for the
sampled parameters. It therefore changes appearance but not tissue occupancy.
The exact-GT original-L and re-acquired-L inputs, targets, loss definitions, and
weights remain unchanged. Their activations need not be bit-identical because
the 36 views share one student BatchNorm forward; this is part of the U-view
weak-to-strong intervention. One independent CUDA generator with seed `1339`
is used; the profile, teacher, student-dropout, and global RNG streams are not
consumed.

## Frozen parent contract

- original `train_sliceeq_occ.py`, not SCPO, is the parent;
- PROMISE12 split, seven labeled patients and first 191 labeled slices;
- seed 1337, 30k, deterministic mode, SGD, EMA 0.99 and shared Pre10000 hash
  `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`;
- loader batch24 and student batch36: 12 original-L, 12 clean re-acquired-L,
  12 appearance-strong re-acquired-U;
- one train-mode teacher forward, one student forward, one optimizer update and
  one EMA update;
- original hard/LCC teacher occupancy, profile range/RNG, complete Occ loss,
  consistency ramp, validation, checkpoint selection, and strict 2-D inference.

There is no new model, head, target, loss coefficient, confidence threshold,
extra forward, or inference operation.

## Activity and decision

Log gamma, contrast and brightness magnitudes plus mean absolute appearance
change normalized by the clean re-acquired image span. The first CUDA smoke run
must show finite tensors, deterministic independent-RNG reproduction, exact
target preservation, and nonzero normalized image change. These diagnostics do
not tune the fixed ranges.

The decision has two distinct levels fixed before training:

1. The optimization proxy passes only if best validation is `>=0.820373`
   (matched parent best: `0.817373`).
2. The user's end-to-end development target passes only if the one
   validation-selected checkpoint is strictly `>0.849` test Dice.

A validation-only pass is evidence of a better proxy trajectory, not a final
Dice improvement. A periodic checkpoint above `0.849` cannot pass either level
when validation did not select it. Test performance never selects an iteration
or authorizes a range/schedule rescue.

If negative, do not add noise, blur, CutMix, longer augmentation chains, adaptive
strength, or tune these ranges. Close appearance composition and move to the
predeclared acquisition-gradient conflict gate or the paper evidence matrix.

## Publication boundary

Weak-to-strong consistency and intensity augmentation are established by
UniMatch, AugSeg, and related work. OAAC cannot claim either as new. Its narrow
role under SliceEqOcc is the ordered factorization of two transformation
semantics: target-changing acquisition is paired with occupancy first, while a
target-invariant appearance operator is composed afterward. The main paper
contribution remains paired re-acquisition and fractional occupancy.
