# SliceEq optimization: acquisition-residual dual-measure risk

## Status

Exploratory H7.3 candidate after the SliceEqOcc seed-1337 run. This document is an
outer-loop design decision. The three mechanism gates are now locked separately in
`research/experiments/h7_slice_profile_reacquisition/h7_3_gate_protocol.md` and
implemented as a read-only analysis in `code/analyze_sliceeq_gates.py`.

## Observed bottleneck

SliceEqOcc proves that profile-weighted occupancy is active, but the active support is sparse:
the post-warmup fractional-pixel fraction is 0.008214 for exact-GT labeled views and 0.008797
for teacher-derived unlabeled views. Mean total-variation deviation from the center target is
only 0.001628 and 0.001821 respectively. The current soft cross-entropy averages over every
pixel and the current soft Dice aggregates over the complete image. Consequently, the novel
acquisition-induced target signal competes with more than 99% consensus pixels, including a
large easy background that is already supervised by the original labeled anchor.

This is a more direct optimization target than widening sigma/radius. Widening the profile
would manufacture more changed pixels without showing that they match an acquisition process.

## Candidate formulation

For central target one-hot vector `e_z(v)` and profile occupancy `o_phi(v)`, define the detached
acquisition residual

`r_phi(v) = 0.5 * ||o_phi(v) - e_z(v)||_1`.

The current loss integrates over the ordinary image measure. Add a second normalized measure
supported only where the sampled acquisition actually changes tissue composition:

`mu_phi(v) = r_phi(v) / (sum_u r_phi(u) + epsilon)`.

Then use a parameter-light dual-measure objective

`L_eq = 0.5 * L_full(p, o_phi) + 0.5 * L_residual(p, o_phi; mu_phi)`.

`L_residual` retains the fractional occupancy target; it is not a hard boundary target,
confidence mask, uniform label smoother, or generic distance-transform loss. If a sample has
zero residual mass, its residual term is exactly zero. The original hard central labeled loss,
network, EMA, sampler, profile distribution, and inference graph remain unchanged.

## Why this may improve the existing result

The exact-GT labeled re-acquisition makes `r_phi` trustworthy. On unlabeled data, `r_phi` is
detached and comes from the same teacher-mask stack already used by SliceEqOcc. The additional
term changes the integration measure, not the target. It therefore increases the optimization
share of the physically mixed interface without injecting foreground mass into unaffected
background, which was CoDA's central failure mode.

The expected effect is better HD95/ASD and preservation of the 23k generalization regime. A
large Dice gain is possible but is not assumed.

## Mandatory diagnostics before a full run

1. On the retained 23k checkpoint, compute the CE gradient norm contributed by residual-support
   pixels versus consensus pixels, separately for exact-GT labeled and teacher-derived
   unlabeled views.
2. On labeled cases, replace GT stacks with frozen teacher stacks and measure whether teacher
   residual support overlaps GT residual support. Report precision, recall, and residual-mass
   correlation. This distinguishes partial volume from cross-slice pseudo-label flicker.
3. Compare the proposed residual measure with a same-support binary boundary weight. If both
   are equivalent, the acquisition-specific claim is not supported.

## Kill conditions

- The current residual support already contributes at least 20% of the re-acquired gradient;
  dilution is then not the limiting mechanism.
- Teacher residual-mass correlation with exact GT is below 0.3 or most teacher residual lies
  outside the GT acquisition support.
- A binary boundary reweighting produces the same gradients/effect as fractional residual
  weighting.

## Lower-priority optimization axes

- Calibrate profile kernels and support from physical slice spacing/thickness metadata. This is
  scientifically valuable but unavailable from the current slice H5 contract and must not be
  approximated by arbitrary wider blur.
- Stabilize the late unlabeled pressure. The inherited consistency coefficient grows from
  about 0.379 at 23k to 0.5 at 30k, matching the observed loss of test generalization, but a
  post-hoc cap would be schedule tuning rather than a method contribution.
- Teacher eval-mode, BN isolation, cosine LR, checkpoint averaging, and stronger backbones may
  improve numbers, but they change shared infrastructure and cannot serve as SliceEq novelty.
