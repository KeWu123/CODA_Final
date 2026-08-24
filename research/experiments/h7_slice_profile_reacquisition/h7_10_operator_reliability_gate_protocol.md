# H7.10 acquisition-operator reliability gate

## Status

Locked on 2026-08-14 after the negative H7.9 APTNA result and mathematically
audited before implementation. This is a zero-training gate. It does not
change the current validation computation, select a test checkpoint, or
authorize a full run until a predeclared fidelity condition passes.

## Motivation

SliceEqOcc intentionally creates fractional occupancy when adjacent anatomical masks are mixed by
the same non-invertible through-plane operator as the images. This physical mixture must not be
treated as ordinary uncertain pseudo-label noise. However, the current train-mode EMA teacher also
contains dropout. The three adjacent slices are flattened into one batch and receive independent
dropout realizations before their masks are mixed. The resulting occupancy can therefore contain
both anatomical partial volume and reducible teacher stochasticity.

H7.10 asks two distinct questions with one labeled-only analysis:

1. **Stack-coherent teacher stochasticity (SCT):** does sharing each dropout realization across
   the three slices of one stack improve exact occupancy fidelity while keeping the EMA teacher in
   train mode while preserving the BN mode, batch composition, and update rule?
2. **Acquisition-aligned dropout uncertainty (ADU):** if two ordinary stochastic teacher passes are
   projected through the same acquisition operator, does their operator-space Jensen--Shannon
   disagreement predict exact occupancy error without suppressing genuine fractional occupancy?

SCT here means shared stochastic teacher state, not 3-D connected-component post-processing and
not the previously rejected scan-coherent profile sampler.

## Frozen analysis contract

- Use only the seven labeled training patients and their exact masks; do not inspect official
  validation or test labels.
- Use retained parent SliceEqOcc student checkpoints at 18k, 24k, and 30k. They do not contain
  the historical EMA state, so every result is explicitly a train-mode frozen-student proxy gate,
  not direct evidence from the unavailable training-time EMA teacher.
- Use the parent radius, profile distribution, spatial transforms, hard argmax, per-slice 2-D LCC,
  and fixed paired profile draws.
- Keep the teacher in train mode. Restore BN buffers before every paired comparison so the gate
  isolates dropout structure; do not replace the locked baseline policy with `eval()`.
- Use eight stochastic draws per stack, paired as four disjoint ADU pairs. Reuse each sampled
  profile across every draw, candidate, checkpoint, and locked batch schedule.
- Preload the 191 labeled slices once, then evaluate two fixed seed-derived case-mixed batch
  schedules. This prevents the train-mode BN result from depending on list-order batches that
  contain almost one patient. Pad the final 11-stack batch deterministically to 12 for the teacher
  forward and exclude the padding sample from every statistic.
- Aggregate pixels/draws/slices within patient first. A patient passes only when all conditions
  hold jointly; checkpoint and across-checkpoint decisions are made from the seven patient reports.
- Report full-image and acquisition-residual Brier/soft-Dice error, foreground mass error, native
  center pseudo Dice, MC occupancy variance, and first/middle/last axial index thirds. These index
  thirds are descriptive and are not asserted to be organ-defined apex/mid/base regions.

## SCT candidate and pass rule

For every teacher dropout layer, reshape the internal batch as `[B,3,C,H,W]`, draw a mask of shape
`[B,1,C,H,W]`, and broadcast it over the three stack slices. Preserve dropout probability, scale,
one 36-slice teacher forward, train-mode BN, 2-D LCC, and all profile/target operations.

SCT passes only if, relative to ordinary independent dropout:

- acquisition-residual MC variance on the fixed exact fractional support decreases by at least
  15%; raw occupancy variance is descriptive only because shared masks can add positive
  cross-slice covariance;
- acquisition-residual Brier error decreases by at least 5%;
- full-image Brier does not increase by more than 1%;
- native center pseudo Dice decreases by no more than 0.002 absolute; and
- the same conjunction holds in at least 5/7 patients and at least 2/3 checkpoints; and
- at least 5/7 named patients also pass the conjunction after taking their median across the three
  checkpoints, preventing different patients from carrying different checkpoints; and
- the non-clamped stratum does not reverse the residual-Brier improvement, so endpoint duplication
  cannot be the sole source of a pass.

Variance reduction without exact-GT fidelity improvement is a failure.

## ADU candidate and pass rule

For two ordinary stochastic teacher passes, construct the existing hard-LCC occupancies `q1` and
`q2` using the same profile. Let `q_bar=(q1+q2)/2` and define per-pixel epistemic disagreement as
`u_JS = H(q_bar) - 0.5*(H(q1)+H(q2))`. A physically fractional pixel on which both passes agree
has zero JS disagreement, even when its occupancy entropy is high. The candidate reliability is
`w=1-u_JS/log(2)` and is evaluated without training.

The primary ADU domain is the union of exact and predicted foreground occupancy. It excludes the
large common background, which would otherwise manufacture favorable correlations and top-20%
statistics. Spearman uses average ranks for ties; top-20% means the exact stable-sort top
ceil(0.2*N) pixels within each patient/pair. The report also includes normalized weight ESS.

ADU passes only if:

- patient-balanced Spearman correlation between JS and exact occupancy error is at least 0.25;
- the top 20% JS region has at least 1.5 times the error of the remaining region;
- normalized JS weighting reduces exact Brier error by at least 5%;
- mean retained weight on exact fractional-support pixels is at least 0.90; and
- all four predeclared pairs have non-degenerate disagreement, nonempty exact fractional support,
  and finite rank/reweighting statistics; and
- the same conjunction holds in at least 5/7 patients and at least 2/3 checkpoints.
- at least 5/7 named patients also pass the conjunction after taking their median across the
  quality-complete checkpoints, with all-pair quality complete at no fewer than 2/3 checkpoints.

Under squared Brier error, q_bar cannot be worse than the mean of the two
single-pass errors by convexity. That equality is asserted as an implementation
sanity check and reported as activity, but it is not an independent pass gate.

## Locked decision

- If only SCT passes, authorize one `SliceEqOcc-SCT` run. It changes only teacher dropout
  correlation across the acquisition support; student batch36, loss, EMA, validation, and
  inference remain unchanged.
- If only ADU passes, authorize one minimal `SliceEqOcc-ADU` run using `q_bar` and normalized JS
  reliability on the existing U occupancy loss. Do not add entropy thresholds or another module.
- If both pass, prefer SCT because it adds no forward pass or loss hyperparameter. ADU is selected
  instead only if its effective weighted-Brier gain exceeds SCT's residual-Brier gain by at least
  two absolute percentage points. Do not combine them in the first run.
- If neither passes, close the small-module optimization loop. Move to the causal matrix,
  multi-seed/external evidence, or a deliberately broader scaffold/backbone change.

Any authorized full run must retain the user's unchanged validation computation and checkpoint
rule. PROMISE12 test is evaluated once after that rule selects the checkpoint.
