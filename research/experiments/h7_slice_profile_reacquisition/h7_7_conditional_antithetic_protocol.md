# H7.7 Conditional-Antithetic SliceEqOcc Protocol

Status: implemented and locked before the first external run.  Date: 2026-08-13.  Experiment identity: `SliceEqOccCAP_PROMISE12`.

## Motivation

SliceEqOcc improves SliceEq from 0.832603 to 0.844566 by retaining acquisition-derived fractional occupancy. Two later variants identify where not to spend the next run:

- SAQ preserves mean severity but assigns quadrature nodes across different anatomy samples and removes continuous tails; it has no gain.
- SliceEqOccSC reduces sample-level profile diversity and reaches 0.836219, losing to SliceEqOcc on 9/10 cases.

The remaining targeted hypothesis is that phase-induced acquisition risk must be integrated **within the same anatomy**, especially for the noisy unlabeled pseudo-occupancy branch.

## Sole method change

For each unlabeled three-slice stack, keep the ordinary SliceEqOcc sample

\[
\sigma\sim U(0.45,0.85),\qquad \phi\sim U(-0.25,0.25),
\]

and construct one conditional antithetic profile using the same width and reflected phase:

\[
w^+=w(\sigma,\phi),\qquad w^-=w(\sigma,-\phi).
\]

Apply each profile to the same image stack and the same detached EMA pseudo-mask stack. The unlabeled risk is

\[
L_U^{CAP}=\tfrac12\left[
L_{occ}(f(R_{w^+}(X)),R_{w^+}(\hat Y))+
L_{occ}(f(R_{w^-}(X)),R_{w^-}(\hat Y))\right].
\]

Because the original phase distribution is symmetric, this estimator preserves the original marginal objective while cancelling sample-conditional odd phase variation. It is not a four-node quadrature rule and adds no learned module.

## Frozen parent recipe

- exact existing root path and shared Pre10000 checkpoint;
- seed 1337, 30,000 self-training updates, loader batch 24, labeled batch 12;
- first 1,000 identity updates;
- original hard labeled anchor plus exact-GT reacquired labeled occupancy;
- detached EMA argmax and 2D LCC pseudo-mask stack;
- sigma `[0.45, 0.85]`, phase `[-0.25, 0.25]`, offsets `(-1,0,1)`;
- soft CE plus squared soft Dice, identical consistency ramp, SGD, EMA, validation and inference;
- no confidence filtering, attention, boundary loss, extra head, posterior target, schedule, or test-time change.

Only the U branch is paired. The exact-GT labeled branch retains the original one-sample estimator so the experiment targets semi-supervised acquisition risk rather than increasing all augmentation indiscriminately.

## Batch and loss accounting

After warmup the one student forward contains 48 views:

- 12 original labeled center views;
- 12 labeled SliceEqOcc views;
- 12 primary unlabeled SliceEqOcc views;
- 12 phase-reflected unlabeled SliceEqOcc views.

The loader batch remains 24. The consistency coefficient is not doubled: each U view contributes half of the unchanged aggregate `L_U` weight. The change does increase student compute and changes training-time BN composition; if successful, a compute-matched control is required before attributing all gain to variance reduction.

## Required diagnostics

- primary and reflected U losses;
- exact phase-pair residual (must be zero);
- profile-weight L1 separation;
- paired image and occupancy separation;
- paired hard-target disagreement;
- inherited SliceEqOcc occupancy and clamp diagnostics.

## Prediction and stop rule

This is one exploratory run because PROMISE12 test feedback has already been used during method development.

- Retain if the locked best checkpoint exceeds 0.844566 and the improvement is not carried by only one case; then repeat seeds and run a 48-view compute-matched control.
- Borderline if Dice is 0.842-0.845: inspect validation and paired activity, but do not tune phase/sigma on the test set.
- Reject if Dice is below 0.842 or paired separation is essentially inactive. Do not add sigma antithetics, a center pseudo-anchor, confidence filtering, or another loss in the same run.

The primary scientific comparison is SliceEqOccCAP versus SliceEqOcc. SliceEqOccSC and SAQ remain documented negative results rather than components of the final method.

