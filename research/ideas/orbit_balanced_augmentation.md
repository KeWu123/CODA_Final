# OBA: Orbit-Balanced Augmentation

## Decision

This is the primary post-BMER direction. BMER is archived and is not eligible for
renderer correction, hyperparameter rescue, or combination with OBA.

The fixed experimental baseline remains the current BCP-derived EMA hard-pseudo-label
self-training implementation. OBA changes the student view of the unlabeled branch; it
does not redefine the network, EMA update, LCC hard pseudo-label, CE+Dice loss,
consistency ramp, split, schedule, or inference path.

## Result-driven observation

The two available runs form an informative asymmetric intervention:

- CoDA changes the unlabeled student view and the uncertainty of its training target.
  Its complete exploratory run reaches test Dice 0.819876, above the user's unpaired
  baseline range of 0.78--0.80, although four changed factors prevent attribution.
- BMER changes only the labeled student image using a donor-derived boundary renderer.
  Its incomplete run peaks early, declines, tests at 0.795949, and loses Dice on 9/10
  paired cases versus CoDA.

This does not prove which CoDA factor works. It does rule against spending the next
cycle on increasingly elaborate labeled-image synthesis. The reliable labeled branch
should act as the anchor; augmentation should challenge only pseudo-labeled examples,
and it should not introduce a systematic one-sided drift that must later be repaired by
target smoothing.

## Core hypothesis

Random weak-to-strong training normally samples one transformation parameter
`a ~ p(a)` and optimizes the pseudo-labeled loss on `T_a(x)`. With few patients, hard
pseudo-labels, and a nonlinear segmentation model, a finite sample of such one-sided
views can have a large odd-order drift around the clean image.

OBA samples a nuisance coordinate `a` and always evaluates the antithetic pair
`T_{+a}(x), T_{-a}(x)`. The unlabeled orbit risk is

`L_OBA = 1/2 [ell(f(T_{+a}(x_u)), y_hat) + ell(f(T_{-a}(x_u)), y_hat)]`,

where `y_hat` is exactly the baseline teacher's LCC hard pseudo-label. For a smooth
loss along the augmentation coordinate, the first-order terms cancel:

`1/2 [g(+a) + g(-a)] = g(0) + 1/2 a^T H_g(0) a + O(||a||^4)`.

Thus OBA retains a curvature/robustness pressure while removing the leading directional
bias of a single sampled intervention. This is a local statement, not a guarantee that
all image transformations preserve anatomy.

## Transformation space

The first implementation must use transformations with a meaningful identity and
opposite coordinate. Do not force blur, erasing, or arbitrary donor transfer into this
framework.

Candidate MRI nuisance coordinates are:

1. log-gamma contrast: `T_a(x) = normalize(x)^(exp(a))` and `T_-a`;
2. multiplicative smooth bias: `T_a(x) = x * exp(b_a)` and `T_-a = x * exp(-b_a)`;
3. zero-mean intensity affine coordinates around the per-slice normalized identity;
4. small stationary velocity fields `+v/-v`, only if the same warp is applied to the
   hard target and label preservation is demonstrated.

The pilot should start with log-gamma and smooth multiplicative bias. Additive Gaussian
noise `+eps/-eps` may be included as a mathematical control, not as the headline MRI
operator.

## What is new and what is not

OBA is not merely two strong views. UniMatch uses two independently sampled strong
views to expand perturbation coverage; it does not impose zero first moment in a
transformation coordinate or test odd-order cancellation. Diverse Co-training and CCVC
seek prediction/model diversity, not quadrature balance of a single augmentation orbit.

OBA is not adaptive severity selection. AugSeg, SAA, and iMAS adjust augmentation by
confidence, sample difficulty, or learning status. OBA can use a fixed distribution and
has no selector.

OBA is not TeachAugment. TeachAugment learns adversarial transformations that are hard
for a target model but recognizable to a teacher. OBA has no augmentation network or
min-max policy; its object is the bias/variance of the orbit-risk estimator.

OBA is not POS or gradient surgery. It does not reweight or project supervised and
unsupervised gradients after they are produced. It constructs an unbiased local
augmentation design before the ordinary baseline loss.

Antithetic sampling itself is classical and has recently been used for diffusion-model
sampling. The publishable novelty must therefore be narrower: **balanced augmentation
quadrature for hard-pseudo-label dense prediction**, together with empirical evidence
that one-sided orbit drift explains instability and that cancellation predicts the
gain. Calling the use of `+a/-a` alone novel is not defensible.

## Required mechanism tests

### Gate 0: frozen-model antithetic response (no training)

Using the same frozen pretrained model and held-out labeled cases:

1. draw `a`, compute signed logit displacement for `+a` and `-a`;
2. compare their correlation with two IID same-cost transformations;
3. measure the norm of the mean displacement, hard-mask disagreement with GT, and
   foreground/boundary/background localization;
4. repeat across five severities for each transformation family.

Go only if the antithetic pair has negative displacement correlation, a smaller mean
drift than IID views at matched average magnitude, and no larger oracle label violation.
If the pair is not actually antithetic in model response, reject OBA before training.

### Gate 1: compute-matched short screen

Reuse one pretrained checkpoint and identical data order for:

- B0: fixed baseline, no new strong view;
- B1: one random strong unlabeled view;
- B2: two IID strong unlabeled views;
- B3: OBA antithetic pair;
- B4: OBA with shuffled pairing, preserving marginal transforms but destroying the
  `+a/-a` relation.

Run 3k--5k self-training iterations for one seed. OBA advances only if B3 beats B2 and
B4, not merely B0. This is essential: otherwise any gain is due to two forwards or the
marginal augmentation bank.

### Gate 2: confirmatory study

Run at least three seeds with shared initialization policy. Report paired patient-level
Dice, Jaccard, physical HD95/ASD, normalized surface Dice, clean validation trajectories,
gradient variance, and orbit-drift diagnostics. PROMISE12 alone is insufficient for a
CVPR claim; use at least one cardiac benchmark and one 3-D organ benchmark, with the
same principle and dataset-appropriate nuisance coordinates.

## Falsification criteria

Reject or demote OBA if any of the following holds:

- `+a/-a` prediction displacements are not more negatively correlated than IID views;
- OBA does not beat the two-IID-view and shuffled-pair controls at matched compute;
- gains disappear when strong-view BatchNorm exposure is controlled symmetrically;
- only one transform family or only PROMISE12 improves;
- improvement is explained by a lower effective augmentation magnitude;
- formal multi-seed improvement is below run-to-run uncertainty.

## Candidate pool and rejection record

| Candidate | Main attraction | Main collision/risk | Decision |
|---|---|---|---|
| Orbit-balanced antithetic augmentation | One equation; fixed baseline; testable cancellation | Two-view compute; antithetic sampling is not itself new | **Primary** |
| Teacher-null adversarial augmentation | Maximally hard while teacher-stable | TeachAugment and tangent/VAT precedents | Backup only |
| Gradient-concordant augmentation | Directly responds to CoDA/BMER asymmetry | POS, gradient surgery, conflict-aware augmentation | Reject as headline |
| Maximal stable-orbit severity | Explicit label-preservation boundary | SAA, iMAS, AugSeg and stability selection | Reject as headline |
| Complementary evidence deletion | Collective rather than per-view sufficiency | UniMatch, masking consistency, co-training | Reject |
| Clean-anchor augmentation residual loss | Isolates invariance from pseudo-label fitting | Standard consistency reformulation | Reject |
| Stochastic virtual slab re-acquisition | Physical slice-profile motivation | Needs volume I/O/spacing; adjacent-slice and SynthSeg precedents | Independent backup |
| Axial-tangent interpolation | Uses real within-patient trajectory | Label mismatch at apex/base; inter-slice augmentation | Reject |
| Oblique virtual reformat | Real acquisition geometry | Equivalent to 3-D rotation/interpolation in practice | Reject |
| Empirical 3-D bias-field bank | Multi-center MRI relevance | Mature bias-field augmentation; weak novelty | Control family only |
| Magnitude k-space perturbation | Protocol realism | No raw complex data; easy to overclaim physics | Reject |
| Partial-volume-consistent resampling | Coupled image/occupancy target | Requires target change and high-resolution validation | Reject under baseline lock |
| Diffusion mask-conditioned synthesis | High diversity | MatchMask/Pseudo-SD/SemiGDA; large infrastructure | Reject |
| Boundary ambiguity distribution | Clinically motivated | AmbiSSL and fuzzy-region literature; multi-annotator data absent | Reject |
| Topology-safe shape deformation | Anatomy-aware | Dense prior art; changes target geometry | Reject |

## Two-sentence paper pitch

Weak-to-strong self-training treats one randomly augmented view as an unbiased sample
of a label-preserving orbit, but hard pseudo-label dense prediction can suffer large
finite-sample directional drift. OBA replaces one-sided perturbations with balanced
antithetic quadrature, canceling the leading odd-order drift while retaining
second-order robustness pressure, without changing the teacher, pseudo-label, network,
or inference model.

## Provisional contributions

1. Reframe strong augmentation in hard-pseudo-label segmentation as estimation of an
   augmentation-orbit risk and identify one-sided finite-sample drift as a measurable
   failure mode.
2. Introduce orbit-balanced antithetic augmentation, a transformation-space sampling
   rule that cancels odd-order local drift while preserving the original training loss.
3. Provide mechanism-first diagnostics and compute-matched controls showing when
   antithetic response, gradient variance reduction, and segmentation gain agree.
4. Demonstrate generality across multiple medical datasets and nuisance groups with no
   inference overhead.

## Reviewer-level risk assessment

The idea is cleaner than another boundary/style renderer and is less directly occupied
than adaptive selection or gradient surgery. It is not yet a CVPR paper. The likely
reviewer objection is: "This is two-view averaging plus a Taylor expansion." Only the
IID-view, shuffled-pair, severity-matched, and cross-dataset mechanism results can
answer that objection. If those controls fail, pivot rather than add modules.
