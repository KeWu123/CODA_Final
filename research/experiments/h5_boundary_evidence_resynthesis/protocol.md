# H5 Protocol: Boundary-Manifold Evidence Resynthesis

Status: **pre-registered design; zero H5 training runs**  
Locked: 2026-08-11 (Git is unavailable in this environment, so the timestamped
research record replaces a commit hash)

## Frozen baseline contract

All method comparisons retain the current baseline's network, two stages, EMA teacher,
teacher mode, LCC hard pseudo-labels, CE+Dice objectives, consistency ramp, BatchNorm,
sampler, label count, iteration counts, and data-list order. BMER may only replace a
labeled input tensor with an augmented labeled tensor whose GT is unchanged. No soft
target, boundary loss, confidence weight, selector, new head, generator, or extra
consistency objective is permitted.

## Hypotheses

- **H5.1:** baseline errors are enriched near the boundary, and object-relative
  evidence statistics explain boundary error beyond global intensity statistics.
- **H5.2:** normal evidence profiles extracted with baseline teacher masks preserve the
  relevant ordering/shape of GT-derived profiles on oracle-labeled cases.
- **H5.3:** full boundary-manifold evidence intervention causes an ordered and spatially
  localized prediction response that scalar local perturbations cannot explain.
- **H5.4:** expanding exact-GT boundary-evidence coverage improves Dice and physical
  HD95 over the unchanged baseline and matched augmentation controls.

## Stage 0: provenance and measurement

1. Record code, list-file, shared pretraining-checkpoint, and evaluated-checkpoint
   hashes; root; complete arguments; seed; versions; and hardware.
2. Disable broad checkpoint auto-search and permissive `strict=False` fallback.
3. Use one pretraining checkpoint for every row in a comparison block.
4. Do not change teacher/BN behavior in only the proposed method. Teacher-mode and BN
   leakage checks are diagnostics, not fixes or contributions.
5. Compute correct physical-spacing HD95/ASD and explicit empty-prediction handling for
   every checkpoint, while retaining the legacy metric column for continuity.
6. Use patient-level splits, at least three seeds (preferably five on PROMISE12),
   per-case pairing, and mean plus standard deviation.

The user's reported small CoDA improvement has no local logs, checkpoints, commands, or
seeds. Record it as an exploratory observation, not as a trajectory result.

## Stage A: first kill test before training

Freeze the same baseline on held-out labeled volumes and use GT masks only for this
oracle mechanism test.

1. Predefine band widths and evidence attributes (two-sided contrast, normal slope,
   transition width, local texture); do not choose them after seeing results.
2. Rank profiles by evidence strength, render strong profiles on weak-evidence recipient
   geometry and weak profiles on strong-evidence recipient geometry, preserving the GT
   and all pixels outside the band.
3. Compare with area- and severity-matched global histogram mapping, boundary-band
   scalar contrast, and boundary-band blur. Match changed-pixel fraction and mean
   absolute perturbation; report SSIM as a secondary strength check.
4. Measure prediction change versus signed distance, normalized surface Dice, HD95,
   Dice, taper-edge gradients, outside-band identity, and simple edge-only prediction.
5. Apply the same renderer to a shape/area-matched sham contour displaced away from the
   GT. Train an augmentation detector and compare edge-to-GT alignment with the real
   unlabeled profile distribution to test whether the renderer leaks the exact label.

Required observation: the full profile intervention has an ordered, directionally
opposite effect (strong-to-weak helps; weak-to-strong harms), the prediction change is
enriched in the predeclared band, and the effect is stronger than scalar/blur controls.
Operationally, require (a) a patient-bootstrap 95% interval above zero for full-profile
minus best-simple-control boundary response, (b) predeclared profile-strength versus
response Spearman `|rho| >= 0.30` with an interval excluding zero, and (c) at least 2x
area-normalized response enrichment inside the band. These gates may be made stricter,
but not relaxed after results are observed.

**Kill H5 immediately** if any of the following holds:

- full profiles do not outperform or differ mechanistically from scalar contrast/blur;
- response is not ordered by donor evidence strength;
- most prediction change occurs outside the boundary band;
- profiles create visible halos, geometry changes, or out-of-support intensities.
- a trivial edge probe or augmentation detector can exploit a systematic GT-centered
  artifact that is absent from real scans.

## Stage A2: unlabeled measurement validity

- On held-out labeled cases, extract profiles from GT and detached LCC teacher masks.
- Measure profile L1, slope/width error, rank correlation, and repeat-pass stability.
- Stratify by teacher boundary error and apex/mid/base position.
- Compare labeled versus unlabeled evidence distributions with a preselected
  patient-level energy-distance/MMD test; compare to global median/MAD/histogram gaps.

Reject use of unlabeled evidence if pseudo-derived profiles cannot preserve GT-profile
ordering. Pre-register median GT/pseudo profile Spearman `rho >= 0.70` and repeat-pass
ICC `>= 0.75` as the minimum fidelity gate. Do not add an uncertainty network to rescue
this test.

## Stage B: isolated short-run comparison

All rows share the exact baseline training path and pretraining checkpoint.

| ID | Input-only change | Purpose |
|---|---|---|
| B0 | none | current locked baseline |
| B1 | global brightness/contrast/histogram | marginal-style control |
| B2 | full-foreground affine jitter | ARHNet-like perturbation control |
| B3 | random non-boundary band with matched area | location control |
| B4 | boundary scalar contrast/blur with matched strength | local-simple control |
| B5 | labeled-profile bank only | tests whether unlabeled coverage matters |
| B6 | unlabeled random profiles without `(s,z)` condition | conditionality ablation |
| B7 | complete BMER | proposed input operator |

Use 2,000--3,000 iterations and one fixed seed only for screening. Promote B7 only if it
beats B4 in both a region and boundary metric and the Stage-A mechanism survives. CoDA
target softening is excluded because it would violate single-factor attribution.

## Stage C: confirmatory study

Run the same operator on PROMISE12, LA, and Pancreas-NIH (or equivalently justified
binary/single-organ MRI and CT benchmarks). Use standard patient-level label regimes
and the same normalized coordinate/radius rule without per-dataset redesign. Multi-class
boundary junctions are outside the first claim and should not be improvised only to add
ACDC.

Report Dice/Jaccard, physical HD95/ASD, normalized surface Dice/boundary F-score,
error-versus-signed-distance, profile-distribution coverage, performance by boundary
contrast and anatomical position, compute cost, and zero inference overhead. Compare
with the unchanged baseline, BCP, a tuned strong-augmentation SSL baseline, a recent
mixing method, and a recent frequency/style method when official implementations allow
a fair protocol.

Required ablations: full evidence versus intensity-only/gradient-only; conditional
versus unconditioned bank; unlabeled versus labeled bank; GT versus teacher masks on
oracle cases; band/taper; 2-D independent versus z-smooth field; BMER in self-training
only versus any pretraining variant. Lock choices before final test evaluation.

## CVPR-level kill criteria

- B4 simple local jitter matches BMER within uncertainty.
- Profile-coverage change does not correlate with boundary-error change.
- Gain is smaller than seed/model-selection variation.
- Gain disappears under explicit checkpoints and physical distance metrics.
- The same operator does not transfer beyond PROMISE12.

If any holds, do not stack selectors, losses, or teacher changes. Reframe for a narrower
medical-imaging venue or abandon the direction.

## Two-week pilot

- Days 1--2: provenance wrapper and uniform evaluation audit.
- Days 3--4: canonical ribbon/profile bank, renderer, identity and seam tests.
- Day 5: Stage-A oracle intervention and go/no-go decision.
- Days 6--8: B0--B7 short screen using a shared pretrain.
- Days 9--14: only after passing, full PROMISE12 multi-seed B0/B4/B7 and an ACDC
  transfer check. Do not start with a broad hyperparameter sweep.
