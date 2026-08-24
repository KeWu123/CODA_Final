# BMER Semantic Rigor Review

Date: 2026-08-11  
Status: **manual six-dimension review; not an ARA Seal**  
Reason: the repository is an autoresearch workspace, not a Level-1-validated ARA with
the required `PAPER.md/logic/trace/evidence` structure.

## Overall

**Provisional Accept as a research plan, not as a scientific result (mean 3.83/5).**

The direction is well scoped, records the negative/pivot history honestly, and has an
unusually strong pre-training kill test. Its main weakness is fundamental and explicit:
there is no experimental evidence that the boundary-profile mechanism exists, that a
pseudo-mask can measure it, or that the renderer avoids label leakage.

## Dimension scores

| Dimension | Score | Assessment |
|---|---:|---|
| D1 Evidence relevance | 3 | Code evidence supports the CoDA/baseline diagnosis and primary sources constrain novelty, but no data supports BMER's mechanism or improvement claim. |
| D2 Falsifiability | 4 | Oracle directionality, matched controls, profile-fidelity, leakage, transfer, and explicit kill criteria are actionable; quantitative thresholds are now pre-registered. |
| D3 Scope calibration | 4 | Contributions are conditional and the first claim is limited to coherent binary/single-organ boundaries; no guaranteed-CVPR or first-ever claim is made. |
| D4 Argument coherence | 4 | The arc from CoDA failure analysis to a conditional boundary-evidence gap and an input-only operator is coherent; the key premise is correctly treated as a gate rather than a fact. |
| D5 Exploration integrity | 4 | BCP-like structural mimicry was explicitly rejected after user correction; CoDA, orbit sets, frequency/style, 3-D trajectory, acquisition simulation, and slab re-acquisition are documented as alternatives or dead ends. |
| D6 Methodological rigor | 4 | Frozen baseline, shared pretrain, matched controls, multi-seed/patient-level analysis, physical boundary metrics, leakage probes, and cross-modality transfer are specified; exact renderer hyperparameters still need locking before implementation. |

## Severity-ranked findings

### F01 — Major — no mechanism evidence yet

Evidence span: “Status: **primary pivot candidate; not yet experimentally validated**.”

The paper-level contribution depends on an unobserved phenomenon: full normal profiles
must control prediction beyond scalar local contrast. The Stage-A oracle intervention is
therefore a prerequisite, not an optional analysis. Do not implement full training or
write outcome language until this gate passes.

### F02 — Major — mask-conditioned label leakage

Evidence span: “resynthesizes that evidence on labeled anatomy while leaving the
geometry and hard ground truth unchanged.”

Any transformation centered exactly on GT can create a synthetic contour cue. The added
sham-contour, edge-only, augmentation-detection, empirical-support, and taper tests are
mandatory. Also report whether a non-deep edge detector becomes artificially better at
recovering GT on augmented images.

### F03 — Major — pseudo-boundary measurement bias

Evidence span: “use the baseline model's detached LCC masks on all unlabeled images to
construct an empirical bank.”

The bank can encode teacher errors as “real” evidence. Enforce the pre-registered
GT/pseudo rank-correlation and repeat-stability gates on oracle cases. If they fail,
reject unlabeled-bank BMER rather than adding a reliability module.

### F04 — Minor — exact renderer still needs an implementation lock

The conceptual operator is closed at the scalar-field level, but independent
reproduction still needs fixed definitions for signed-distance sign, ribbon radius,
normal sampling/interpolation, robust normalization, endpoint/taper function, profile
conditioning bins, smoothing along `s/z`, bank update policy, and RNG. Lock these before
the first result; do not choose them from validation performance.

### F05 — Minor — threshold justification

The `rho`, ICC, bootstrap, and enrichment gates are operational but conventional rather
than theory-derived. Keep them fixed, report sensitivity around them, and present them
as engineering go/no-go rules rather than universal constants.

### F06 — Minor — short runs are screening only

The 2k--3k one-seed chain can reject a direction but cannot support a positive claim.
Only the full multi-seed runs with explicit checkpoint identity and paired case-level
statistics count as confirmatory evidence.

## Decision

Proceed only to the Stage-A oracle/profile-fidelity tests. The artifact is rigorous
enough to authorize that pilot, not yet a BMER implementation campaign or a CVPR claim.

