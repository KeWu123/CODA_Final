# CVPR Research Brief: Semi-Supervised PROMISE12 Data Augmentation

## Bottom Line

The proposed paper should not be "a new augmentation operation." The strongest available direction is a **hard-but-valid augmentation selector** that decomposes augmentation quality into teacher-estimated semantic validity and student-estimated learning value.

Provisional title:

> **Hard but Valid: Separating Augmentation Reliability from Learning Utility in Semi-Supervised Medical Image Segmentation**

Provisional method name: **ViSA-MT** (Validity-Informativeness Separated Augmentation in Mean Teacher).

## Two-Sentence Pitch

Semi-supervised segmentation needs strong perturbations to create learning signal, but the same perturbations can destroy anatomical evidence and turn uncertain pseudo-labels into harmful supervision. We select, for each unlabeled sample, the most challenging view for the student among views that remain transformation-consistent for a stable teacher, explicitly separating augmentation validity from learning utility.

## Why This Is More Defensible

BCP/ABD/OMF/PSC/SF-DA are cross-sample mixing methods. AugSeg and DPCL adapt augmentation using confidence or learning progress. ViSA-MT instead measures two counterfactual responses to each candidate transformation: whether the teacher remains stable and whether the student is challenged. The central claim is mechanistic and testable before full training.

## Paper Contributions

1. Formulate strong augmentation selection as constrained optimization on a validity-informativeness plane.
2. Propose a lightweight selector requiring no extra inference-time module and no cross-patient mixing.
3. Provide a transformation-aware consistency objective for invariant intensity transforms and equivariant geometric transforms.
4. Introduce an oracle audit showing when augmentation helps or corrupts pseudo-labels, plus broad patient-level evaluation across datasets.

## Required Scope for a CVPR Attempt

- PROMISE12 alone is insufficient. Add at least ACDC and LA or NIH Pancreas; preferably include one non-medical semantic segmentation benchmark to demonstrate generality.
- Compare against Mean Teacher, FixMatch/UniMatch, BCP, AugSeg, ABD, and current 2025-2026 methods whose code is available.
- Use patient-level cross-validation on PROMISE12. Never use slice-level random splits.
- Report three or more independent label draws/seeds, confidence intervals, HD95/ASD, calibration, and compute.
- Make the mechanistic H1.1 result a main figure: teacher validity score versus actual augmentation-induced target error.

## Fastest Two-Week Pilot

1. Reproduce a clean Mean Teacher on one PROMISE12 fold and one ACDC split.
2. Implement five candidate intensity transforms and compute teacher/student predictions for three candidates per unlabeled sample.
3. Run H1.1 offline on validation labels; do not train ViSA-MT yet.
4. If correlation is adequate, train four conditions: fixed strong, difficulty-only, validity-only, constrained hard-but-valid.
5. Repeat with three seeds. Continue only if the constrained selector improves both overlap and boundary metrics consistently.

## Stop Conditions

- No correlation between teacher stability and true augmentation harm.
- Gain smaller than run-to-run variance.
- Improvement disappears under a tuned UniMatch-style baseline.
- Training cost exceeds roughly 1.5-2x without a clear accuracy/calibration benefit.

## Venue Calibration

- CVPR: broad validation, clear general principle, strong current baselines, and a mechanistic or theoretical result.
- MICCAI: medical-only scope is acceptable if anatomy/acquisition analysis is deep and evaluation is rigorous.
- MedIA/TMI: favor extensive multi-center validation, clinical error analysis, and reproducibility over a short novelty pitch.
