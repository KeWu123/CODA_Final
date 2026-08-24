# H4 Pilot: Does a Strong View Deserve a Hard Pseudo-Label?

## Objective

Test the central CoDA-MT premise before a full semi-supervised training campaign: information-destroying augmentations create target over-certainty, and coupling target entropy to evidence loss mitigates negative transfer.

## Stage A: Failure-Curve Diagnostic

Use only the existing `val.list` as the diagnostic set. Do not create a held-out subset or edit any list. Train or load the exact audited EMA hard-pseudo-label baseline.

For standard-deviation-scaled Gaussian noise and resolution degradation, sweep five severities. At every severity record:

- Dice and HD95 against ground truth;
- voxel accuracy/NLL/Brier/ECE;
- teacher confidence and entropy;
- weak-to-strong disagreement;
- foreground, boundary-band, and background metrics separately.

The key plot overlays prediction accuracy and target confidence. A useful failure exists if accuracy degrades materially while target confidence stays fixed or becomes increasingly miscalibrated.

## Stage B: Controlled Training Comparison

Keep all list files and order, index boundaries, sampler, optimization, schedules, and augmentation draws fixed. Compare:

1. B0: audited same-view + LCC hard-target baseline;
2. B1: weak teacher / strong student + LCC hard target;
3. B2: weak/strong + raw teacher soft target;
4. B3: weak/strong + fixed label smoothing;
5. B4: weak/strong + image-aware static/GeoLS-style smoothing;
6. B5: weak/strong + global severity-coupled target;
7. B6: spatial realized-evidence CoDA.

Run at least three seeds on the same locked split. Do not generate alternative patient splits or label draws.

## Go/No-Go Gates

Continue to full training only if all conditions hold:

- at least two realistic degradation families show a clear severity-dependent calibration gap under hard targets;
- hard-target strong augmentation becomes worse than a milder setting or no-strong baseline at high severity;
- B6 improves both Dice and calibration over B3, B4, and B5 in at least two of three seeds;
- the gain is concentrated in plausibly affected boundary or locally degraded regions, not only in background calibration;
- the same target rule transfers to a second dataset without per-dataset redesign.

## Kill Criteria

- The hard-target baseline stays best across severity sweeps.
- Improvements vanish after tuning fixed label smoothing.
- `gamma` requires a learned multi-module controller or dataset-specific thresholds.
- Calibration improves but Dice/HD95 consistently degrade.
- The effect appears only on PROMISE12 or only under unrealistic corruptions.
