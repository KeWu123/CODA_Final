# H7.15: OAAC method-parameter calibration

Status: protocol specified; no result yet
Date: 2026-08-17

## User constraint

This search may tune only SliceEq/OAAC method parameters. All inherited
baseline infrastructure is frozen:

- SGD, effective learning rate 0.01, momentum and weight decay;
- EMA decay 0.99 and teacher train-mode behavior;
- consistency coefficient/ramp and 1k identity warmup;
- shared Pre10000 network/optimizer checkpoint and seed 1337;
- loader batch24, effective student batch36 and 30k updates;
- loss definitions, validation cadence/metric, checkpoint rule and inference.

The isolated H7.15 successor archives raw student weights every 1000
iterations. Original SliceEqOcc/OAAC sources remain unchanged. This changes
storage cadence only: validation remains every 200 iterations and
`unet_best_model.pth` is still selected by the unchanged validation rule.

H7.14 weight averaging and PolyLR are therefore not the next experiment.

## Current reference

Current OAAC uses:

```text
log-gamma       [-0.20, 0.20]
log-contrast    [-0.15, 0.15]
brightness/span [-0.10, 0.10]
application probability 1.0
```

Across 146 logged batches, every sampled U image is active. The mean
normalized absolute image change is 0.055178 and the 95th percentile of the
batch mean is about 0.071082. Best validation is 0.834863. The reported test
maximum 0.849538 is test-selected and remains development-only.

## Round 1: three local OAAC candidates

Introduce one scalar `oaac_scale` multiplying all three symmetric ranges and
one Bernoulli `oaac_probability`. The transform order and relative range ratios
remain fixed.

| Config | scale | probability | Gamma | Contrast | Brightness/span | Purpose |
|---|---:|---:|---|---|---|---|
| Reference | 1.00 | 1.00 | +/-0.20 | +/-0.15 | +/-0.10 | completed OAAC |
| Mild-all | 0.75 | 1.00 | +/-0.15 | +/-0.1125 | +/-0.075 | lower per-view distortion, full coverage |
| Current-mix | 1.00 | 0.75 | +/-0.20 | +/-0.15 | +/-0.10 | identity/current mixture |
| Strong-all | 1.25 | 1.00 | +/-0.25 | +/-0.1875 | +/-0.125 | test unused augmentation headroom |

`Mild-all` and `Current-mix` have approximately matched expected aggregate
change but different coverage, making the search interpretable rather than a
flat parameter sweep. Identity selection and transform sampling must use the
independent OAAC generator and must not advance parent profile/dropout RNG.

Run the three new configurations with the same seed and all frozen baseline
settings. Rank only by the unchanged five-case validation metric. A candidate
must exceed 0.834863 to replace the current configuration; an improvement of
at least 0.003 absolute validation Dice (>=0.837863) is treated as materially
promising rather than small-set noise.

Test only the validation-selected checkpoint of the single winning config.
Do not inspect every checkpoint. The existing ten-case split is already a
development set, so even a score above 0.85 is not confirmatory evidence.

## Round 2: SliceEq profile severity, only after Round 1 is locked

Keep the winning OAAC configuration fixed. Tune one joint acquisition-severity
scalar around the current profile, not sigma and phase independently:

| Config | Sigma range | Phase range |
|---|---|---|
| Reference | [0.45, 0.85] | [-0.25, 0.25] |
| Profile-mild 0.85x | [0.3825, 0.7225] | [-0.2125, 0.2125] |
| Profile-strong 1.15x | [0.5175, 0.9775] | [-0.2875, 0.2875] |

All paired image/occupancy operations remain identical. This stage is
heuristic because the current H5 inputs do not expose trustworthy physical
slice-profile metadata. It may improve in-distribution Dice but cannot be
presented as protocol-conditioned physics.

## Stop rules

- Do not tune learning rate, EMA, consistency/ramp, batch, loss weights,
  training length, validation or test-time processing.
- Do not choose the augmentation seed or best test checkpoint.
- Do not search each gamma/contrast/brightness bound independently.
- If none of Round 1 beats current validation, retain current OAAC and move
  directly to Round 2; do not insert intermediate scales post hoc.
- If neither profile candidate beats the locked Round-1 winner, stop method
  tuning. The remaining work is multi-seed/external confirmation and causal
  ablation, not a larger grid.

## Paper interpretation

The chosen scalar is a sensitivity setting, not a new contribution. The useful
secondary analysis is whether performance follows augmentation severity or
coverage. The method contribution remains the fixed ordering:

```text
paired target-changing re-acquisition -> fractional occupancy supervision
-> coordinate-preserving target-invariant U appearance perturbation
```
