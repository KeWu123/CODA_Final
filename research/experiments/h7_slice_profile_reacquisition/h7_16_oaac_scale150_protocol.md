# H7.16: OAAC 1.50x outer-bracket protocol

Status: locked before implementation/run
Date: 2026-08-17

## Motivation

OAAC scale 1.25 improves the unchanged validation best from `0.834863` to
`0.836475`, while increasing the measured appearance change by approximately
25%. With only scale 1.0 and 1.25 observed, scale 1.25 cannot yet be called a
local optimum. One stronger outer point is needed; a dense search is not.

## Single change

Relative to the original OAAC reference, set the joint scale to 1.50:

| Parameter | Reference | H7.15 1.25x | H7.16 1.50x |
|---|---:|---:|---:|
| log-gamma bound | 0.20 | 0.25 | 0.30 |
| log-contrast bound | 0.15 | 0.1875 | 0.225 |
| brightness/span bound | 0.10 | 0.125 | 0.15 |
| application probability | 1.0 | 1.0 | 1.0 |

Transform order, independent seed 1339 and absence of clipping remain fixed.

## Frozen contract

- same shared Pre10000 checkpoint and seed 1337;
- same SliceEq sigma/phase profile;
- same SGD/lr/momentum/weight decay;
- same train-mode EMA and decay;
- same consistency/ramp/warmup;
- same batch36, loss, 30k length and inference;
- same validation every 200 and strict best-model comparison;
- isolated ordinary periodic raw-student archive every 1000 iterations.

Original SliceEqOcc, OAAC and Strong-all sources remain unchanged.

## Decision

- Primary selector: unchanged best validation Dice.
- H7.16 replaces Strong-all only if it exceeds `0.836475`.
- `>=0.839475` is a material improvement; a smaller positive change is a
  development improvement but requires multi-seed confirmation.
- Evaluate only `unet_best_model.pth` once after the run.
- If H7.16 does not exceed `0.836475`, retain scale 1.25 and close OAAC-scale
  tuning. Do not add scale 1.125/1.375/1.75 or component-wise bounds post hoc.

The PROMISE12 test split remains development-only. This experiment optimizes a
method sensitivity parameter and does not create a new paper contribution.
