# H7.3 preregistered mechanism gates

## Status and scope

Locked before running the analysis. These gates decide whether the
acquisition-residual dual-measure objective is allowed to enter a training
experiment. They do not modify or retrain SliceEqOcc.

The default evidence source is the retained seed-1337 SliceEqOcc student
checkpoint at iteration 23,000. That checkpoint contains only a student
`state_dict`; it does not contain the EMA teacher. Gate 2 is therefore a
**frozen-student proxy gate**, not direct evidence about the unavailable EMA
teacher. This limitation must remain visible in the output decision.

## Locked inputs

- Dataset: PROMISE12 training H5 root already used by SliceEqOcc.
- Split: the first 191 training slices are labeled; later slices are treated as
  unlabeled. Unlabeled H5 labels are never read by any gate statistic.
- Network: unchanged 2D U-Net, two classes.
- Acquisition operator: radius 1, offsets `(-1, 0, 1)`, sigma sampled uniformly
  from `[0.45, 0.85]`, and phase sampled uniformly from `[-0.25, 0.25]`.
- Random seed: 1337; profile streams use seeds 1338 (labeled) and 1337
  (unlabeled), matching SliceEqOcc.
- Pseudo masks: argmax followed by the same per-slice largest connected
  foreground component used by the baseline.
- Residual: `r(v) = 0.5 * ||o_phi(v) - e_z(v)||_1`; support is `r > 1e-6`.

## Gate 1 — gradient dilution

For the current full-image SliceEqOcc soft CE+Dice loss, compute the logit
gradient norm at every pixel. Report

`G_support / G_all = sum_{r(v)>1e-6} ||dL/dz(v)||_2 / sum_v ||dL/dz(v)||_2`.

The exact-GT labeled statistic is primary. The unlabeled pseudo statistic is
descriptive because it depends on frozen-model masks.

- **Pass:** exact-GT residual support contributes less than `0.20` of the total
  CE+Dice logit-gradient norm.
- **Kill:** the share is at least `0.20`; dilution is then not the demonstrated
  bottleneck.

CE-only shares are also reported for diagnosis but do not change the decision.

## Gate 2 — pseudo-residual fidelity

On labeled stacks only, create exact-GT occupancy/residual and frozen-student
LCC occupancy/residual using identical sampled profile weights. Statistics are
computed on the union of their residual supports so common zero background
cannot inflate agreement.

Report union-support pixel Pearson correlation, per-sample residual-mass
Pearson correlation, support precision/recall/IoU, and the fraction of proxy
residual mass outside exact-GT support.

- **Pass:** union-support Pearson correlation is at least `0.30`, per-sample
  mass correlation is at least `0.30`, and outside-support proxy mass is at
  most `0.50`.
- **Kill:** any of these three conditions fails.

Because this uses the saved student as proxy, a pass is explicitly
`provisional`; it cannot be reported as direct EMA-teacher validation.

## Gate 3 — acquisition specificity

Use the same residual support and the same fractional occupancy target for two
normalized soft-CE risks:

- fractional measure: weights proportional to `r(v)`;
- binary matched-support measure: weights proportional to `1[r(v)>1e-6]`.

Compare their gradients with respect to logits. The primary statistic is mean
cosine similarity on exact-GT labeled batches. The unit-gradient L2 distance
and within-support residual coefficient of variation are diagnostic outputs.

- **Pass:** mean gradient cosine is below `0.98`.
- **Kill:** cosine is at least `0.98`; the fractional acquisition measure is
  then operationally indistinguishable from binary same-support weighting at
  the preregistered tolerance.

The unlabeled proxy comparison is descriptive only.

## Joint decision

- `provisional_proceed`: all three primary gates pass. The result authorizes an
  H7.3 implementation, subject to the explicit student-proxy limitation.
- `stop_h7_3`: at least one primary gate fails. Do not rescue the idea by
  changing thresholds, widening the profile, adding confidence selection, or
  altering the locked baseline in the same experiment.

The analysis writes one JSON artifact containing arguments, file and checkpoint
hashes, sample counts, all gate metrics, thresholds, and the joint decision.
