# H7.4 posterior-commutation fidelity gate

## Status

Locked after H7.3 failed its preregistered dilution gate and before implementing
or running H7.4. This is a zero-training target-fidelity test, not authorization
for a full training run.

## Hypothesis

The current hard/LCC pseudo-mask stack recovers acquisition-change location and
per-sample amount but weakly recovers local fractional magnitude. Applying the
same non-invertible slice-profile operator directly to teacher posteriors may
produce a more faithful occupancy target:

`f_student(A_h X) ~= A_h f_teacher(X)`.

The proposal changes target construction, not spatial loss weighting. H7.3
residual reweighting remains prohibited.

## Locked evidence source

- Same seed-1337 SliceEqOcc iteration-23,000 checkpoint and SHA-256 recorded in
  the H7.3 result.
- Same 191 labeled stacks, deterministic ordering, radius 1, sigma
  `[0.45,0.85]`, phase `[-0.25,0.25]`, and profile draws as the H7.3 labeled
  stream.
- Frozen student in evaluation mode is again an explicit proxy because the
  checkpoint contains no EMA state.
- Exact target: `O_gt = A_h(one_hot(Y))`.

## Target variants

1. **Hard-LCC reference:**
   `O_hard = A_h(one_hot(LCC(argmax(q))))`.
2. **Raw posterior commutation:**
   `O_raw = A_h(q)`.
3. **Topology-gated posterior commutation:** for foreground LCC mask `m`, set
   `q_gate_fg = m * q_fg`, `q_gate_bg = 1 - q_gate_fg`, then
   `O_gate = A_h(q_gate)`.

No temperature fitting, confidence threshold, ensemble, morphology tuning, or
test-set choice is allowed in this gate.

## Metrics

For each variant report:

- occupancy Brier/MSE against `O_gt` on exact acquisition-residual support;
- full-image occupancy Brier/MSE;
- Pearson correlation between candidate and exact acquisition-residual
  magnitudes on their union support;
- candidate acquisition-residual mass outside exact support;
- foreground-occupancy volume bias;
- metrics separately for endpoint-clamped and non-clamped stacks.

The per-pixel occupancy Brier/MSE is locked as
`B(v) = mean_c (O_candidate(c,v) - O_gt(c,v))^2`. Full-image and
exact-support values are arithmetic means of this map over their respective
pixel sets.

The exact residual remains
`r_gt = 0.5 * ||O_gt - one_hot(Y_z)||_1`. Candidate residuals use the matching
candidate central reference before applying `A_h`: raw posterior for `O_raw`,
topology-gated posterior for `O_gate`, and LCC one-hot for `O_hard`.

## Pass/kill rule

At least one soft candidate must satisfy all of the following against the
hard-LCC reference:

1. exact-support occupancy Brier is at most `0.85x` the hard-LCC value;
2. residual-magnitude Pearson is at least `0.50`;
3. outside-exact-support residual-mass fraction is at most `0.15`;
4. full-image occupancy Brier is at most `1.05x` the hard-LCC value.

If both candidates pass, select the one with lower exact-support Brier. If
neither passes, close posterior commutation and broaden away from additional
SliceEq target/loss modifications. Thresholds must not be relaxed after the
result.

## Conditional training authorization

Only a gate pass authorizes an independent H7.4 training entry. That entry must
keep the original exact-GT labeled anchor and labeled re-acquired occupancy,
replace only the unlabeled re-acquired target construction with the selected
posterior-commuted variant, and retain the current network, optimizer, EMA,
profile distribution, schedule, batch structure, validation, and inference.
