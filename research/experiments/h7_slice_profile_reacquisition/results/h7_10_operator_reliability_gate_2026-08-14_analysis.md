# H7.10 operator-reliability gate result

## Provenance and scope

- Source artifact: `h7_10_operator_reliability_gate_2026-08-14.json`
- Artifact SHA-256: `b23ca1cf3621569dec3b5256d96046225c9621ad3050061b2f21083f39509670`
- Post-run audit script: `code/reaggregate_h7_10_result.py`
- Derived audit artifact: `h7_10_strict_reaggregation_2026-08-14.json`
- Created: 2026-08-14 03:14:47 UTC
- Runtime: PyTorch 2.11.0+cu128
- Checkpoints: parent SliceEqOcc students at 18k, 24k, and 30k
- Checkpoint role: `student_as_proxy_teacher`; historical EMA states were not saved
- Data: 191 labeled training slices from seven patients, two fixed case-mixed batch schedules
- Access audit: zero unlabeled-label, validation-label, and test-label reads
- Stochastic contract: eight draws and four disjoint ADU pairs per stack

The uploaded result was produced by analyzer SHA-256
`ea894665fa75089d1e062d56d82f616d556460fc13689950516d87ce9fdf3f65`.
The local analyzer was subsequently hardened for support-pixel pooling,
deterministic CuDNN, exact checkpoint naming, and stricter cross-checkpoint
quality accounting. The raw uploaded ADU patient/pair records were therefore
conservatively reaggregated under the stricter quality rule. Support-pixel
pooling and deterministic execution cannot be reconstructed from the summary
artifact, so this result is an **exploratory-positive mechanism screen**, not
a final-hash confirmatory run. It authorizes one exploratory full training;
paper-level confirmation requires rerunning the final frozen analyzer.

## Checkpoint-level result

| Step | SCT patients | SCT residual variance reduction | SCT residual Brier reduction | SCT full Brier ratio | ADU patients | ADU JS/error rho | ADU top-20 ratio | ADU weighted Brier reduction | Fractional weight retained |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18k | 0/7 | 7.49% | -1.99% | 1.0263 | 7/7 | 0.6745 | 2.695 | 7.34% | 0.9634 |
| 24k | 0/7 | 7.99% | -1.62% | 1.0170 | 7/7 | 0.6765 | 2.460 | 7.04% | 0.9642 |
| 30k | 0/7 | 5.75% | -1.40% | 1.0188 | 5/7 | 0.6133 | 2.257 | 5.26% | 0.9655 |

SCT decreases the acquisition-residual stochastic variance slightly, but the
shared stochastic state consistently moves the pseudo occupancy farther from
the exact target. Its Brier reduction is negative at every checkpoint, its
full-image Brier is worse, and no patient passes. SCT is rejected and must not
be rescued with post-hoc variants in this project. This rejects the tested
elementwise stack-shared SCT construction; it does not prove every possible
shared-stochasticity design false.

ADU passes all three checkpoints. JS/error association remains strong at the
late 30k model, high-JS pixels carry more than twice the remaining-region
error, and normalized reliability weighting reduces effective risk on the
same `q_bar` error map by more than the locked 5% threshold. This is evidence
that JS ranks unreliable regions, not that `q_bar` itself has lower
unweighted Brier error. The exact fractional support retains about 96% mean
weight, while normalized effective sample size remains about 0.994. Thus the
ranking gain is not obtained by discarding modeled acquisition-derived
partial volume or most pixels.

## Strict named-patient reaggregation

All four pairs are finite and non-degenerate at all three checkpoints for all
seven patients. Applying the hardened rule gives:

| Patient | Quality-complete checkpoints | Median rho | Median top-20 ratio | Median weighted-Brier reduction | Median fractional weight | Pass |
|---|---:|---:|---:|---:|---:|---|
| Case04 | 3/3 | 0.6872 | 3.133 | 7.53% | 0.9486 | yes |
| Case08 | 3/3 | 0.6372 | 2.236 | 5.19% | 0.9631 | yes |
| Case15 | 3/3 | 0.6554 | 2.004 | 7.35% | 0.9698 | yes |
| Case23 | 3/3 | 0.6952 | 2.893 | 7.47% | 0.9581 | yes |
| Case25 | 3/3 | 0.6745 | 2.695 | 6.32% | 0.9634 | yes |
| Case35 | 3/3 | 0.6615 | 2.075 | 6.26% | 0.9684 | yes |
| Case48 | 3/3 | 0.6765 | 2.460 | 7.04% | 0.9663 | yes |

The stricter cross-checkpoint decision is therefore ADU 7/7 and SCT 0/7.
This is a case-agnostic gate rule with consistent evidence on seven labeled
training subjects: patient identity is used only to avoid pseudo-replication
in the gate, never as a deployed input, weight, threshold, or sampling rule.

## Decision

`authorize_slice_eq_occ_adu_training`

The next full run is one isolated SliceEqOcc-ADU experiment. It retains the
parent student batch36, full occupancy coefficient, train-mode EMA, profile,
sampler, optimizer, ramp, 30k length, validation metric, checkpoint rule, and
2-D inference. It adds one state-isolated stochastic teacher pass, projects
both hard-LCC pseudo stacks with the same profile, forms their mean occupancy,
and continuously weights the existing U soft CE and squared-Dice with
`1-JS/log(2)`. There are no case rules, thresholds, new parameters, or
inference modules.

The gate is exploratory mechanism evidence, not a promised Dice improvement.
ADU also couples two effects: the two-pass mean target `q_bar` and JS
reliability weighting. Only the validation-selected full run can determine
whether the package improves segmentation; a positive result then requires a
compute-matched `q_bar`-only (`w=1`) control before the gain can be attributed
specifically to reliability weighting.
