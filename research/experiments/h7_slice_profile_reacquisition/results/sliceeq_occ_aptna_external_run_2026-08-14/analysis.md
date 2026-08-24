# SliceEqOcc APTNA result analysis

## Outcome

APTNA does not improve the selected SliceEqOcc method. Its unchanged-rule best validation is
`0.804265@28.8k`, compared with `0.817373@30k` for the matched SliceEqOcc run. The supplied
validation-selected test report is `0.829420`, while the accepted exploratory SliceEqOcc result
is `0.844566`. The user-reported approximately `0.835` periodic-checkpoint observation is also
below SliceEqOcc and cannot be selected by the frozen validation rule.

## Trajectory

The late behavior is high-variance oscillation, not a clean monotonic overfitting curve. Across
iterations 25k--30k, validation has mean `0.788313` and population standard deviation `0.007490`;
it reaches `0.804265@28.8k` but falls to `0.772908@30k`. Relative to the matched SliceEqOcc
trajectory, the paired mean APTNA difference changes from `+0.000144` over 0.2k--5k to
`-0.001256` over 5.2k--10k, `-0.008946` over 10.2k--15k, `-0.016886` over 15.2k--20k,
`-0.022756` over 20.2k--25k, and `-0.013403` over 25.2k--30k. It wins only one of the final
25 paired validation points.

This means the transient native term changes the optimization basin during the early/middle
stage; making its coefficient nearly zero at the end does not return the weights to the parent
SliceEqOcc trajectory. A smaller coefficient or a different cutoff would be a post-hoc schedule
search, not a new mechanism.

## Mechanism

The implementation satisfies the locked contract: the successful occupancy term retains its full
coefficient; the parent 36-view main forward is unchanged; the native-U branch is separate and
does not update student BN running statistics or advance the parent CUDA RNG; the EMA teacher and
validation policy are unchanged. The negative result is therefore not explained by the DA/CAP
48-view main-batch confound or by an obvious slicing/target bug.

The remaining explanation is informational redundancy. Native-U and measurement-U are derived
from the same teacher stack; acquisition changes the categorical target only extremely sparsely,
whereas the native loss supplies a much broader duplicate semantic gradient. APTNA therefore
perturbs optimization without repairing the unresolved source of error: reliability of the
teacher-derived fractional occupancy.

## Case behavior

Against the accepted SliceEqOcc development checkpoint, APTNA changes the ten case Dice values by
`+0.011790, +0.017974, +0.008938, -0.017005, -0.031109, -0.090802, -0.009022, +0.000333,
-0.020778, -0.021777`. It wins 4/10 cases; the mean and median changes are `-0.015146` and
`-0.013014`. Case36 is the largest failure (`Dice 0.777384`, `HD95 27.883686`), but removing it
does not reverse the overall conclusion. Case49 remains relatively insensitive to this family of
small objective changes, whereas surface outliers vary substantially across other cases.

## Closed and open directions

Closed: native anchors (DA/APTNA), extra measurement views (CAP), case-shared synthetic profiles
(SC), and marginal quadrature (SAQ). The only bounded next question is whether stochastic teacher
instability is being encoded as if it were genuine acquisition occupancy. H7.10 therefore starts
with a labeled-only, zero-training operator-reliability gate; it does not authorize another full
training run in advance.

