# H7.12 SliceEqOcc-SCPO result analysis

## Decision

SCPO is neutral/negative and the pseudo-topology branch is closed. Its best
validation Dice is `0.817824@28.8k`, only `+0.000451` above the matched parent
best and `-0.002549` below the locked success threshold `0.820373`. The
validation-selected checkpoint tests at Dice `0.842378`, below the accepted
SliceEqOcc development result `0.844566` by `0.002188`.

The user reports that a preceding periodic checkpoint reached approximately
`0.849`. This is retained as a post-hoc observation, not mixed with the supplied
validation-selected performance artifact. By the user's comparison it matches,
rather than exceeds, the corresponding parent checkpoint.

## Trajectory

The run is complete through 30k. Validation peaks at 28.8k and ends at
`0.802209`, a best-to-final drop of `0.015615`. The late curve is oscillatory,
not monotonically decreasing. The current five-case validation ranking and the
user-observed test checkpoint ranking therefore remain poorly aligned.

Against the accepted SliceEqOcc 23k development checkpoint, SCPO wins four of
ten cases and has median paired Dice delta `-0.001423`. Mean Jaccard changes by
`-0.003387`, HD95 by `+0.786360`, and ASD by `+0.143851`.

## Why SCPO did not produce a stable gain

After warmup, SCPO changes only `0.004527%` of unlabeled slab pixels on average.
Removed and added fractions are `0.002275%` and `0.002255%`, the foreground-mass
ratio is `0.999904`, and only `10.17%` of sampled slabs show any change. Over
24k--30k the changed fraction falls to about `0.003294%` and active slabs to
about `6.18%`. The recorded batch at the best-validation iteration has no SCPO
activity.

Thus the current teacher masks are already almost always compatible with a
single short-slab component. The implementation is not wrong; the proposed
intervention is simply too close to the parent target to alter optimization
reliably. Do not tune connectivity, slab width, morphology, or component
thresholds. A subsequent hypothesis must pass a non-identity/activity screen
and address an optimization mechanism that affects the complete SliceEqOcc
objective.

