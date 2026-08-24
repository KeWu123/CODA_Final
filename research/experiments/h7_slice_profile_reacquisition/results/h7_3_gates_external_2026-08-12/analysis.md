# H7.3 three-gate result analysis

## Verdict

H7.3 is rejected by its preregistered primary Gate 1. Do not implement or train
the acquisition-residual dual-measure loss, and do not change its threshold
after observing the result.

## Gate 1: the residual is not diluted

Exact-GT acquisition residuals occupy only `0.842669%` of labeled pixels but
already contribute `65.6513%` of the full soft CE+Dice logit-gradient norm and
`54.4893%` of CE-only gradient norm. This is a `77.91x` and `64.66x` enrichment,
respectively, relative to uniform spatial share. The registered prediction was
less than `20%`; the observed result is more than three times that threshold.

The unlabeled proxy branch independently shows the same pattern: `0.813309%`
support contributes `50.7788%` of full CE+Dice gradient, a `62.43x` enrichment.
Further residual normalization would therefore over-amplify an already dominant
boundary/interface signal and is a plausible route to instability, not a
supported optimization.

## Gate 2: geometry is good, local magnitude is not

The frozen-student proxy passes the registered fidelity gate. Residual support
precision is `0.887736`, recall `0.855451`, IoU `0.771941`, outside-support mass
is only `0.112088`, and per-sample residual mass correlation is `0.992766`.
Thus the model identifies which slices and boundary regions undergo acquisition
change with high fidelity.

However, union-support pixelwise residual correlation is only `0.348508`, just
above the `0.30` floor. Location and total amount are reliable, but the local
fractional magnitude produced after hard/LCC pseudo segmentation is weakly
aligned with exact occupancy. This is the actionable bottleneck exposed by the
gate result.

Gate 2 remains proxy evidence because the checkpoint has no EMA teacher state.

## Gate 3: the acquisition weights are specific but unnecessary

Fractional residual weighting and same-support binary weighting are not
equivalent: labeled mean gradient cosine is `0.933901` and mean unit-gradient
distance is `0.313423`. The residual coefficient of variation is `0.452083`, so
the operator produces meaningful within-support weights. This preserves the
scientific distinction from a generic binary boundary loss, but Gate 1 shows
that adding the separate weighted risk is unnecessary.

## Direction

Keep SliceEq and H7.2 as the positive parent direction, but close H7.3. The next
candidate is acquisition-commuted posterior supervision:

`f_student(A_h X) ~= A_h f_teacher(X)`.

It replaces the current sequence `A_h(one_hot(LCC(argmax(q))))` on the unlabeled
target with a profile-transformed teacher posterior, optionally topology-gated
by the existing LCC. Exact-GT labeled occupancy and the original labeled anchor
remain unchanged. This directly targets local occupancy-magnitude fidelity and
does not add more boundary weight.

Before training, compare current hard-LCC occupancy, raw soft posterior
commutation, and LCC-gated soft posterior commutation on labeled stacks with the
same frozen 23k checkpoint and identical profile draws. Require a material
improvement in exact-support occupancy error and residual correlation without
increasing acquisition-change mass outside the exact support. Failure closes
posterior commutation and forces a broader pivot rather than another SliceEq
loss modification.
