# SliceEqOcc-OAAC result analysis

## Conclusion

OAAC is the first bounded SliceEqOcc successor in the current sequence with a
nontrivial intervention coverage and a positive post-hoc oracle difference:
`0.849538` versus `0.844566` (`+0.004972`). The user clarified that 27k was
chosen as the maximum among multiple checkpoints inspected on test. Conditional
on these selected checkpoints, 7/10 cases improve and removing the largest
positive case still leaves an apparent mean delta of about `+0.002285`.

This is a test-selected development result, not merely an ordinary exploratory
periodic checkpoint. The supplied checkpoint is 27k, whereas validation
selected 23.8k. Ten cases and one seed do not establish a significant or
generalizable CVPR result; a simple case bootstrap for the conditional mean
delta includes zero and does not account for checkpoint-selection bias.

## The late curve is oscillation, not severe overfitting

Validation peaks at 0.834863 and finishes at 0.831964, a gap of only 0.002899.
From 27k through 30k, mean validation is 0.826892 with population SD 0.004865.
The self-training optimizer keeps LR=0.01 for the entire stage. The observed
high-level fluctuation is therefore more consistent with a constant-LR
trajectory around a good basin than with the large monotonic collapse seen in
DA/AP-TNA.

## Immediate no-model-change decision

1. The validation-selected 23.8k model may be evaluated once for protocol
   completeness, but the result remains development-only because this test was
   already used to select 27k.
2. A same-trajectory 24k/27k/30k weight average with BN recalibration can still
   be investigated as an internal optimizer, but it was proposed after seeing
   27k win on test and therefore cannot recover an unbiased primary result.
3. If a full retrain is affordable, apply the same LR-decay/SWA recipe to both
   SliceEqOcc and OAAC and evaluate it on a new untouched outer fold or external
   test. This is training stabilization, not a method contribution.

Do not search test checkpoints, averaging windows, NMS, TTA, thresholds, or
seeds. Existing `.pth` files contain only the student state, so low-LR resume
cannot reconstruct the optimizer or EMA trajectory.

## Paper interpretation

OAAC may be included as a performance extension only after a frozen
validation-selected rule is evaluated on a new untouched/hidden set and
multi-seed matched controls are positive. The paper's core contributions remain
paired through-plane re-acquisition and fractional occupancy. Generic
weak-to-strong photometric augmentation is established prior art and cannot be
claimed as the standalone novelty.
