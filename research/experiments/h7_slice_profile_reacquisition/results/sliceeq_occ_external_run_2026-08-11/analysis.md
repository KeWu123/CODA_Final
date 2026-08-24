# SliceEqOcc fixed-seed result analysis

## Decision

The run is a successful implementation and mechanism-activity check, but it does not yet
confirm H7.2 under the locked validation-selection rule. SliceEq remains the primary research
direction. The artifact-backed 0.844566 intermediate checkpoint is important exploratory evidence that
the trajectory contains a better-generalizing state, but selecting it after inspecting test
Dice would turn the test set into a validation set.

## Audited primary result

- Best validation under the implemented rule: 0.817373 at iteration 30000.
- Supplied test Dice for `unet_best_model.pth`: 0.827368.
- Jaccard: 0.707575; legacy HD95: 6.383736; ASD: 2.227637.
- Delta from SliceEq v1: -0.005235 Dice, with 4/10 case wins and median paired delta
  -0.005058.
- Delta from archived CoDA: +0.007493 Dice, with 6/10 case wins.
- Delta from archived OBA: +0.008497 Dice, but only 4/10 case wins.

The result exceeds the user-reported 0.78--0.80 legacy baseline range but does not exceed
SliceEq v1 (0.832603), the user-reported UniMatch value (about 0.83), or the predeclared 0.86
continuation threshold.

## Validation trajectory

The final validation maximum is not an isolated spike. Values at 28.8k, 29.0k, and 30.0k are
0.816474, 0.817338, and 0.817373. Over iterations 25.6k--30k, validation has mean 0.807252,
SD 0.007538, and median 0.808608. The actual isolated anomaly is iteration 25.4k, where Dice
drops to 0.673511 and recovers to 0.800198 after 200 steps.

The supplied performance artifact confirms validation 0.815152 at iteration 23k and test
Dice 0.844566; the user reports several adjacent earlier checkpoints near 0.84 as well. The
23k validation value is only 0.002221
below the final maximum, which is negligible relative to the observed five-case validation
fluctuation. This demonstrates poor checkpoint-rank agreement between the five-case validation
set and ten-case test set; it does not justify declaring the higher validation checkpoint
invalid after observing test results.

## Mechanism activity

After warmup, the logged means are:

- labeled fractional-pixel fraction 0.008214, entropy 0.005591, deviation 0.001628;
- unlabeled fractional-pixel fraction 0.008797, entropy 0.006108, deviation 0.001821;
- unlabeled hard-target-change fraction 0.000026.

All diagnostics are finite and satisfy their expected bounds. Labeled and unlabeled occupancy
activity is balanced, while hard argmax changes are nearly absent. This directly supports the
reason for H7.2: the fractional target carries acquisition information that v1 hard argmax
mostly discards. It also shows that the active supervision is spatially sparse, affecting
roughly 0.8--0.9% of pixels per logged batch.

## Reporting boundary and next gate

For current research bookkeeping, use 0.827368 as the admissible selected-checkpoint result.
The 0.844566 value is now backed by an exact model path and complete per-case performance file;
only the checkpoint binary hash remains absent. It may be shown as an exploratory
oracle-checkpoint result, but cannot be promoted to the primary unbiased test result because
multiple test checkpoints were inspected.

The 23k checkpoint is not a one-case outlier. Relative to the final selected checkpoint it
wins 9/10 cases, with mean paired delta +0.017198 and median +0.012773. Relative to SliceEq v1,
it gains +0.011963 mean Dice, has a positive median delta of +0.002077, and wins 5/10 cases;
its losses are generally small while several improvements are material. This upgrades H7.2
from a mere implementation success to a strong exploratory positive result, while leaving the
checkpoint-selection validity boundary unchanged.

Do not discard SliceEq or immediately tune the operator. First preserve the 23k checkpoint and
its test file. For the next untouched evaluation, pre-register a test-independent checkpoint
rule, such as the earliest checkpoint within a fixed tolerance of the maximum validation Dice,
or evaluate on a second untouched dataset. This separates a promising training trajectory
from the current checkpoint-selection failure.
