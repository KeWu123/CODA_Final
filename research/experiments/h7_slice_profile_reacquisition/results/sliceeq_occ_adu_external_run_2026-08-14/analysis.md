# H7.11 SliceEqOcc-ADU result analysis

## Decision

H7.11 is neutral/negative and is closed. The supplied best observed test
checkpoint reaches Dice `0.843335`, but does not exceed the accepted
SliceEqOcc development result `0.844566` (`-0.001231`). The run also fails the
locked validation gate: its supplied-log best is `0.815701`, below both the
matched parent best `0.817373` (`-0.001672`) and the preregistered ADU pass
threshold `0.820373` (`-0.004672`). Do not rescue ADU by tuning JS temperature,
weight floors, pass count, seeds, or schedules.

## Artifact identity and completeness

The log is a seed-1337 `SliceEqOccADU_PROMISE12` run with the locked recipe and
shared Pre10000 SHA-256. It contains 138 validation points through iteration
27600 and no completion marker, so no claim is made about 27800--30000. The
performance file explicitly evaluates `iter_20800_dice_0.8103.pth`; this is not
the best-validation checkpoint within the supplied log, which is 27000. The
user reports that later test checkpoints decline. Only one test report is
present in the supplied artifacts, so that test trajectory is accepted as a
user observation rather than reconstructed evidence.

The validation trajectory is oscillatory rather than monotonically declining:
mean validation Dice is `0.79220` over 15k--20k, `0.80079` over 20k--24k, and
`0.80847` over 24k--27.6k. It reaches `0.810292 @ 20.8k`, `0.813657 @ 23k`, and
`0.815701 @ 27k`, before the supplied last value `0.808051 @ 27.6k`. Therefore
the observed late test degradation and five-case validation selection are not
interchangeable notions of overfitting.

## Comparison with SliceEqOcc

Against the accepted SliceEqOcc 23k development checkpoint, ADU wins six of
ten cases but loses on mean Dice. The largest Dice losses are Case16
`-0.018595`, Case30 `-0.016889`, and Case09 `-0.009133`; the largest gain is
Case43 `+0.014904`. Mean surface metrics are also worse: HD95 changes from
`3.651809` to `4.045004`, and ASD from `1.439373` to `2.150427`. This is not a
uniform collapse or a single-case failure; it is a near-neutral redistribution
that does not improve the primary result.

## Why the positive mechanism gate did not become a Dice gain

The H7.10 gate showed that operator-space JS ranks errors inside a foreground
union. In full training, however, the intervention is nearly identity on the
whole image. Across 1k--10k, JS is active on only about `0.619%` of pixels and
mean reliability is `0.998742`; over 24k--27.6k activity falls to `0.370%` and
mean reliability rises to `0.999280`. Mean pseudo-fractional weight increases
from about `0.887` to `0.931`, while the two hard occupancies differ on only
about `0.14%` of late pixels.

Thus ADU can rank a small stochastic error subset without materially changing
the dominant optimization trajectory. Moreover, two passes from the same
teacher agree on shared systematic errors. Averaging them and applying a
near-unit normalized weight cannot correct a spatially coherent but wrong
pseudo component. The result closes dropout-disagreement weighting for this
project; it does not refute the main SliceEqOcc occupancy mechanism.

## Outer-loop consequence

All three local successor families are now closed: profile Monte-Carlo
reallocation (SAQ/SC/CAP), native-anchor additions (DA/AP-TNA), and stochastic
teacher reliability (ADU). A further method experiment must address the
formation of a coherent latent source occupancy before the through-plane
operator, rather than sampling the same profile more often or reweighting a
tiny disagreement set. The next bounded hypothesis is H7.12 slab-coherent
pseudo occupancy; it changes only the U pseudo-stack topology before the
existing SliceEq operator and leaves the parent model, batch36, loss,
train-mode EMA, validation, and inference untouched.
