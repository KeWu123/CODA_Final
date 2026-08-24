# SliceEqOcc-OAAC-Strong-ARCP analysis

## Overall verdict

ARCP has three different outcomes at three levels:

1. **Mechanistically active:** about 69.5% of diagnostic samples receive a nontrivial profile calibration and mean `|alpha-1|` is 0.1323. This is not a near-identity intervention.
2. **Weakly positive on the validation proxy:** best validation rises from the parent `0.836475` to `0.838425`, an absolute gain of `0.001950`; the late validation window also has a higher mean and lower variance.
3. **Negative as a final replacement:** it misses the preregistered `0.839475` line and the validation-selected test Dice `0.851062` is below the parent `0.851960`, with worse Jaccard and surface metrics.

H7.18 is therefore a neutral/negative successor and does not replace SliceEqOcc-OAAC-Strong.

## Why validation improves without a test improvement

ARCP targets a real issue: the same random profile weights produce different image-residual magnitudes on different axial stacks. The calibration changes the training distribution, so a small validation and stability benefit is plausible.

However, equal response magnitude is not equivalent to equal segmentation value. Fast axial change at the prostate apex/base can produce a large acquisition residual precisely where fractional occupancy is informative. Pulling some profiles toward identity and strengthening others according to H5 image response can suppress this useful anatomical nonuniformity.

The diagnostics support this interpretation:

- mean center shift is only `0.0147`, so ARCP did not collapse into a global severity sweep;
- aggregate reference mismatch falls by only about `5.95%`;
- only `54.79%` of batch-level records move closer to the reference;
- the upper parent-support bound is hit in about `16.0%` of samples.

The result is a real but inconsistent redistribution of profile severity rather than a reliable acquisition-risk normalization.

## Trajectory and late behavior

The best checkpoint is 29.8k and the 30k value is lower by `0.008230`. This is better described as late five-case-validation oscillation than monotonic overfitting: the 20k--30k linear trend remains positive and both 29.4k and 29.8k are high points.

The ARCP 27k--30k validation mean/SD is `0.828529/0.006839`, compared with the parent record `0.821553/0.012075`. ARCP may stabilize late training, but that stability does not translate into a better validation-selected test model.

## Surface metrics

Test HD95 rises from the parent `3.228864` to `6.217004`, mainly because Case16 reaches `33.015148`. Removing Case16 gives an ARCP nine-case HD95 of about `3.239433`, close to the parent's overall value. The surface failure is therefore dominated by one remote false-positive/error component rather than a universal boundary collapse.

This case must not be used to design an ARCP-specific postprocess. The requested optimization is method-level, and the same PROMISE12 test has already been used throughout development. Even if the outlier were removed, ARCP Dice and Jaccard still do not exceed the parent.

## Research decision

- Do not adopt ARCP as the final method.
- Do not tune alpha, epsilon, reference quantiles, center-weight support, or the diagnostic grid after seeing this result.
- Do not stack ARCP with bin-integrated Gaussian weights or adversarial/learned profile selection.
- Keep `SliceEqOcc-OAAC-Strong` as the final selected method.
- If ARCP is mentioned in the paper, place it in an appendix/design analysis and state only that H5-observed axial-response calibration was active and mildly improved validation but did not improve the selector-compatible test result.

This result indicates that the current bottleneck is not simple dispersion of profile effect magnitude. Without original acquisition metadata, further H5-statistic-based weight calibration has insufficient evidence to justify another method extension.
