# SliceEqOcc-OAAC Strong-all analysis

Strong-all is a real positive parameter-calibration result. It raises the
unchanged validation best from `0.834863` to `0.836475` and its validation-best
checkpoint reports test Dice `0.851960`. Unlike the prior OAAC result, the
supplied test artifact follows the validation-best model identity.

The result does not yet prove that scale 1.25 is optimal. Only scale 1.0 and
1.25 have been observed, so the augmentation-response curve has not been
bracketed. At the same time the validation gain is modest (`+0.001612`, below
the locked `+0.003` material margin), late validation variability rises, and
the per-case development difference versus the earlier OAAC oracle has only
5/10 wins with a slightly negative median. This rules out a fine Cartesian
grid but supports one outer bracketing point.

The next and final OAAC-scale experiment is scale 1.50 with probability 1.0.
If its unchanged best validation does not exceed `0.836475`, scale 1.25 is
retained as the local winner and the OAAC-scale search closes. No 1.125, 1.375,
component-wise, seed, or test-checkpoint rescue is allowed.
