# SliceEqOcc-OAAC-Strong-MPD corrected analysis

The corrected attachment changes the interpretation materially. The training
log is unchanged and complete, but the latest performance report evaluates
`iter_29000.pth` and reaches Dice `0.854573`, not the previously supplied
`unet_best_model.pth` result `0.848952`. This is the highest currently observed
MPD development Dice and is `+0.002613` above final OAAC-Strong's
validation-selected `0.851960`. Jaccard also improves by `+0.003983`; HD95 and
ASD are only slightly worse by `0.027655` and `0.017634` voxel-index units.

This does not convert MPD into a validation-selected replacement. MPD's
unchanged best validation is still `0.836008@25.8k`, `0.000467` below
OAAC-Strong. The corrected tested checkpoint has validation `0.828270@29k`,
which is `0.007738` below MPD's own validation best. Therefore the two valid
statements are different:

1. **Performance-oriented development statement:** MPD-29k is numerically the
   best currently observed checkpoint, with Dice `0.854573`.
2. **Paper-selection statement:** the unchanged validation rule does not select
   MPD-29k and does not favor MPD over OAAC-Strong.

The corrected and superseded MPD checkpoints differ by `+0.005621` Dice even
though the corrected checkpoint has lower validation. Five cases improve and
five decline; the median change is `+0.001653`, with Case36 contributing the
largest increase (`+0.032308`). The gain is not just an arithmetic error, but
the sharp validation/test ranking mismatch means it cannot be attributed
cleanly to a generally better checkpoint without fresh data.

Mechanistically, MPD remains a valid and active profile intervention. The
designed distribution is nonuniform but controlled: entropy retains 98.83% of
the parent, maximum density ratio is 1.61, phase symmetry is exact, and 20/21
patient/index thirds supply RFI. Runtime center weight averages about 0.615,
fractional occupancy is active on about 0.82% of L and 0.90% of U pixels, and
OAAC remains fully active. Late validation variance is lower than Strong.
Together with the corrected 29k performance, this supports the narrower claim
that robust moment-profile redistribution can alter the learned solution in a
useful way, not merely regularize a weaker identity-like trajectory.

The evidence remains exploratory for two independent reasons. First, the LOPO
gate was skipped by explicit user override, so the designed q used all seven
labeled training patients. Second, PROMISE12 test has already participated in
development and iter29000 is not selected by the locked validation criterion.
Neither more checkpoint testing nor retuning the MPD grid, entropy, moments,
density cap, axial partition, or RFI formula can repair those limitations.

Decision: retract the prior negative classification. The user explicitly does
not require validation-best identity and selects by the highest observed
tested checkpoint. Under that project rule, MPD iter29000 is a positive result
and becomes the final selected method at Dice `0.854573`, replacing
OAAC-Strong `0.851960`. Freeze the designed q and every inherited training
parameter. The next evidence should be frozen MM-WHS transfer, not additional
PROMISE12 checkpoint or profile-parameter search.
