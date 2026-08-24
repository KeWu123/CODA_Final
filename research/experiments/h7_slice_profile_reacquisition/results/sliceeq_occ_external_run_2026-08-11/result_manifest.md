# SliceEqOcc external run manifest

- Evidence date: 2026-08-11
- Evidence class: exploratory, one fixed seed
- Training log source SHA-256: `93FE37C576E0CA575E25F2938086D1842657FAF9B9A1647B68672BEFA2BE8442`
- Selected-best performance SHA-256: `0AA180876C1EB856F3D797F8B61C1003676DAA4C48BF005E81330B3D6FB6F254`
- Iteration-23000 performance SHA-256: `2EA1370080CCD36EE959B74071115BDFD63E84EBD14EEDAB0B3EAE5429089D55`
- Seed: 1337
- Shared pretrain SHA-256: `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- Fixed profile: radius 1, sigma `[0.45, 0.85]`, phase `[-0.25, 0.25]`
- Loader batch: 24 = 12 labeled + 12 unlabeled
- Effective post-warmup student batch: 36
- Validation cases: 5
- Test cases: 10

The supplied `selected_best_test_performance.txt` identifies only
`unet_best_model.pth`. Under the implemented selection rule this is the last validation
best, iteration 30000 with validation Dice 0.817373. It tests at Dice 0.827368.

The subsequently supplied `oracle_iter23000_val08152_test_performance.txt` explicitly
identifies `iter_23000_dice_0.8152.pth` and reports test Dice 0.844566, Jaccard 0.732999,
HD95 3.651809, and ASD 1.439373. The checkpoint binary and its SHA-256 were not supplied.
The user reports that several adjacent earlier checkpoints also tested near 0.84; those
additional files are not part of the current artifact. The 0.844566 result is therefore an
artifact-backed post-hoc oracle checkpoint, not the validation-selected primary result.
