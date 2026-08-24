# SliceEq external fixed-seed run manifest

- Imported: 2026-08-11
- Evidence class: exploratory, single fixed seed
- Seed: 1337
- Self-training: 30,000 iterations, batch 24 (12 labeled + 12 unlabeled)
- Labeled volumes: 7
- Pretrained checkpoint: `/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth`
- Checkpoint SHA-256 reported by the run: `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`
- SliceEq profile: radius 1, sigma 0.45--0.85 slice units, phase -0.25--0.25 slice units
- Source `Z:\Downloads\log.txt` SHA-256: `C2E8997F7608FE63072176057E0FE6B66B4C01A28F8AF5FB7DDA88CD309B9E38`
- Source `Z:\Downloads\performance.txt` SHA-256: `38FF278C8A647583DE01C64351FC74AF2B53177D483911E62847748619B35A02`
- Archived raw files: `training_log.txt`, `test_performance.txt`

The text log contains validation observations but not the TensorBoard-only SliceEq activity
diagnostics. In particular, `target_changed_fraction`, occupancy entropy, foreground-volume
change, and endpoint-clamping activity cannot be reconstructed from the supplied files.

