# SliceEqOcc-OAAC Strong-all (H7.15 first run)

This is the isolated first parameter-calibration run requested after the
original OAAC result. It does not edit or import-modify the original
`train_sliceeq_occ.py` or `train_sliceeq_occ_oaac.py` sources.

## Exactly what changes

- OAAC log-gamma range: `[-0.20, 0.20] -> [-0.25, 0.25]`.
- OAAC log-contrast range: `[-0.15, 0.15] -> [-0.1875, 0.1875]`.
- OAAC brightness/span range: `[-0.10, 0.10] -> [-0.125, 0.125]`.
- OAAC application probability remains `1.0`.
- Ordinary periodic raw-student weights are archived every `1000` iterations
  instead of every `3000` iterations in this successor only.

Validation remains every 200 iterations. Validation metrics, the strict
`performance > best_performance` rule, `unet_best_model.pth`, best-Dice saves,
optimizer, learning rate, EMA mode/decay, consistency ramp, warmup, batch,
loss, 30k length and inference are unchanged.

## Files

- `code/train_sliceeq_occ_h7_15_base.py`: isolated SliceEqOcc copy with only
  the periodic archive interval changed to 1000.
- `code/utils/sliceeq_oaac_strong.py`: fixed Strong-all appearance ranges.
- `code/train_sliceeq_occ_oaac_strong.py`: locked training entry.
- `code/test_sliceeq_occ_oaac_strong.py`: strict single-checkpoint 2-D test.
- `tests/test_sliceeq_oaac_strong_contract.py`: isolation and recipe contract.

## Run

From the repository `code` directory:

```bash
python train_sliceeq_occ_oaac_strong.py
```

The output directory is:

```text
../model/SliceEqOccOAACStrong_PROMISE12_7_labeled/self_train/unet/
```

It contains ordinary checkpoints `iter_1000.pth` through `iter_30000.pth`,
plus the unchanged validation-best artifacts.

Before training on the CUDA machine, run:

```bash
python -m unittest tests.test_sliceeq_oaac tests.test_sliceeq_oaac_contract tests.test_sliceeq_oaac_strong_contract -v
```

Test one explicitly selected checkpoint without auto-search:

```bash
python test_sliceeq_occ_oaac_strong.py \
  --checkpoint_path ../model/SliceEqOccOAACStrong_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

Checkpoint density is for diagnosis and archiving. It does not change the
selection rule and does not authorize choosing the final model by test Dice.
