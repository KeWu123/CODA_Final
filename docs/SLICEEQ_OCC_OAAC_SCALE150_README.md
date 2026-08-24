# SliceEqOcc-OAAC scale1.50 (H7.16)

This is the single outer-bracket experiment after Strong-all scale1.25 reached
validation Dice `0.836475` and validation-selected test Dice `0.851960`.

## Single method change

- log-gamma: `[-0.30,0.30]`;
- log-contrast: `[-0.225,0.225]`;
- brightness/span: `[-0.15,0.15]`;
- application probability: `1.0`.

Everything else is inherited from the isolated H7.15 base, including the
1000-iteration ordinary checkpoint archive. Original SliceEqOcc, OAAC and
Strong-all files are unchanged.

## Train

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u train_sliceeq_occ_oaac_scale150.py
```

Output:

```text
../model/SliceEqOccOAACScale150_PROMISE12_7_labeled/self_train/unet/
```

## Test the unchanged validation-best checkpoint

```bash
python -u test_sliceeq_occ_oaac_scale150.py \
  --checkpoint_path ../model/SliceEqOccOAACScale150_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

Decision: scale1.50 must exceed validation `0.836475` to replace scale1.25.
Otherwise OAAC scale tuning stops and scale1.25 remains selected.
