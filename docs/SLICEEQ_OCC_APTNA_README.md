# SliceEqOcc AP-TNA

`train_sliceeq_occ_aptna.py` is an independent successor to the retained
SliceEqOcc experiment. It does not edit or overwrite the parent, DA, CAP, or
SC entries.

## Locked change

After the inherited 1,000-step warmup, the complete SliceEqOcc objective stays
active:

```text
L = Lsup + lambda * L_measurement + mu * L_native
mu = 0.5 * lambda * (1 - lambda / (5 * consistency))
```

The main student forward remains 36 views. Native-U is a separate 12-view
auxiliary forward with student BatchNorm running-stat updates disabled and its
CUDA RNG forked/restored. The EMA teacher remains in train mode.

Validation is copied unchanged from SliceEqOcc: the same five validation
cases, `val_2d.test_single_volume`, evaluation every 200 iterations, mean
Dice, and strict improvement checkpoint rule.

## Training

From `code/` on the CUDA server:

```bash
python train_sliceeq_occ_aptna.py
```

All defaults, including the PROMISE12 root and fixed pretrained checkpoint,
match the previous scripts. The isolated output directory is:

```text
../model/SliceEqOccAPTNA_PROMISE12_7_labeled/self_train/unet
```

## Testing

Use only the checkpoint selected by the unchanged validation rule:

```bash
python test_sliceeq_occ_aptna.py \
  --checkpoint_path ../model/SliceEqOccAPTNA_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

