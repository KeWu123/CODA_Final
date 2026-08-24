# SliceEqOccDA: native-measurement dual anchor

`SliceEqOccDA` is an isolated successor to `SliceEqOcc`. It keeps the network,
shared pretrained checkpoint, EMA policy, optimizer, profile distribution,
warmup, consistency ramp, validation, and single-slice inference graph fixed.

After the first 1,000 identity iterations, the student sees 48 views:

- 12 native labeled center slices;
- 12 paired re-acquired labeled slices;
- 12 paired re-acquired unlabeled slices;
- 12 native unlabeled center slices.

The labeled loss is unchanged:

```text
L_sup = 0.5 * (L_native_L_hard + L_measurement_L_soft)
```

The only method change is the unlabeled dual anchor:

```text
L_U = 0.5 * (L_measurement_U_soft + L_native_U_hard)
L_total = L_sup + lambda(t) * L_U
```

Both unlabeled targets come from the same train-mode EMA prediction stack.
The native target is the center LCC hard pseudo mask; the measurement target
is the existing profile-weighted fractional occupancy. No unlabeled ground
truth is consumed. `ema_model.eval()` is deliberately not called, matching the
baseline and all preceding SliceEq variants.

## Training

From `code/`:

```bash
python train_sliceeq_occ_da.py
```

The locked defaults retain the existing paths and parameters. To supply the
same checkpoint explicitly:

```bash
python train_sliceeq_occ_da.py \
  --pretrained_checkpoint /home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

Outputs are isolated under:

```text
../model/SliceEqOccDA_PROMISE12_7_labeled/self_train/unet/
```

The log records the measurement/native loss, CE, and Dice separately every
200 iterations.

## Strict inference

```bash
python test_sliceeq_occ_da.py \
  --checkpoint_path ../model/SliceEqOccDA_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

Inference is identical to SliceEqOcc and has no added views or parameters.

