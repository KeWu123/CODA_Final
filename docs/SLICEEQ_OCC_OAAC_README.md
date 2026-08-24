# SliceEqOcc-OAAC

OAAC is a single full-run successor of the original SliceEqOcc. It preserves
paired acquisition and fractional occupancy, then applies a monotonic,
coordinate-preserving appearance transform only to the 12 unlabeled
re-acquired student images. It does not include SCPO or ADU.

The semantic order is:

```text
weak teacher stack
  -> paired SliceEq acquisition of image and occupancy
  -> appearance-strong U image, unchanged occupancy target
```

The L branches, teacher train mode, student batch36, loss/ramp, optimizer/EMA,
validation and 2-D inference remain the parent implementation.

“L unchanged” refers to the L inputs, targets, loss definitions, and weights.
Because all 36 student views share one U-Net forward with BatchNorm, changing
the U appearance can also change the joint batch statistics seen by L. This is
part of the weak-to-strong training intervention, not a separate L transform.

## Training-server tests

From the repository root:

```bash
python -m unittest tests.test_sliceeq_oaac tests.test_sliceeq_oaac_contract -v
```

The tensor tests must execute with PyTorch and pass. Then:

```bash
cd code
python train_sliceeq_occ_oaac.py
```

The entry locks the complete recipe, Pre10000 hash and appearance ranges.

## Strict evaluation

```bash
python test_sliceeq_occ_oaac.py \
  --checkpoint_path ../model/SliceEqOccOAAC_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

Only the validation-selected checkpoint is primary. Best validation must reach
at least `0.820373` to pass the optimization proxy. The user's end-to-end
development target passes only if that one selected checkpoint is strictly
`>0.849` test Dice. A validation-only pass is not yet a final Dice gain, and a
different periodic test checkpoint cannot select the model.

The completed seed-1337 run deviated from that second-stage rule: the user
clarified that multiple checkpoints were inspected on the local test split and
`iter_27000.pth` was retained because it had the highest observed test Dice
(`0.849538`). It is therefore a test-selected development oracle. Evaluating
the validation-selected checkpoint later can complete the record but cannot
restore an untouched test; confirmatory evidence requires a fresh hidden,
external, or outer-fold evaluation.

## Stop rule

If OAAC is negative, do not add noise, blur, CutMix, longer chains, adaptive
strength or range sweeps. Move to the acquisition-gradient conflict gate or
the SliceEqOcc paper evidence matrix.
