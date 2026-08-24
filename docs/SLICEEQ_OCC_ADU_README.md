# SliceEqOcc-ADU

`SliceEqOcc-ADU` is the single exploratory full experiment authorized by the
H7.10 mechanism screen. It remains a bounded SliceEqOcc successor with no
case-specific logic and no inference change. The ADU package contains two
coupled training-time operations: a two-pass mean occupancy target `q_bar`,
and JS-derived reliability weighting of that target. It must not be described
as a pure reliability-only intervention.

## Run training

First run the functional and isolation tests from the repository root. The
CUDA isolation test must report `ok`, not `skipped`:

```bash
python -m unittest tests.test_sliceeq_adu tests.test_sliceeq_adu_contract -v
```

Then run from the repository `code` directory on the CUDA server:

```bash
python train_sliceeq_occ_adu.py
```

The entry rejects changes to the established PROMISE12 root, fixed recipe,
seed 1337, loader batch 24, student batch36, 30k iterations, and experiment id
`SliceEqOccADU_PROMISE12`. It also verifies the shared Pre10000 checkpoint
SHA-256 before training.

## Test the validation-selected checkpoint

```bash
python test_sliceeq_occ_adu.py \
  --checkpoint_path ../model/SliceEqOccADU_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

Do not test periodic checkpoints to choose a better iteration. The validation
calculation and strict best-checkpoint rule are byte-equivalent to the parent
SliceEqOcc implementation. The validation-selected checkpoint is tested once
whether this exploratory run is positive or negative; test performance cannot
change the selected checkpoint or trigger a rescue variant.

The locked matched parent has best validation Dice `0.817373`; ADU passes the
exploratory method gate only at best validation Dice `>=0.820373`. The desired
test Dice of 0.85 is an optimization aim, not a checkpoint-selection rule.

## What changes

- The primary EMA pass is exactly the parent train-mode hard-LCC path.
- One extra teacher pass is isolated from persistent BN and CUDA RNG state.
- Both hard pseudo stacks use the same sampled slice profile.
- Their mean modeled acquisition-derived fractional occupancy is supervised
  with continuous
  `1-JS/log(2)` reliability in both soft CE and squared soft Dice.

There are no new trainable parameters, thresholds, schedules, student views,
or inference operations. Post-warmup forward views increase from 72 to 108;
peak student/backprop memory is unchanged apart from small target tensors.

## Expected logs

Every 200 iterations the training log includes:

- `JS(mean/max/active)`;
- `weight(mean/ESS/pseudo_fractional)`;
- `occupancy(abs/hard)`;
- the unchanged SliceEqOcc profile, occupancy, loss, and validation values.

TensorBoard writes the same values under `sliceeq_adu/`. They diagnose method
activity only and are not hyperparameters.

If the full run passes the locked validation criterion, the first attribution
control is a compute-matched `q_bar`-only run with `w=1`. Until that control is
positive against the parent and weaker than full ADU, any gain belongs to the
two-pass ADU package and cannot be assigned specifically to reliability
weighting.
