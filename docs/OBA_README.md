# OBA final single-seed run

OBA is implemented in parallel files and does not edit the baseline or BMER method paths.
CoDA and OBA now share the same checkpoint-loading helper solely for fair initialization.

## Fixed defaults

- data root: `/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source`
- experiment: `OBA_PROMISE12`
- seed: `1337`
- labelnum: `7`
- loader batch: `24` (`12` labeled + `12` unlabeled)
- post-warmup student batch: `36` (`12` labeled + `12 U+` + `12 U-`)
- initialization/self-training: fixed shared Pre10000 checkpoint / `30000`
- OBA: per-sample antithetic log-gamma, smooth bias, or Gaussian noise after the
  original 1,000-step identity warmup

## Train

Run from the `code` directory so the unchanged relative model path matches the other
training entries:

```bash
cd code
python train_oba.py
```

The script performs the same strict PROMISE12 preflight as CoDA, strictly loads the
shared Baseline Pre10000 `net+opt` checkpoint, resets the self-training RNG, and runs
only OBA self-training. The default checkpoint is:

```text
/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

The output is written to:

```text
../model/OBA_PROMISE12_7_labeled/self_train/unet
```

Use `--pretrained_checkpoint /moved/path/unet_best_model.pth` only if the fixed
file was relocated. No checkpoint search or ranking is performed.

After iteration 1,000, the two antithetic views are concatenated into one symmetric
forward. Runtime and activation memory are expected to be roughly 1.5 times the
baseline self-training cost; the loader batch itself remains 24.

## Test

```bash
cd code
python test_oba.py
```

Automatic checkpoint fallback is disabled. The evaluator must find
`../model/OBA_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth`, or an explicit
checkpoint can be supplied:

```bash
python test_oba.py --checkpoint_path /absolute/path/to/unet_best_model.pth
```

## TensorBoard diagnostics

In addition to the inherited losses and validation metrics, inspect:

- `oba/family_fraction_*`
- `oba/severity_mean`
- `oba/displacement_cosine`
- `oba/midpoint_drift`
- `oba/plus_mean_absolute_change`
- `oba/minus_mean_absolute_change`
- `oba/pair_loss_gap`
- `oba/pair_prediction_disagreement`
- `oba/pair_probability_gap`

These are diagnostics only and do not change training.
