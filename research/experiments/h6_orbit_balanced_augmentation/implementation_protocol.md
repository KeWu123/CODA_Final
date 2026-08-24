# H6 OBA Final Single-Seed Implementation Protocol

## 2026-08-11 shared-checkpoint amendment

For prospective reruns, the user authorized replacing OBA's internal supervised pretraining
with the fixed Baseline Pre10000 checkpoint used by CoDA and SliceEq. `train_oba.py` now
strictly restores `net+opt`, logs SHA-256, resets the self-training RNG, and runs only the
30,000-step self-training stage. The archived OBA result predates this amendment and remains
identified as an independently pretrained run.

## Authorization and evidence class

The user explicitly requested the complete final OBA implementation before any
sub-experiment and will run only the current fixed seed (`1337`). This supersedes the
previous Gate-0-first sequencing. The first full result is **exploratory single-seed
evidence**; later factor controls may explain a positive result but may not retroactively
make the first run confirmatory.

The workspace has no Git executable/repository, so the protocol cannot be committed.
This timestamped file and the research log are the available pre-implementation record.

## Frozen baseline contract

Keep the current defaults and logic unchanged:

- root: `/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source`;
- experiment identity changes only to `OBA_PROMISE12`;
- seed 1337, deterministic cuDNN, 2-D U-Net, two classes, 256x256;
- fixed shared 10,000-step supervised-pretraining checkpoint + 30,000 self-training iterations;
- loader batch 24 = 12 labeled + 12 unlabeled, labelnum 7;
- SGD lr 0.01, momentum 0.9, weight decay 1e-4;
- EMA decay 0.99;
- original TwoStream sampler, list order, patient-to-slice mapping and validation;
- teacher receives the original unlabeled loader tensor;
- teacher remains in the inherited baseline mode;
- teacher output becomes the same 2-D LCC hard mask;
- student uses the same hard CE + Dice losses and consistency ramp;
- student remains the validated/saved/inference model;
- the first 1,000 self-training iterations use the exact identity baseline path.

Do not edit `train_baseline.py`, `test_baseline.py`, `dataset.py`, or `test_coda.py`.
The later authorized `train_coda.py` edit is limited to shared-checkpoint infrastructure;
the CoDA method path is unchanged.

## Final OBA operator

For every unlabeled slice after iteration 1,000, independently sample one family and
one magnitude. Generate an antithetic pair from the same sampled coordinate:

`x_plus, x_minus = T(x, +a), T(x, -a)`.

The fixed family bank is:

1. `log_gamma`, magnitude `a ~ U(0.10, 0.40)`. Min-max normalize each slice, use
   exponents `exp(+a)` and `exp(-a)`, then map to the original range.
2. `smooth_bias`, magnitude `a ~ U(0.10, 0.35)`. Sample an 8x8 field per slice,
   bicubically upsample, remove its mean, normalize its RMS and maximum magnitude, and
   apply opposite offsets in normalized-intensity logit space.
3. `gaussian_noise`, magnitude `a ~ U(0.05, 0.15)` relative to per-slice standard
   deviation, using exactly opposite noise `+eps/-eps`.

All sampling is per sample, not per batch. A dedicated CUDA generator seeded with the
existing seed prevents OBA draws from consuming the model/dropout RNG stream. Constant
images remain exact identity. Outputs must be finite, detached, and shape preserving.

## Symmetric-batch training integration

At each post-warmup step:

1. construct one student batch `[labeled, x_plus, x_minus]`;
2. forward this batch once, so both orbit endpoints share the same BatchNorm
   realization and the labeled anchor is evaluated once;
3. obtain one clean teacher LCC hard mask from the original unlabeled inputs;
4. compute the original supervised hard loss once and average the two unchanged
   unlabeled hard CE+Dice losses;
5. execute exactly one backward pass, optimizer step, and EMA update.

The loader and sampling contract remain batch 24. The post-warmup student forward has
effective batch 36 (`12 L + 12 U+ + 12 U-`), as explicitly authorized by the user.
This is expected to use about 1.5x baseline activation memory and compute. The labeled
images are not augmented, and the supervised gradient retains its original scale.

## Diagnostics included in the final run

TensorBoard records total/supervised/unsupervised losses, consistency weight, family
fractions, mean severity, mean absolute displacement for both signs, input displacement
cosine, pair midpoint drift, plus/minus unsupervised-loss gap, pair prediction
disagreement, and pair probability gap. Every 20 steps it logs original, plus, minus,
pair mean, and absolute displacement images.

These diagnostics do not select samples or change the loss.

## Test entry

Add `test_oba.py` as an inference-only wrapper around the existing evaluator. Its
default experiment is `OBA_PROMISE12`; automatic cross-experiment checkpoint search is
disabled by default to prevent silently evaluating CoDA/BMER/baseline weights.

## Acceptance checks before handoff

- utility tests cover exact antithetic Gaussian noise, reciprocal gamma parameters,
  deterministic sampling, per-sample family assignment, constant-image identity,
  finite outputs, and invalid arguments;
- contract tests freeze all existing source hashes and OBA parser defaults;
- AST/source checks confirm the clean teacher input, original LCC hard target, hard
  CE+Dice, one symmetric student batch, averaged pair loss, one optimizer step, and one
  EMA update;
- all Python sources compile if a Python runtime is available;
- run the available unit suite; record unavailable dependencies/runtime explicitly.

No metric is claimed by this implementation cycle.
