# H4 CoDA-MT Implementation Protocol

## 2026-08-11 shared-checkpoint amendment

Prospective CoDA reruns no longer execute a separate supervised-pretraining stage.
`train_coda.py` defaults to the same fixed Baseline Pre10000 checkpoint used by OBA and
SliceEq, strictly restores `net+opt`, records SHA-256, resets the self-training RNG, and runs
only Self30000. This is fairness infrastructure and does not change the CoDA augmentation or
target formulation. The archived CoDA result predates the amendment.

Status: **locked before implementation**  
Date: 2026-08-10  
Experiment type: confirmatory infrastructure; no metric result is claimed by this commit.

## Source and Isolation

- Read-only source: `E:/Desktop/Baseline`.
- Source fingerprints are recorded in `baseline/audit.md`.
- Independent implementation root: `implementation/coda_mt_baseline`.
- Preserve the baseline directory layout and filenames where possible.
- Never modify or write checkpoints/logs into the source Baseline directory.

## Locked Baseline Contract

The implementation must not change:

- `train_slices.list`, `val.list`, `test.list`, their order, or their lookup paths;
- `patients_to_slices`, labeled/unlabeled index construction, or `TwoStreamBatchSampler`;
- `labelnum=7`, batch `24=12+12`, patch `256x256`, seed `1337`;
- the embedded two-class 2D U-Net;
- supervised pretraining and supervised loss;
- SGD, learning rate, momentum, weight decay, EMA decay, consistency ramp, iteration budgets, and validation frequency;
- validation/test membership or inference post-processing.

## Allowed Method Change

Only the unlabeled branch of `self_train` may change:

1. EMA teacher receives the existing loader tensor `weak_u`.
2. A coordinate-preserving degradation produces `strong_u` and a detached local evidence-loss field `gamma`.
3. Student receives the unchanged labeled batch concatenated with `strong_u`.
4. Teacher probabilities are constrained by the existing 2D largest-connected-component foreground prior without being converted to one-hot inside the component.
5. The dense target is `q_aug = (1-gamma) * q_lcc + gamma / C`.
6. Unlabeled loss is the arithmetic mean of soft cross-entropy and soft Dice. The existing 1,000-iteration delay and consistency weight are unchanged.

## Version-1 Degradations

- `resolution`: bilinear downsample-upsample with a sampled scale in `[0.25, 0.75]`; `gamma` is normalized local Sobel-gradient energy loss.
- `gaussian_noise`: zero-mean noise with standard deviation sampled in `[0.05, 0.20]` times each slice standard deviation; `gamma` is a bounded local noise-to-signal ratio.
- One family is sampled per unlabeled batch using PyTorch's seeded RNG. No geometry or label coordinates change.

The severity ranges are exposed as command-line arguments for ablation but the original baseline arguments/defaults remain untouched. CoDA-specific arguments use a `coda_` prefix.

## Numerical and Semantic Invariants

- `strong_u.shape == weak_u.shape`; `gamma.shape == [B,1,H,W]`.
- `gamma` is finite, detached, and in `[0,1]`.
- `q_lcc` and `q_aug` are finite distributions whose class probabilities sum to one per pixel.
- `gamma=0` returns `q_lcc`; `gamma=1` returns the uniform distribution.
- Soft CE and soft Dice are finite and differentiable with respect to student logits only.
- Teacher outputs, `gamma`, and pseudo-targets carry no gradient.

## Acceptance Checks Before Training

1. Source Baseline hashes remain unchanged.
2. The copied baseline matches the source before the method patch, excluding IDE caches, Python caches, and unrelated logs.
3. Every copied Python file parses with `ast`.
4. CPU unit tests cover shapes, bounds, probability normalization, limiting cases, deterministic augmentation under a fixed seed, LCC behavior, and backward propagation.
5. A synthetic mini-batch executes the CoDA helper path without PROMISE12 data.
6. A diff report shows that the method implementation changes only the copied training entry point plus new helper/test/documentation files.

## First Real-Data Execution Order

After moving the independent folder to the Linux data environment:

1. reproduce B0 with the original source code and existing lists;
2. run B1 severity diagnostic;
3. run B2--B6 only after B0/B1 sanity checks pass;
4. use the same fixed list hashes and seeds for every comparison.
