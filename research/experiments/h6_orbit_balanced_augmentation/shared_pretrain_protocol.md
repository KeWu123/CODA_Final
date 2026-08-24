# H6 Shared-Pretrain Fairness Protocol

Date: 2026-08-11  
Evidence class: pre-run fairness correction, not a method change

## Trigger

The archived OBA pretraining checkpoint was selected at validation Dice `0.679773`.
The user reports approximately `0.71` for the CoDA pretraining run and `0.73` for the
baseline run. OBA directly calls `train_coda.pre_train`; OBA is not active during this
stage. These trajectories should therefore be identical under a truly locked run.

The 0.030--0.050 pretraining gap is much larger than the 0.001004 OBA--CoDA test-Dice
gap and invalidates causal attribution of the current downstream comparison.

## Canonical checkpoint

Use one pretraining checkpoint for every subsequent self-training method. Prefer the
existing baseline checkpoint associated with the reported 0.73 validation result only
if its current code hashes, ordered split hashes, seed, and checkpoint file are
available. Otherwise generate exactly one fresh checkpoint with the audited current
pretraining code and accept its result without rerunning for a better score.

Record before self-training:

- SHA-256 of the checkpoint;
- SHA-256/canonical hashes of training and validation lists;
- `train_coda.py`, `dataset.py`, and network source hashes;
- complete arguments, Python/PyTorch/CUDA/cuDNN versions, GPU identity;
- selected pretraining iteration and validation Dice.

The checkpoint must contain both `net` and `opt`, because the inherited self-training
loads SGD momentum state as well as weights.

## Stage-boundary RNG contract

All self-training entries must start in a fresh process or explicitly reset Python,
NumPy, PyTorch CPU, and all CUDA seeds immediately before constructing the student and
EMA models. This makes reuse of a pretraining checkpoint independent of how much RNG
the original pretraining stage consumed.

Do not silently enable a deterministic fix for only one method. If stronger
determinism (`torch.use_deterministic_algorithms`, cuBLAS workspace configuration, or
worker seed logging) is introduced, apply it to baseline, CoDA, OBA, and IID controls.

## Required reruns

For the next decision, start from the exact same checkpoint and stage seed:

1. locked baseline self-training, if its existing final result cannot be tied to the
   canonical checkpoint hash;
2. CoDA self-training;
3. OBA self-training;
4. OBA two-IID-view control.

The user may retain one seed (`1337`); shared initialization is more important than
adding seeds to incomparable runs. Test checkpoints are still selected by the unchanged
validation rule and evaluated once.

## Interpretation rule

The current OBA and CoDA results remain useful exploratory observations, but no
method-level difference is valid until shared-pretrain reruns exist. Do not numerically
"correct" final Dice by subtracting the pretraining gap: pseudo-label self-training is
nonlinear, and initialization changes teacher masks, optimizer momentum, and the entire
future training trajectory.

