# OAAC optimization without architecture changes

Date: 2026-08-17

## Question

Can the completed SliceEqOcc-OAAC trajectory be improved without changing the
U-Net, its parameter count, the training objective, or the single-slice
inference graph?

## Primary references

### Stochastic Weight Averaging

- Izmailov et al., *Averaging Weights Leads to Wider Optima and Better
  Generalization*, UAI 2018.
  <https://arxiv.org/abs/1803.05407>
- The method equally averages multiple SGD iterates sampled under a constant or
  cyclical learning rate. It is designed to move a single model toward the
  center of a flat basin without ensemble inference.
- Relevance: OAAC uses SGD with a constant learning rate of 0.01 for all 30k
  self-training steps and exhibits a high but oscillatory late validation
  trajectory. This is a much closer match to SWA's premise than another target
  or augmentation repair.

### Model soups

- Wortsman et al., *Model Soups: Averaging Weights of Multiple Fine-Tuned
  Models Improves Accuracy Without Increasing Inference Time*, ICML 2022.
  <https://proceedings.mlr.press/v162/wortsman22a.html>
- The paper supports the general fact that models in one low-error basin can be
  averaged in weight space while retaining one-model inference cost.
- Boundary: the published setting averages separately fine-tuned models; our
  proposed first gate averages three points from one OAAC SGD trajectory and is
  therefore better described as sparse trajectory weight averaging, not a new
  model-soup contribution.

### Batch-normalization recalibration

- PyTorch SWA guidance explicitly requires recomputing BatchNorm activation
  statistics after swapping to averaged weights.
  <https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/>
- Relevance: the OAAC U-Net contains BatchNorm throughout the encoder/decoder.
  Averaging stale running means/variances is not a faithful evaluation of the
  averaged parameters. Calibration must use the same 36-view OAAC student input
  distribution, with dropout disabled and no gradient update.

### Learning-rate scheduling in semi-supervised segmentation

- The official UniMatch implementation uses polynomial decay
  `lr=lr0*(1-iter/total_iters)^0.9`.
  <https://github.com/LiheYoung/UniMatch/blob/main/fixmatch.py>
- Relevance: the local parent never updates optimizer parameter-group learning
  rates after loading the shared pretraining optimizer; it logs the fixed
  `base_lr=0.01`. A matched PolyLR rerun is a defensible second optimization
  recipe if trajectory averaging is neutral.

### Mean Teacher readout

- Tarvainen and Valpola, *Mean Teachers Are Better Role Models*, NeurIPS 2017.
  <https://papers.nips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html>
- The original work evaluates exponential-moving-average weights. The local
  SliceEqOcc implementation instead saves only raw student weights. Saving an
  EMA readout is a valid diagnostic for a future matched rerun, but it is not
  the first recommendation because the local teacher is deliberately kept in
  train mode and its BatchNorm-buffer policy is entangled with the historical
  baseline.

## Synthesis

The highest-value no-architecture intervention is fixed trajectory weight
averaging, because it targets the observed optimization variance and has full
coverage over all weights. It adds no module, target, hyperparameterized mask,
or inference branch. A single fixed sparse average can be evaluated from
existing checkpoints before spending another full training run.

The existing `0.849538` is the maximum selected after evaluating multiple
checkpoints on the local ten-case test split. Any soup, LR schedule, or EMA
readout evaluated on that same split remains development-only even if it
exceeds 0.85. Paper-level confirmation requires a frozen rule on a fresh
hidden/external evaluation.
