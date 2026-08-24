# H7 SliceEq Implementation Protocol

Status: locked before implementation; exploratory fixed-seed full version requested by the user.
No performance improvement is claimed before a completed run.

## Frozen baseline contract

- Use the existing PROMISE12 root default, ordered split lists, first 191 labeled slices,
  2D U-Net, batch 24 with 12 labeled samples, SGD 0.01, EMA 0.99, seed 1337,
  30k self-training iterations, hard CE+Dice, the inherited consistency ramp, 2D LCC,
  validation cadence, checkpoint selection, and inference graph.
- Consume exactly the fixed Baseline Pre10000 checkpoint at the established default path,
  while permitting an explicit CLI path override if the file is moved. Require both `net`
  and matching `opt`; load strictly, record SHA-256, and reset all self-training RNGs after
  loading. Never auto-discover or rank pretraining checkpoints.
- Keep the labeled student input and target as the original transformed central slice.
- Keep iterations 0--999 on the exact baseline identity path.
- Do not add confidence filtering, soft targets, extra losses, extra heads, 2.5D inference,
  acquisition metadata, or a second optimizer.

## Single intervention

For every training-list item, load the real within-case stack at integer offsets
`[-1, 0, +1]`. Clamp offsets only at the first/last slice of a case, and reject malformed,
duplicate, or non-contiguous slice indices before training. Apply one shared draw of the
baseline spatial transform to the complete image/label stack.

After the 1k warmup, the teacher predicts every slice in each unlabeled three-slice stack.
Each prediction is converted to the baseline's hard 2D-LCC mask. For sample `b`, draw

`sigma_b ~ Uniform(0.45, 0.85)` and `phase_b ~ Uniform(-0.25, 0.25)`,

then form normalized nonnegative weights

`h_b(k) proportional to exp(-0.5 * ((k - phase_b) / sigma_b)^2)`.

The student receives

`x'_b = sum_k h_b(k) x_b(z+k)`

and is supervised by the hard paired occupancy target

`y'_b = argmax_c sum_k h_b(k) one_hot(y_teacher,b(z+k), c)`.

The same sampled weights must be used for image and target. The student still receives one
channel. The supervised and unsupervised losses are evaluated once, followed by exactly one
backward pass, optimizer update, and EMA update.

## Runtime invariants and diagnostics

- Kernel weights are finite, nonnegative, and sum to one for every sample.
- Output image shape is `[B_u, 1, H, W]`; hard target shape is `[B_u, H, W]` and values lie
  in `[0, num_classes-1]`.
- The teacher and sampled acquisition tensors are detached from autograd.
- Log mean `sigma`, absolute phase, center weight, image absolute change, target changed
  fraction against the teacher's center mask, foreground fractions before/after, and the
  fraction of samples whose neighbor paths were clamped.
- Save original/re-acquired unlabeled images, paired target, center teacher mask, and their
  changes in TensorBoard.

## Failure conditions

- Abort before training if the ordered slice list cannot produce an unambiguous contiguous
  within-case adjacency map or if the configured fixed checkpoint is absent/incompatible.
- Abort a run on non-finite weights, outputs, losses, or diagnostics.
- Do not interpret a validation spike as evidence. The method is not promising unless the
  selected test result exceeds CoDA/OBA and approaches or exceeds the user-reported UniMatch
  result near 0.83 without the sustained late validation collapse observed for OBA.
- If the target changed fraction is numerically zero for almost the entire run, the paired
  occupancy contribution was inactive; if it is large or non-boundary-localized in the
  labeled validity audit, reject the profile range rather than tuning after test inspection.
