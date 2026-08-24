# H7.2 Fractional-Occupancy SliceEq Protocol

Status: locked before implementation on 2026-08-11. This is the independent next optimization
version requested after SliceEq v1 reached test Dice 0.832603. No positive result is claimed.

## Question

Does preserving the slice-profile-weighted tissue occupancy, rather than reducing it to a
nearly invariant hard argmax target, improve the median case and acquisition-sensitive
boundary performance while retaining the current 2D inference model?

## Frozen identity

- Do not modify or overwrite `train_sliceeq.py`, `test_sliceeq.py`, `utils/sliceeq.py`, or
  `dataloaders/sliceeq_dataset.py`.
- Keep the PROMISE12 root, split/order, first 191 labeled slices, seed 1337, fixed Pre10000
  checkpoint, 2D U-Net, SGD state, EMA 0.99, 30k iterations, loader batch 24, 12 labeled
  samples, consistency schedule, LCC pseudo masks, validation cadence, checkpoint selection,
  and inference graph.
- Keep the SliceEq profile at radius 1, sigma 0.45--0.85, phase -0.25--0.25. Do not tune
  severity after seeing v1 test cases.
- Keep iterations 0--999 on the exact baseline identity path.
- The new test entry must disable NIfTI prediction saving by default and must not auto-search
  checkpoints.

## Intervention

For a sampled normalized nonnegative profile `h`, construct

`x_phi = sum_k h(k) X[z+k]`

`o_phi = sum_k h(k) one_hot(Y[z+k])`.

After warmup:

1. Preserve the original labeled central slice and hard CE+Dice loss.
2. Construct one labeled re-acquired view using the exact neighboring GT masks and supervise
   it with `o_phi^L` through soft cross-entropy plus the baseline-compatible squared-
   denominator soft Dice.
3. Construct one unlabeled re-acquired view from detached EMA predictions after the existing
   per-slice hard argmax and 2D-LCC cleanup; supervise it with `o_phi^U` using the same soft
   losses.
4. Form one student forward of effective size 36:
   `12 original-L + 12 reacquired-L + 12 reacquired-U`.
5. Normalize the two supervised views:
   `L_sup = 0.5 * (L_hard_original + L_soft_reacquired_labeled)`.
6. Preserve the inherited unsupervised coefficient:
   `L = L_sup + lambda(t) * L_soft_reacquired_unlabeled`.
7. Perform exactly one backward, optimizer step, and EMA update.

Softness is not entropy injection. If all neighboring masks agree, occupancy remains exactly
one-hot. Fractional values occur only where the sampled profile mixes different masks.

## Diagnostics and failure gates

- Log labeled and unlabeled occupancy entropy, fractional-pixel fraction, occupancy deviation
  from the center mask, hard-argmax change fraction, foreground occupancy, image change,
  center weight, sigma, phase, and endpoint clamping.
- Mirror the scalar mechanism diagnostics into the text log at validation cadence so result
  analysis does not depend on receiving a TensorBoard event file.
- Reject the mechanism if fractional occupancy is effectively zero, is dominated by volume
  endpoints, or produces mean-only improvement while paired median/case coverage deteriorate.
- Require the same-checkpoint baseline and image-only slab control before making a causal or
  paper claim. This full fixed-seed run remains exploratory until those controls exist.

## Pre-implementation v1 identity hashes

- `code/train_sliceeq.py`: `64DBB13FD64B873C067425F61892B244725D0BFDDCEB401257D03D066856C12F`
- `code/test_sliceeq.py`: `6587C72E0F73A018EB1A5611951BFBABEB276F8B4D4506B9E1E235252E1A2E15`
- `code/utils/sliceeq.py`: `44A956A92ECCDBB2109034A05AC5EC72F190B33F740AD95C1ED8B505BAE168F7`
- `code/dataloaders/sliceeq_dataset.py`: `9CC39CD6ED373E22EC854340C7975868025C2BF74223370C1EEB69E928FC19D5`

The environment has no usable `git` executable, so a temporal protocol commit cannot be made;
this file and the research log provide the available pre-implementation record.

