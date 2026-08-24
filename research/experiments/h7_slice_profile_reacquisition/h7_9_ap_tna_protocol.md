# H7.9 Acquisition-preserving transient native anchor protocol

## Status

Locked before implementation on 2026-08-14. This is one exploratory
optimization experiment built directly on the selected SliceEqOcc method.
Git is unavailable in the local workspace, so the protocol timestamp and
frozen-parent source hashes are used as the local preregistration record.

## Motivation

H7.8 dual-anchor training replaced half of the successful unlabeled
measurement objective with a highly redundant native hard-pseudo-label loss.
Its negative result therefore does not isolate whether a small native-state
bridge can stabilize optimization without weakening fractional occupancy.
H7.9 tests that narrower hypothesis.

## Locked intervention

Let the inherited consistency coefficient be `lambda(t)` and let
`lambda_max = 5 * consistency`, which is exactly the maximum already defined
by the baseline ramp. After the inherited 1,000-step identity warmup:

```text
r(t)       = clamp(lambda(t) / lambda_max, 0, 1)
mu(t)      = 0.5 * lambda(t) * (1 - r(t))
L_total    = L_sup + lambda(t) * L_U_measurement
                   + mu(t) * L_U_native
```

The complete SliceEqOcc measurement term is never attenuated. The native
center-slice hard pseudo-label term is a transient auxiliary bridge derived
from the same train-mode EMA pseudo stack and becomes exactly zero when the
inherited ramp reaches its maximum. No new schedule hyperparameter or manual
iteration breakpoint is introduced.

## Forward and state-isolation contract

- The parent 36-view main student forward is unchanged:
  12 original-L + 12 reacquired-L + 12 reacquired-U.
- The 12 native-U images use a separate auxiliary student forward.
- Only for that auxiliary forward, student BatchNorm modules use their current
  running statistics without updating them; dropout remains in train mode and
  all affine parameters remain trainable.
- The auxiliary CUDA RNG stream is forked and restored. Consequently, its
  dropout draws do not advance the parent global RNG stream used by the next
  train-mode EMA call.
- There remains one optimizer step and one EMA update per iteration.
- The EMA teacher deliberately remains in train mode, matching the baseline
  and all prior SliceEq experiments.

## Frozen factors

- PROMISE12 lists/split and seven labeled cases;
- seed 1337 and the shared net+optimizer pretraining checkpoint;
- U-Net, EMA decay 0.99, and teacher train-mode policy;
- 30,000 self-training iterations and 1,000-step identity warmup;
- loader batch 24 and labeled batch 12;
- SliceEq radius, sigma/phase ranges, profile RNG streams, and paired operator;
- complete labeled SliceEqOcc objective and complete unlabeled occupancy loss;
- inherited consistency ramp, optimizer, and EMA update;
- validation dataset, `val_2d.test_single_volume`, evaluation every 200
  iterations, mean-Dice computation, strict `performance > best_performance`
  checkpoint rule, periodic checkpoints, and inference graph.

No validation smoothing, metric change, early stopping, checkpoint averaging,
EMA validation, confidence filtering, posterior target, or post-processing
change is authorized.

## Logging and falsification

Record the inherited measurement loss and coefficient separately from the
native auxiliary loss and `mu(t)`. The implementation must assert that
`0 <= mu(t) <= 0.5 * lambda(t)` and must keep the parent measurement
coefficient byte-explicit in the objective.

The internal optimization target is Dice >= 0.85. The experiment is rejected
as an optimization successor if it does not improve the unchanged validation
selection result over matched SliceEqOcc, or if any apparent improvement is
obtained by changing evaluation/selection rather than the locked objective.
No ratio, cutoff, or ramp rescue is authorized after observing this run.

