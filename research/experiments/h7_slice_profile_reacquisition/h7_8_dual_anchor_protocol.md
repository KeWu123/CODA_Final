# H7.8 Native-measurement dual-anchor protocol

## Status

Implementation locked before external training. This is an exploratory
optimization experiment built on the selected SliceEqOcc method.

## Motivation

SliceEqOcc gives the labeled branch both a native semantic anchor and a paired
re-acquired fractional-occupancy target. Its unlabeled branch, however, keeps
only the re-acquired occupancy target although deployment uses a native center
slice. CAP showed that spending the fourth student view on another measurement
sample did not help: it produced nearly duplicate occupancy supervision and
reduced Dice. H7.8 instead spends the same fourth-view budget on the missing
native unlabeled state.

## Locked intervention

For an unlabeled stack and its train-mode EMA LCC pseudo stack:

```text
L_U_measurement = soft_loss(f(A_h X_U), A_h onehot(Yhat_U))
L_U_native      = hard_loss(f(X_U_center), Yhat_U_center)
L_U             = 0.5 * (L_U_measurement + L_U_native)
```

The factor `0.5` keeps the inherited total unlabeled coefficient fixed. The
student batch is 48 views, matching CAP's compute shape. The EMA teacher is not
put in eval mode, so teacher BN/dropout behavior remains matched to the baseline
and earlier SliceEq experiments.

## Frozen factors

- PROMISE12 split and seven labeled cases;
- seed 1337 and shared net+optimizer pretraining checkpoint;
- U-Net and train-mode EMA teacher, decay 0.99;
- 30,000 self-training iterations and 1,000-step identity warmup;
- loader batch 24, labeled batch 12;
- SliceEq radius, sigma, phase, RNG streams, and paired operator;
- labeled SliceEqOcc objective;
- consistency ramp and total unlabeled weight;
- validation schedule, checkpoint rule, and inference graph.

No confidence filtering, posterior target, curriculum, EMA-mode change, or
post-processing change is introduced.

## Prediction and decision

Primary prediction: restoring native-U self-training improves native-grid
validation/test segmentation while preserving the acquisition-specific gains
of fractional occupancy. The internal optimization target is Dice >= 0.85, but
the run is interpreted by its locked validation rule and paired case behavior,
not by selecting a checkpoint after repeated test queries.

If H7.8 is positive, the next causal control is a 48-view duplicate/re-acquired
sham to separate native-state complementarity from batch/BN and compute. If it
is neutral or negative, do not tune the dual-anchor ratio immediately; close
the native-anchor branch and move to metadata-grounded profile calibration.

