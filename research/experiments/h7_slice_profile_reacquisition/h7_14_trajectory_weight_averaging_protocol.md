# H7.14: OAAC trajectory weight averaging

Status: protocol specified; no result yet  
Date: 2026-08-17

## Hypothesis

The completed OAAC run explores one high-performing but noisy constant-LR
basin. An equal average of three evenly spaced late iterates, followed by
BatchNorm recalibration on the unchanged OAAC student-view distribution, can
improve validation and development Dice without changing training targets,
network architecture, parameter count, or inference cost.

## Evidence motivating the gate

- OAAC uses constant SGD learning rate 0.01 through 30k.
- Validation best/final are 0.834863@23.8k and 0.831964@30k.
- The retained periodic checkpoints have validation Dice
  0.828298@24k, 0.828406@27k, and 0.831964@30k.
- The ten-case test score 0.849538 is a post-hoc test-selected oracle at 27k;
  the current test split is development-only.

## Stage A: one fixed zero-training candidate

Average exactly these raw-student checkpoints with equal coefficient 1/3:

```text
iter_24000.pth
iter_27000.pth
iter_30000.pth
```

No alternate window, greedy soup, interpolation coefficient, validation-ranked
subset, EMA checkpoint, or test-driven constituent replacement is allowed.

For trainable floating tensors, compute the arithmetic mean in float64 and
cast back to the original dtype. Copy non-BatchNorm non-floating buffers from
30k. Do not use averaged BatchNorm running statistics for evaluation.

## BatchNorm recalibration

Recompute running means/variances once, without gradients, using the original
training-only data and the same post-warmup 36-view student distribution:

```text
12 original labeled centers
+ 12 paired re-acquired labeled images
+ 12 OAAC-transformed paired re-acquired unlabeled images
```

Use the locked PROMISE12 training split and seeds 1337/1338/1339. Put only
BatchNorm modules in training mode; keep Dropout disabled. Reset BN counters
before the pass. Do not access validation or test images during calibration.

## Evaluation and stopping rule

1. Run the existing validation implementation unchanged.
2. If validation Dice is not strictly greater than the OAAC best 0.834863,
   stop and do not test the averaged checkpoint.
3. If validation improves, test this single fixed averaged checkpoint once.
4. Regardless of its test value, classify it as development-only because the
   same ten-case split was previously queried at multiple checkpoints.

## Stage B: prospective confirmation if Stage A is positive

On a fresh matched OAAC and SliceEqOcc run, maintain an inference-only equal
average of raw student weights every 200 updates from iteration 22,600 through
30,000 (the last quarter, aligned to the existing validation interval). This
average must not feed the teacher or the optimizer. Recalibrate BN once at the
end and use the final averaged model as the preregistered readout; no checkpoint
search is involved.

Apply the same readout to SliceEqOcc and OAAC. SWA/TWA is an optimization
recipe, not a SliceEq/OAAC contribution.

## If trajectory averaging is neutral

Do not sweep soup windows. The next and only remaining no-architecture training
candidate is a matched fresh rerun with standard polynomial LR decay
`0.01*(1-t/30000)^0.9`, separately for SliceEqOcc and OAAC. Do not combine
PolyLR with SWA in its first run. EMA decay, teacher mode, target, augmentation
range, loss, batch, validation, and inference remain fixed.

## Publication boundary

Even a development Dice above 0.85 cannot enter the primary table. The paper
requires a frozen selector/readout on fresh hidden or external data, plus
matched multi-seed controls. OAAC remains the method contribution; weight
averaging or PolyLR is reported only as a shared training recipe.
