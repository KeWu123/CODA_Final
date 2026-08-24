# SliceEqOcc after APTNA: locked next step

APTNA is negative under the unchanged validation rule. The native-anchor family is closed; do not
run another ratio, ramp, cutoff, or native-view variant.

The next action is one zero-training, labeled-only H7.10 gate. It tests whether the train-mode EMA
teacher's independent dropout noise across a three-slice stack is being converted into false
fractional occupancy, and whether operator-space disagreement predicts target error. It does not
change validation, inspect test labels, or start another 30k run.

Decision order:

1. Run the joint SCT/ADU gate on seven labeled training patients and three parent checkpoints.
2. Train exactly one candidate only if it passes its preregistered exact-GT fidelity thresholds.
3. Keep the parent student batch36, EMA train mode, profile/loss where applicable, validation, and
   2-D inference fixed.
4. If neither passes, stop small extensions and build the CVPR causal/multi-seed/external evidence;
   a larger numeric gain would then require an explicitly broader SSL scaffold or backbone change.

A fixed checkpoint average or learning-rate schedule may be evaluated symmetrically as a training
recipe, but it is not the next method contribution and cannot be selected using test performance.

