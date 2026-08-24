# H7.11 SliceEqOcc-ADU implementation protocol

## Status

Locked on 2026-08-14 after the exploratory-positive H7.10
operator-reliability screen and before full training. The uploaded H7.10 run
predates final analyzer hardening, so it authorizes one exploratory full run,
not a final-hash confirmatory claim. The implementation is intentionally a
parallel successor; no SliceEqOcc parent source or validation code is edited.

## Hypothesis

Fractional occupancy created by the SliceEq acquisition operator is useful
measurement supervision, but a train-mode stochastic teacher can also create
reducible pseudo-mask variation. Entropy of one occupancy cannot distinguish
these two effects. Disagreement between two hard-LCC pseudo occupancies after
the same acquisition operator provides an operational proxy that ranks target
error while leaving an agreed modeled acquisition-derived fractional target
at full weight, even when its entropy is high. It is not an identifiable
uncertainty decomposition.

H7.10 supplies the mechanism evidence. Across 18k/24k/30k student-proxy
checkpoints, operator-space JS has patient-balanced error correlation
0.61--0.68, top-20 error concentration 2.26--2.69, normalized weighted-risk
reduction 5.26%--7.34%, and exact fractional-support weight retention
0.963--0.965. The risk reduction means that the same `q_bar` error map has
lower reliability-weighted than unweighted risk; it is not evidence that
`q_bar` itself has lower unweighted Brier error. A post-run conservative
reaggregation of the raw pair records gives seven of seven named patients.

## One bounded ADU package

Relative to SliceEqOcc, ADU couples two changes: the one-pass occupancy target
is replaced by a two-pass mean target `q_bar`, and JS-derived reliability
weights both loss terms. The first exploratory run evaluates this bounded
package. A positive result cannot be attributed specifically to reliability
weighting until a compute-matched `q_bar`-only (`w=1`) control is run.

After the unchanged 1k warmup:

1. Run the original train-mode EMA teacher once on the flattened `B x 3`
   unlabeled stack. Preserve this forward's persistent BN update and CUDA RNG
   consumption exactly as in SliceEqOcc.
2. Run one additional train-mode stochastic teacher forward on the identical
   stack. It uses the fixed independent seed `seed + 7000003 + iteration`
   inside a forked target-GPU RNG scope. Restore all EMA buffers to their state
   immediately after the primary forward. The extra pass therefore has no
   persistent BN/RNG effect on the parent path.
3. Apply hard argmax and the existing per-slice 2-D LCC independently to both
   teacher outputs. Do not average logits, probabilities, or masks before LCC.
4. Sample exactly one parent profile `a` and apply it to both hard pseudo
   stacks: `q1=A_a(M1)` and `q2=A_a(M2)`. The student input remains the one
   parent reacquired image `A_a(X)`.
5. Define `q_bar=(q1+q2)/2`,
   `JS=H(q_bar)-0.5*(H(q1)+H(q2))`, and
   `w=clamp(1-JS/log(2),0,1)`. Detach all target and reliability tensors.

No confidence threshold, temperature, weight floor, loss coefficient,
schedule, extra head, new parameter, patient rule, or inference branch is
allowed.

## Objective

The complete parent supervised objective and full acquisition coefficient are
unchanged:

`L = L_sup + lambda(t) * L_U_ADU`.

For U logits `z`, probabilities `p`, target `q_bar`, and pixel weight `w`:

`CE_w = sum_i w_i[-sum_c q_ic log softmax(z)_ic] / sum_i w_i`.

`Dice_w = mean_c[1 - (2 sum_i w_i p_ic q_ic + eps) /
(sum_i w_i p_ic^2 + sum_i w_i q_ic^2 + eps)]`.

`L_U_ADU = 0.5*(CE_w + Dice_w)`.

When `w=1`, the loss and gradient must reproduce the parent
`soft_segmentation_loss`. An all-zero reliability map produces a differentiable
zero loss rather than a NaN.

## Frozen parent contract

- root path, label budget 7, and first 191 labeled slices;
- fixed shared Pre10000 network and optimizer checkpoint, SHA-256
  `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`;
- seed 1337, SGD, EMA 0.99, 30k iterations, and 1k warmup;
- loader batch 24 and unchanged student batch36:
  12 original-L + 12 reacquired-L + 12 reacquired-U;
- exact-GT labeled occupancy branch and original hard labeled anchor;
- radius 1, sigma `[0.45,0.85]`, phase `[-0.25,0.25]`;
- one optimizer update and one EMA update per iteration;
- EMA remains in train mode, matching the user's baseline policy;
- unchanged 2-D inference and no prediction saving by default.

The extra teacher forward changes teacher compute only. Student/backprop views,
student BN composition, model parameters, optimizer, and inference cost remain
identical to SliceEqOcc. Total network-forward views per post-warmup update rise
from 72 to 108.

## Evaluation contract

The validation block is copied verbatim from `train_sliceeq_occ.py`:

- five-case validation loader and `val_2d.test_single_volume`;
- every 200 iterations;
- identical mean Dice computation;
- strict `performance > best_performance` selection;
- raw student checkpoint;
- periodic checkpoint every 3000 iterations.

No validation smoothing, EMA validation, changed metric, early-stop rule,
checkpoint averaging, or manual periodic-checkpoint selection is introduced.
The selected best checkpoint is tested once with the strict ADU test entry,
whether the exploratory validation result is positive or negative. Test
performance cannot select another checkpoint, change the method decision, or
authorize a rescue variant.

The matched parent is the SliceEqOcc run whose training log SHA-256 is
`93fe37c576e0ca575e25f2938086d1842657faf9b9a1647b68672befa2be8442`;
its best validation Dice is `0.817373`. H7.11 passes the exploratory
optimization gate only if its best validation Dice is at least `0.820373`.
The user-facing numerical aim of 0.85 test Dice remains aspirational, not a
test-driven selection rule.

## Diagnostics and falsification

Log JS mean/max/activity, mean reliability, normalized ESS,
pseudo-fractional-support weight, occupancy absolute/hard disagreement, and the inherited occupancy
diagnostics. These values are observables only and cannot tune the run.

Reject H7.11 if the full run does not improve the locked validation criterion,
if the isolated second forward changes persistent EMA buffers or parent RNG,
or if any gain requires manual checkpoint selection. Do not rescue a negative
result with JS thresholds, temperatures, more MC passes, or another schedule.

## Publication boundary

The main contribution remains paired through-plane re-acquisition and
fractional occupancy. ADU is a secondary reliability component. Safe wording:
we measure dropout-induced disagreement after the same non-invertible modeled
acquisition operator, using it as an operational reliability proxy that leaves
agreed acquisition-derived mixture unpenalized. Do not claim first MC dropout,
uncertainty weighting, JS reliability, calibration, or a strict
aleatoric/epistemic decomposition. If H7.11 is positive, the `q_bar`-only
control is mandatory before assigning an improvement to reliability weighting.
