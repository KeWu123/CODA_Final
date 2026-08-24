# H5 BMER External Run Analysis

Date: 2026-08-11  
Evidence class: **EXPLORATORY, NEGATIVE FOR THE CURRENT IMPLEMENTATION**  
Protocol deviation: Stage A/A2 and the B0--B7 short screen were not run before a
nominal 30k training launch; the supplied run itself stopped at iteration 20,800.

## Provenance

- `results/bmer_external_run_2026-08-11/self_train_log.txt`
  - SHA-256: `2D54933087190132EDAFF2C562E9E99E972A9FFDA9457237273260C65FBD1ED3`
- `results/bmer_external_run_2026-08-11/test_performance.txt`
  - SHA-256: `6834E381B57DFB7050BD01E50E00276762D62907F0F0FE01B86825BE8C75FAD6`

The log records the complete argument namespace but not the pretraining-checkpoint
hash, list hashes, TensorBoard diagnostics, environment versions, or hardware. The
run used the locked defaults: radius 8, 16 sectors, 3 position bins, probability 0.5,
strength 0.5--1.0, seed 1337, and 7 labeled patients.

## Direct observations

- The run stopped at iteration 20,800 rather than the declared 30,000 iterations.
- Best validation Dice was **0.788501 at iteration 15,400**. The last reported value
  was **0.767375**, a decline of **0.021126** from the selected maximum.
- Window means show a plateau followed by degradation:
  - 10,200--15,000: 0.777564 +/- 0.004845;
  - 15,200--20,000: 0.773047 +/- 0.007961;
  - 20,200--20,800: 0.767551 +/- 0.002663.
- Across the 28 validations from the selected best point onward, mean Dice was
  0.771930 +/- 0.007557. The best point is an isolated selection spike rather than a
  stable terminal plateau.
- Test Dice/Jaccard were **0.795949 / 0.664106**. Relative to the user's unpaired
  baseline range 0.78--0.80, the result is -0.004051 to +0.015949 and therefore does
  not establish improvement beyond the known baseline range.
- On the same ten test cases, BMER is **-0.023927 mean Dice** versus the archived
  CoDA combination. Nine of ten cases lose Dice. Case34 is the sole overlap-metric
  improvement (+0.080808), but its legacy HD95 increases from 32.02 to 107.02,
  consistent with a severe distant false-positive component under NMS-off inference.
- Legacy mean HD95 increases from 7.95 for CoDA to 16.62 for BMER. These are index-unit
  distances because spacing is not passed to MedPy, but the symmetric implementation
  still exposes a large relative surface-failure regression.
- Building the frozen bank accepted 371 of 749 unlabeled slices and skipped 378.
  Accepted counts were 94/209/68 across the three longitudinal bins. The skip fraction
  cannot be interpreted as teacher failure without distinguishing true background-only
  slices from missed prostates; that distinction was not logged.
- BMER processed 20.8k iterations in approximately the same wall time that the
  archived CoDA run processed 30k, corresponding to roughly 30% lower throughput
  (about 43% more time per step). The CPU distance-transform/render loop is not free.

## A concrete implementation confound

The frozen donor bank is extracted from unaugmented slices. In the training loader,
however, labeled recipient images and masks first undergo random rotations/flips and
then BMER is applied. Both donor and recipient fields use 16 sectors defined by the
absolute image-plane angle around the mask centroid. A rotation or flip therefore
changes the anatomical meaning of the recipient sector without applying the same
transformation to the donor field. The claimed tangential-position conditioning is
frequently misaligned.

This is a specific, single-factor implementation problem, not a reason to tune radius,
probability, or strength. It prevents the current run from cleanly falsifying the
abstract empirical-boundary-evidence hypothesis. Other core risks remain unresolved:
the Stage-A oracle effect was never shown, and teacher/GT profile fidelity was never
measured.

## Go/no-go decision

### Stop

- Do not continue tuning the current renderer.
- Do not launch another 30k run with altered radius, probability, strength, bank size,
  confidence filtering, losses, or teacher behavior.
- Do not interpret a possible late recovery as BMER evidence: the incomplete schedule
  and rising baseline consistency weight can produce late gains unrelated to BMER.

### Authorize exactly one diagnostic correction cycle

1. Run a no-training rotation/flip equivariance test for the renderer. Apply a known
   geometric transform before and after resynthesis and quantify the commutation error.
2. Run the pre-registered Stage-A oracle intervention against matched scalar contrast
   and blur. Retain the original numerical gates; do not relax them.
3. Run Stage A2 on held-out labeled cases to separate true empty slices from teacher
   misses and measure GT/pseudo profile rank fidelity.
4. Only if all gates pass, make sector alignment equivariant by applying BMER before
   the shared geometric transform or by transforming donor fields with the same
   recorded transform. Then run only the shared-pretrain 2--3k B0/B4/B6/B7 screen.

If the corrected BMER does not beat the matched local-simple B4 control in both region
and boundary metrics, or if either Stage-A gate fails, mark H5 rejected and pivot. No
second rescue cycle is justified.

## Direction decision

The current implementation is **NO-GO**. The underlying BMER hypothesis receives one
short diagnostic opportunity because sector misalignment is concrete and falsifiable,
not because the training curve looks noisy. Given the negative paired pattern, missing
mechanism evidence, added training cost, and the 2026 FDIF collision on distance-driven
appearance synthesis, the prior probability of a CVPR-level rescue is low. Allocate at
most one working day to the diagnostic cycle; otherwise pivot to a newly preregistered
acquisition-model direction rather than stacking components onto BMER.

