# External CoDA Run Analysis

Date: 2026-08-11  
Evidence class: **EXPLORATORY** (post-hoc artifact; incomplete paired-run identity)

## Archived evidence

- `results/coda_external_run_2026-08-11/self_train_log.txt`
  - SHA-256: `E59E1AE0240FEB98554D3870A7A5A54D3D30D88C9CD724D244FC462672F0F2DB`
- `results/coda_external_run_2026-08-11/test_performance.txt`
  - SHA-256: `93AF76214EE0ED5F3FDC73313FBB585C0C8C01EEC5BB081FD61F5171ADBD294A`

The self-training log does not contain the argument namespace, pretraining
trajectory, checkpoint hash, split hashes, or TensorBoard diagnostic series.
The test file identifies the expected CoDA experiment path and ten test cases,
but does not record whether strict checkpoint loading succeeded.

## Direct observations

- The self-training stage completed 30,000 iterations with 15 iterations per
  epoch and 150 validation evaluations (one every 200 iterations). All reported
  validation values are finite.
- Best validation Dice: **0.804897 at iteration 27,200**.
- Final validation Dice: **0.797537 at iteration 30,000**.
- Best-checkpoint selection advantage over the final iterate: **+0.007360**.
- Validation Dice over iterations 25,200--30,000: **0.794807 mean**, **0.004774
  sample SD**, range **0.780180--0.804897**. The final 20 evaluations have mean
  0.794519 and sample SD 0.004993.
- Test Dice/Jaccard: **0.819876 / 0.697646** over ten cases. The case-level Dice
  sample SD is 0.051567, with range 0.746351--0.898327.
- Reported HD95/ASD: 7.952128 / 2.305603. Case34, Case09, and Case05 contribute
  approximately 74% of the summed HD95, so surface error is dominated by a small
  long-tail subset.
- The user reports a baseline final Dice range of **0.78--0.80**. Relative to
  that non-paired range, this CoDA result is **+0.019876 to +0.039876 absolute
  Dice** (about +1.99 to +3.99 percentage points).

## What this run does and does not establish

This is a positive exploratory result for the **complete current CoDA recipe**.
It is not evidence that augmentation-conditioned target entropy is the causal
mechanism. The implementation changes four factors together relative to the
locked baseline: the student's unlabeled view, hard versus probability-preserving
LCC targets, hard versus soft CE/Dice, and interpolation toward a uniform target.
Consequently, even `gamma=0` is not the baseline.

The text log omits the quantities needed to test the proposed mechanism:
`coda/evidence_loss_mean`, sampled family and severity, target entropy,
consistency loss, and consistency weight. It also does not separate resolution
and noise runs. The result could therefore arise from strong-view regularization,
soft pseudo-labels, BatchNorm coupling between labeled and strong-unlabeled
samples, uniform smoothing, or an interaction among them.

The baseline range is not accompanied by the same ten per-case values, seed,
pretraining checkpoint hash, or model-selection trace. A paired patient-level
test and a multi-seed method comparison are therefore impossible. The run also
uses a five-volume validation set and selects the maximum across 150 evaluations;
the 0.736-point gap between best and final validation Dice quantifies material
selection sensitivity.

Only Dice and Jaccard should currently be treated as interpretable legacy
metrics. `test_coda.py` calls MedPy HD95/ASD without voxel spacing, so those
distances are in array-index units rather than physical millimetres. Empty
prediction cases are also assigned zero distance, although that branch was not
triggered for the ten nonzero-Dice cases in this artifact.

## Outer-loop decision

1. Reclassify H4 from an undocumented small-gain report to **exploratory positive,
   mechanism unresolved**.
2. Do not revive CoDA as the headline contribution from this run alone. Retain
   the complete CoDA recipe as a strong empirical comparator for BMER.
3. The next minimum comparison is a same-split, same-pretraining-checkpoint,
   same-seed baseline producing the identical ten per-case rows. Then report
   paired case differences and repeat both methods for at least three seeds.
4. If CoDA mechanism work is resumed, run the locked factor chain: same-view hard
   target; strong-view hard target; strong-view LCC-soft target with `gamma=0`;
   and the complete current coupling. Resolution and noise must be separated.
5. Export the TensorBoard event directory before interpreting evidence loss or
   target entropy. Without it, H4.1 remains untested.
6. Keep BMER as the primary direction. This run changes the empirical comparator,
   not the novelty analysis or the mandatory H5.1/H5.2 kill gates.

