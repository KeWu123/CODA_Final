# H6 OBA External Run Analysis

Date: 2026-08-11  
Evidence class: **EXPLORATORY, SINGLE FIXED SEED**  
Decision: **MEAN-NEUTRAL VERSUS CODA; UNSTABLE; ONE DECISIVE CONTROL BEFORE GO/NO-GO**

## Provenance

- `results/oba_external_run_2026-08-11/pre_train_log.txt`
  - SHA-256: `BF2EA62518E5963B543AE70B7F62A7109225F6F3BA7A97DA5F17D7EE8055A899`
- `results/oba_external_run_2026-08-11/self_train_log.txt`
  - SHA-256: `CA7B24D7234BFDC5B9986D8972302669792BB4EFF9B0A09163F642932776829B`
- `results/oba_external_run_2026-08-11/test_performance.txt`
  - SHA-256: `4130D4F3523C595D2586DFBBA257940BBA0362DB907EDF3E2C7E22390EAB44A5`

The archived files are byte-identical to the supplied `Z:/Downloads` artifacts. The
logs record the expected OBA namespace, seed 1337, 10,000-step pretraining, 30,000-step
self-training, loader batch 24, and post-warmup effective student batch 36. The
checkpoint hash, split hashes, TensorBoard event file, environment, and hardware remain
unavailable.

## Direct observations

- Supervised pretraining itself is highly selection-sensitive: best validation Dice is
  `0.679773` at 7.8k, while the 10k final value is `0.570292`, a `0.109481` gap. The
  selected pretraining checkpoint is used by self-training. Because CoDA/BMER rerun the
  same deterministic pretraining code but do not have archived pretraining logs or
  checkpoint hashes, shared initialization is expected from code but not proven.
- Training completed all 30,000 iterations, with 150 finite validation evaluations.
- Best validation Dice was **0.793850 at iteration 13,800**.
- Final validation Dice was **0.699411**, producing an unusually large
  **0.094439 best-to-final gap**.
- Validation windows expose progressive degradation rather than stationary noise:
  - 10.0k--15.0k: `0.785966 +/- 0.004729`;
  - 15.2k--20.0k: `0.772288 +/- 0.013710`;
  - 20.2k--25.0k: `0.743147 +/- 0.010786`;
  - 25.2k--30.0k: `0.717645 +/- 0.014415`.
- The baseline consistency weight rises from approximately `0.116` at the selected
  OBA checkpoint to `0.285` at 20k, `0.433` at 25k, and `0.500` at 30k. This temporal
  alignment is not causal proof, but it is consistent with the paired hard-target
  pressure becoming harmful as its weight grows.
- The selected checkpoint reaches test Dice/Jaccard **0.818872 / 0.694761**. Relative
  to the user's unpaired baseline range `0.78--0.80`, this is an apparent
  `+0.018872 to +0.038872` Dice improvement.

## Paired comparison with CoDA

CoDA reaches test Dice `0.819876`; OBA differs by only **-0.001004**. The equality of
the means hides a highly nonuniform paired pattern:

| Case | OBA Dice | CoDA Dice | OBA - CoDA |
|---|---:|---:|---:|
| Case05 | 0.800276 | 0.748922 | +0.051354 |
| Case09 | 0.766989 | 0.781444 | -0.014455 |
| Case16 | 0.823028 | 0.850497 | -0.027469 |
| Case30 | 0.841397 | 0.850162 | -0.008765 |
| Case34 | 0.830188 | 0.746351 | +0.083837 |
| Case36 | 0.858637 | 0.845228 | +0.013409 |
| Case38 | 0.877468 | 0.898327 | -0.020859 |
| Case43 | 0.831751 | 0.872494 | -0.040743 |
| Case45 | 0.773283 | 0.801710 | -0.028427 |
| Case49 | 0.785699 | 0.803623 | -0.017924 |

- OBA wins only **3/10** cases; median paired Dice difference is **-0.016190**.
- Excluding the two large OBA gains on Case05 and Case34, the remaining eight-case
  mean difference is **-0.018154**.
- An exact two-sided paired sign-flip test on the ten Dice differences gives
  `p=0.9434`; this tiny sample supplies no evidence of a mean advantage.
- OBA's gain is negatively correlated with CoDA case Dice (`r=-0.707`, exploratory),
  suggesting that OBA reallocates performance toward CoDA's hard cases.

## Tail signal and surface metrics

OBA has a potentially useful but currently underpowered tail-risk signal:

- case-level Dice SD falls from `0.051567` (CoDA) to `0.036590` (OBA);
- minimum Dice rises from `0.746351` to `0.766989`;
- bottom-three mean Dice rises from `0.758906` to `0.775324`;
- maximum legacy HD95 falls from `32.015621` to `16.431677`;
- maximum legacy ASD falls from `7.935953` to `3.833853`.

However, mean HD95/ASD improvement is dominated by Case34. Median paired HD95 is
actually `+0.158313` worse for OBA. The evaluator does not pass voxel spacing to MedPy,
so these distances are array-index units, not physical millimetres. With ten cases and
one selected checkpoint, the tail pattern is a hypothesis, not a contribution.

## Mechanistic interpretation

The current OBA objective averages two endpoint losses,
`0.5[L(p_plus,y_hat)+L(p_minus,y_hat)]`. Antithetic sampling can cancel odd-order
variation along the sampled augmentation coordinate, but it cannot cancel:

1. errors in the clean teacher's hard LCC pseudo-label;
2. even-order curvature that penalizes both transformed endpoints;
3. confirmation bias amplified by the increasing consistency weight; or
4. the change in BatchNorm mixture from `12L+12U` to `12L+12U+ +12U-`.

The late collapse is therefore compatible with early useful robustness pressure being
overtaken by symmetric overfitting to hard pseudo-label errors. This explanation is
stronger than attributing the run to generic optimization noise, but the missing
TensorBoard OBA diagnostics prevent directly relating pair disagreement or probability
gap to the decline.

The nearest top-conference controls remain decisive. UniMatch (CVPR 2023) already uses
two strong student streams guided by one common weak view, so using two views and a
shared pseudo-label is not novel. OBA's only defensible increment is the antithetic
coordinate design. Strong-augmentation BatchNorm damage has also been explicitly
identified by the ICCV 2021 strong-augmentation baseline. Averaging predictions over
augmentations is established by MixMatch (NeurIPS 2019), so moving the loss to a simple
prediction barycenter is a diagnostic variant, not a new headline by itself.

## Go/no-go decision

Do **not** promote current OBA as the paper method and do not tune its severity ranges,
family probabilities, consistency ramp, or checkpoint duration from this curve.

The user's later report that CoDA pretraining reaches about `0.71` and baseline
pretraining about `0.73` introduces a fairness blocker. OBA pretraining is only
`0.679773`, even though OBA literally calls the same `train_coda.pre_train` function and
does not apply its method during pretraining. The initialization gap is 30--50 times
larger than the `0.001004` OBA--CoDA test gap. It can alter initial student quality,
teacher LCC masks, SGD momentum, confirmation bias, and the late collapse. Current
method comparisons are therefore unpaired in initialization as well as in method.

Authorize one bounded deepening cycle, with shared pretraining now first:

1. Lock one canonical pretraining checkpoint and optimizer state by SHA-256, reset the
   stage RNG identically, and reuse it for CoDA, OBA, and the IID control. If the
   baseline 0.73 checkpoint has adequate provenance, it is the natural locked baseline
   anchor; otherwise generate one fresh checkpoint once and do not select among reruns.
2. Export the existing TensorBoard event file and test whether pair prediction
   disagreement/probability gap rise with the validation collapse. This is read-only
   analysis, not a new run.
3. Run exactly one full, compute-matched **two-IID-view control** using the same three
   families, per-sample severity marginals, effective batch 36, clean teacher, hard LCC
   target, loss, schedule, seed, and pretraining checkpoint. Only replace `+a/-a` by
   two independent signed coordinates.
4. Reject H6 if IID matches OBA within 0.005 Dice, has a materially more stable late
   validation curve, or reproduces the hard-case tail improvement. This would show that
   extra/diverse views, not orbit balance, explain the result.
5. Only if OBA beats IID and H6.1 shows negative paired displacement should a
   probability-barycenter objective be tested as a bounded stability correction.

This is a **DEEPEN once** decision, not a new full research direction. Failure of the
IID control ends OBA and triggers a clean pivot; success would justify mechanism
ablations and a second benchmark.

## Literature checked for this reflection

- [UniMatch, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)
- [A Simple Baseline for Semi-Supervised Semantic Segmentation With Strong Data Augmentation, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_A_Simple_Baseline_for_Semi-Supervised_Semantic_Segmentation_With_Strong_Data_Augmentation_ICCV_2021_paper.html)
- [MixMatch, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html)
- [U2PL, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Semi-Supervised_Semantic_Segmentation_Using_Unreliable_Pseudo-Labels_CVPR_2022_paper.html)
