# SliceEq fixed-seed result analysis

## Decision

The run is a successful **direction-selection** result and supports retaining SliceEq as the
primary research direction. It is not yet evidence that the proposed paired image/occupancy
mechanism caused the gain, and it does not meet the preregistered SOTA-facing continuation
threshold.

## Observed result

| Method/run | Mean Dice | Mean Jaccard | HD95 | ASD |
|---|---:|---:|---:|---:|
| SliceEq | 0.832603 | 0.715429 | 4.548882 | 1.746296 |
| Archived CoDA | 0.819876 | 0.697646 | 7.952128 | 2.305603 |
| Archived OBA | 0.818872 | 0.694761 | 5.943047 | 1.994647 |

SliceEq is +0.012728 Dice over CoDA and +0.013732 over OBA. Those archived methods were run
before the shared-checkpoint amendment, so these are useful exploratory comparisons rather
than controlled method effects. The user-reported UniMatch value of 0.832233 is essentially
tied with SliceEq (+0.000370), and no UniMatch artifact has been imported.

Validation improves by training phase rather than undergoing OBA's sustained collapse:

| Iteration range | Mean validation Dice | SD | Maximum |
|---|---:|---:|---:|
| 0--10k | 0.763548 | 0.014513 | 0.778914 |
| 10--20k | 0.785795 | 0.010492 | 0.809343 |
| 20--30k | 0.795354 | 0.009230 | 0.811287 |

The best validation score is 0.811287 at 24.8k; the 30k score is 0.788273. The 0.023014
best-to-final gap and five-case validation set still imply substantial checkpoint-selection
noise.

## Case heterogeneity

Against CoDA, SliceEq wins only 4/10 cases. Its paired median Dice change is -0.018262, and
removing the three largest gains (Case05, Case09, Case34) changes the remaining-case mean to
-0.021907. Against OBA it wins 5/10 cases, has median change +0.003500, and the mean after
removing the three largest gains is -0.005672. The mean improvement is therefore concentrated
rather than population-wide.

This concentration is not automatically a failure: a slice-profile method should have a
larger effect on cases with particular through-plane acquisition or apex/base geometry. It
becomes supporting evidence only if Case05/09/34 share preregistered acquisition properties;
without spacing/thickness/vendor or slice-stratified metadata it remains an unexplained tail
effect.

The average HD95 and ASD are lower, but legacy distance metrics are computed without physical
voxel spacing and their mean improvements are influenced by large CoDA/OBA outliers. They
cannot yet support a physical-boundary claim.

## Mechanism audit

For the implemented three-tap Gaussian range, the central profile weight is approximately
0.485--0.855 and is below 0.5 for only about 3.5% of the uniformly sampled sigma/phase
parameter area. With a binary hard target and `argmax` after occupancy averaging, the central
mask therefore usually survives unchanged. A hard target can flip only in a narrow parameter
region and when the two neighbors jointly outweigh and disagree with the center.

Consequently, SliceEq v1 may behave mainly as real-neighbor slab image augmentation while its
claimed paired target branch is nearly inactive. The missing TensorBoard
`target_changed_fraction` trace is the direct check. Increasing sigma merely to force hard
label changes is not recommended because it increases anatomical mismatch.

There is a second attribution risk: the inherited EMA teacher remains in train mode, and
SliceEq forwards all three neighboring slices through it. This changes teacher BN/dropout
behavior relative to a center-only baseline. It must be neutralized or exposed in a matched
control before a paper claims that the acquisition operator itself caused the gain.

## Optimization selected for the next method version

The most principled extension is **fractional-occupancy SliceEq**, not a wider arbitrary
severity range:

`x_phi = sum_k h_phi(k) X[z+k]`

`o_phi = sum_k h_phi(k) one_hot(Y[z+k])`

Use `o_phi` directly in soft cross-entropy and soft Dice instead of immediately reducing it to
`argmax(o_phi)`. Unlike CoDA's uniform smoothing, this leaves every region with agreeing
neighbor masks exactly one-hot; softness appears only where the sampled acquisition genuinely
mixes different tissue occupancy. Thus it activates the paired target mechanism without
injecting foreground probability throughout the background.

For the complete paper-facing version, retain the original hard central labeled anchor and add
the same paired operator to a labeled view using exact neighboring GT masks. A normalized
original/re-acquired supervised loss provides clean occupancy supervision, while the
unlabeled branch uses detached LCC pseudo-mask occupancy. The network and inference graph
remain unchanged. If memory permits, an effective student batch of 36 can hold 12 original
labeled, 12 re-acquired labeled, and 12 re-acquired unlabeled slices; the loss terms must be
normalized so the gain is not caused by a larger supervised gradient.

Profile parameters should ultimately be expressed in physical millimetres and conditioned on
source slice spacing/thickness, because a universal sigma in slice units represents different
physical slabs across PROMISE12 sites. This is secondary to occupancy preservation and should
not be approximated with a learned selector if metadata are absent.

## Required gates

1. Recover the existing TensorBoard event and report target-change fraction, occupancy entropy,
   foreground-volume change, center weight, and endpoint clamping. This requires no retraining.
2. Run the locked baseline from the identical checkpoint hash and seed. CoDA/OBA are not a
   substitute for this causal control.
3. For the next full run, log soft occupancy entropy and its boundary distance. Reject the
   mechanism if nonzero occupancy is not boundary/apex/base localized.
4. Stratify Case05/09/34 versus the remaining cases by z-spacing, thickness/vendor if present,
   and apex/mid/base Dice. Reject the acquisition explanation if no interaction appears.
5. Do not claim SOTA from 0.832603. It is 0.003797 below PSC, 0.004897 below beta-FFT, 0.025397
   below the closest top-conference PMPC result, and 0.042397 below the broader journal ceiling,
   subject to protocol comparability.

