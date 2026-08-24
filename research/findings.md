# Research Findings

## Research Question

Within the user's locked PROMISE12 split and BCP-derived EMA baseline, can the local evidence loss produced by a sampled strong augmentation determine the corresponding dense pseudo-target uncertainty, improving segmentation and calibration without changing the data protocol?

## Outer-Loop History

The first idea, ViSA-MT, separated augmentation validity from student difficulty. It was literature-grounded but remained a multi-component synthesis and collided increasingly with reliability-guided mixing and pre/post augmentation regional supervision such as SGRS-Net. It is retained only as an optional comparison.

The second idea, CoDA-MT, negated a hidden weak-to-strong assumption: semantic identity can remain unchanged while blur, noise, downsampling, or masking removes evidence, so the strong view may not deserve an equally sharp pseudo-target.

The third outer loop audited the actual code in `E:/Desktop/Baseline` and constrained CoDA to a minimal, implementation-compatible form.

## What the Baseline Actually Is

The code should be described as **EMA hard-pseudo-label self-training derived from BCP**, not BCP and not yet canonical weak-to-strong Mean Teacher.

In `self_train`:

- the student forwards the complete batch;
- the EMA teacher forwards the unlabeled tail of that same tensor;
- there is no separate weak and strong view;
- teacher probabilities undergo argmax and per-slice largest-connected-component filtering;
- a one-hot `ema_output_soft` tensor is computed but unused;
- after a 1,000-iteration delay, the unlabeled loss is hard CE + hard Dice.

The locked defaults are 2D U-Net, two classes, `256x256`, 10k supervised pretraining plus 30k self-training, batch `24=12 labeled+12 unlabeled`, `labelnum=7`, SGD with constant `0.01`, EMA `0.99`, and seed `1337`.

Dataset membership is supplied externally through `train_slices.list`, `val.list`, and `test.list`. These files are absent from the copied Windows folder. Their contents, ordering, labelnum mapping, and derived labeled/unlabeled index ranges are hard constraints and will not be changed.

## Final Candidate: CoDA-MT

The method changes only the unlabeled block of self-training:

1. Preserve the loader output as teacher weak view `x_w`.
2. Sample an intensity-only evidence-degrading augmentation that returns both student view `x_s` and spatial evidence-loss map `gamma`.
3. Retain the existing largest-component topology prior but preserve teacher probabilities inside it, producing `q_M`.
4. Construct

   `q_A(v) = (1 - gamma(v)) q_M(v) + gamma(v) / C`.

5. Train the student's unlabeled prediction with soft CE + soft Dice against `q_A`.

The labeled branch, pretraining, data lists, batch sampler, model, optimizer, schedule, EMA, validation/test cases, and inference remain unchanged. The method adds no parameters, cross-patient mixing, or inference cost.

## Version-1 Augmentations

Use only two coordinate-preserving families:

- resolution degradation by downsample-upsample, with `gamma` derived from local gradient-energy loss;
- Gaussian noise scaled by slice standard deviation, with `gamma` derived from a bounded local noise-to-signal ratio.

Use Rician noise only after verifying that the stored values are non-negative magnitude-MRI intensities. Avoid Copy-Paste, CutMix, Fourier style exchange, learned augmentation policies, and boundary-specific erasing in the main method.

## Novelty Boundary

The closest collision set is now explicit:

- AISTATS 2025 supervised mollification couples image degradation and label smoothing for classification.
- GeoLS (MIDL 2024) creates image-aware spatial soft labels for supervised medical segmentation.
- Enhanced Soft Label (ICCV 2023) uses dynamic soft pseudo-labels in semi-supervised segmentation.
- SGRS-Net (MICCAI 2025) partitions regional supervision using behavior before and after mix augmentation.
- DyCON and HESS already occupy uncertainty-aware consistency and improved pseudo-label uncertainty.

Therefore, the defensible claim is not "soft labels" or "uncertainty-aware augmentation." It is the remaining intersection:

> The realized, spatially varying information loss of the exact sampled strong view transforms the corresponding unlabeled dense pseudo-target.

This distinction must be demonstrated against fixed smoothing, GeoLS-style image-aware smoothing, raw teacher probabilities, and global severity coupling.

## Mandatory Ablation Chain

All variants share the exact locked split and training budget:

1. B0: current same-view + LCC hard-target baseline.
2. B1: weak teacher / strong student + LCC hard target.
3. B2: weak/strong + raw teacher soft target.
4. B3: weak/strong + fixed label smoothing.
5. B4: weak/strong + image-aware static/GeoLS-style smoothing.
6. B5: weak/strong + global severity-coupled target.
7. B6: spatial realized-evidence CoDA.

B1 is essential: without evidence that hard targets become over-certain or harmful under degradation, CoDA lacks its motivating phenomenon.

## Candidate Contributions

1. Identify augmentation-induced target overcertainty in semi-supervised dense prediction and quantify it against corruption severity.
2. Introduce a paired augmentation interface returning both a strong MRI view and its spatial realized-evidence-loss field.
3. Construct corruption-coupled dense pseudo-targets in the audited EMA baseline without additional parameters or inference cost.
4. Provide a locked mechanistic ablation separating hard targets, generic soft pseudo-labels, fixed smoothing, image-aware smoothing, global degradation coupling, and spatial realized-evidence coupling.

## Evaluation Constraints

- Retain current Dice, Jaccard, HD95, and ASD implementation for direct comparability.
- Current HD95/ASD omit voxel spacing and are voxel-index distances, not millimeters.
- Add foreground NLL, Brier score, ECE, and boundary-band ECE on the same unchanged cases.
- Multiple seeds may be run on the same split; no new PROMISE12 split or label draw is allowed.
- Do not tune on `test.list`.
- A second medical segmentation benchmark is required before a general CVPR claim, but it does not alter the locked PROMISE12 protocol.

## Reproducibility Findings

- The copied folder contains no baseline checkpoint, `log.txt`, or `performance.txt`; the available window-ablation logs belong to a different SRCLQT/SDAA method and are not baseline evidence.
- `test_baseline.py` ends in a bare `vvvv...` identifier, producing a `NameError` after results have been written. Removing that expression is a reproducibility bug fix, not a method change.
- Before experiments on Linux, record SHA-256 hashes of all three list files and the audited code files.

## Implementation Status

An isolated implementation now exists at `implementation/coda_mt_baseline`.
It preserves the Baseline directory layout and contains a source manifest.
Twenty-two copied Python files outside the two intentional entry-point edits
match the source hashes byte for byte. Contract tests also verify the original
parser defaults, labeled/unlabeled index construction, two-stream sampler, and
validation interval.

The method is confined to the copied `self_train` unlabeled branch. Its paired
augmentations use an independent RNG stream derived from the existing seed, so
augmentation sampling does not consume the Baseline's dropout/model RNG. The
first 1,000 self-training steps preserve the original student input as well as
the original zero consistency loss.

Thirteen CPU tests cover shapes, finite values, evidence-map bounds, constant
background behavior, deterministic sampling, LCC probability preservation,
target limiting cases, hard-CE equivalence, soft-loss gradients, source hashes,
frozen defaults, and scope confinement. All Python files also pass AST parsing
and `compileall`.

This is code validation, not H4 evidence. Full entry-point and GPU validation
remain pending because this Windows environment lacks the Baseline's
`tensorboardX` dependency and the copied folder contains no PROMISE12 data,
lists, or checkpoint.

The validated implementation has also been deployed beside the original files
under `E:/Desktop/Baseline`: `code/train_coda.py` and `code/test_coda.py` are
parallel entry points, while `code/utils/coda.py` and
`code/utils/promise12_preflight.py` contain the new method and data contract.
Their defaults are `E:/Desktop/PROMISE12` and `CoDA_MT_PROMISE12`. The original
training, testing, and dataset file hashes still match the pre-deployment audit.

The desktop data audit resolved the previously missing split context: 35 train
cases produce 940 ordered slices, and the first seven cases produce exactly the
first 191 slices expected by `patients_to_slices`. There are five validation
and ten test cases. Runtime preflight now enforces these counts and the exact
list hashes without editing them.

The initial desktop copy contained Git LFS pointers, but the user has now
replaced them with real HDF5 objects. All 955 assets referenced by the locked
train/validation/test lists have valid HDF5 signatures. Sampled labeled-boundary
slices, validation volumes, and test volumes contain finite float32 images and
binary int8 labels.

The replacement lists use Windows CRLF while the original locked lists used LF.
Their ordered entries are otherwise exactly identical. Split locking now hashes
a canonical newline-joined sequence read with UTF-8-BOM tolerance. This fixes
the false positive without relaxing the actual constraint: any entry addition,
deletion, rename, or reorder still fails preflight. All 14 desktop tests now
pass without a skip, so code and data are ready for the first B0/B1 run.

The first actual desktop launch then exposed a Windows-only runtime defect inherited
from the Baseline structure: each training phase defined `worker_init_fn` locally,
which cannot be pickled by the `spawn` multiprocessing context. The parallel CoDA
entry now uses a single module-level callback with `functools.partial`; this retains
four workers and the original `seed + worker_id` behavior. Fourteen tests pass, the
callback serializes successfully, and a real four-worker spawned loader read a
two-sample 256x256 PROMISE12 batch. Original Baseline source hashes are still intact.

## Go/No-Go Decision

Keep CoDA as the headline only if B1 exposes a reproducible overcertainty/negative-transfer regime and B6 beats B3, B4, and B5 for at least two degradation families while improving both segmentation and calibration.

If B3 or B5 matches B6, simplify to global corruption-coupled pseudo-label calibration. If B1 never becomes harmful, reject H4 and return to pseudo-label quality instead of adding modules.

## Optimization Trajectory

No real-data training run has been executed because the copied folder contains
code but no data lists, checkpoints, or baseline outputs. The isolated CoDA
implementation passed its CPU and source-contract checks. The next confirmatory
sequence is still B0 reproduction followed by the pre-registered B1 severity
diagnostic; B6 must not be interpreted until B0 and B1 pass sanity checks.

## 2026-08-11 Current-Workspace Audit and Pivot

This section supersedes the earlier next-step recommendation where the current
workspace and the historical desktop record disagree. The present repository contains
the baseline and CoDA code, but no locally auditable training log, event file,
checkpoint, performance file, list file, or PROMISE12 H5 data. The user's report that
CoDA gives no or only a very small improvement is credible exploratory evidence, but it
has no seed, command, checkpoint, or curve with which to estimate variance or attribute
the effect.

### Baseline identity

The fixed baseline is best named **BCP-derived EMA hard-pseudo-label self-training**.
It is not BCP: BCP's defining labeled/unlabeled bidirectional Copy-Paste and mixed
GT/pseudo-target supervision are absent. It is not canonical Mean Teacher either:
teacher and student receive the same loader-augmented unlabeled tensor, teacher
probabilities are reduced by argmax plus 2-D largest-connected-component filtering, and
the student is trained with hard CE+Dice rather than a prediction-consistency loss.

The user's baseline remains locked for method comparison. Network, two-stage schedule,
teacher behavior, LCC target, losses, EMA, consistency ramp, sampler, data order, and
inference stay unchanged. Evaluation errors (spacing, empty masks, and checkpoint
identity) may be corrected only symmetrically and reported beside the legacy metrics.

### Why the current CoDA result is not diagnostic

The current `train_coda.py` changes four factors at once: student weak/same view to a
strong view; hard LCC one-hot to teacher probability inside the LCC; hard CE/Dice to
soft variants; and an evidence-dependent interpolation toward the uniform target.
Therefore `gamma=0` is still not the baseline.

In binary small-organ segmentation, the uniform endpoint does not “ignore destroyed
evidence.” It assigns foreground probability `gamma/2` to every originally hard
background pixel, allowing background mass to dominate the intended local effect.
Moreover, strong unlabeled images share a BatchNorm forward with labeled images, and
the Sobel/SNR gamma maps have not been shown to predict task error. The exploratory
small gain is consequently compatible with both a weak hypothesis and an entangled,
counteracting implementation.

### Literature-driven rejection map

- generic Copy-Paste, patch displacement, and foreground-aware regional exchange are
  crowded by BCP, ABD, RCP4CL, PSC, MiDSS, and related work;
- global Fourier/style directions are crowded by MiDSS, FRCNet, beta-FFT, and UGAC;
- uncertainty weighting/selection is crowded by DyCON, ALHVR, AC-MT, and many
  conservative/radical-teacher methods;
- augmentation-orbit intersection/union is directly preceded by TTA uncertainty,
  CPCL, DUEB, and conformal segmentation sets;
- generic boundary attention, loss, sampling, or contrast is occupied by BoundaryMix,
  LoCo, BoCLIS, and many boundary-aware networks;
- adjacent-slice interpolation and partial-volume/resolution synthesis already have
  direct precedents, reducing the novelty ceiling of a simple 3-D resampling proposal.

### Primary direction: BMER

BCP now serves only as a bar for a simple, visible, single intervention. The selected
direction is independent of its method structure:

> Estimate the empirical distribution of two-sided image evidence on the unlabeled
> object boundary manifold, then resynthesize sampled evidence on labeled GT anatomy
> while leaving geometry, hard targets, and the complete self-training baseline fixed.

BMER works in `(tangential position, signed normal distance, longitudinal position)`
rather than Cartesian patch coordinates. It does not Copy-Paste, pair labeled and
unlabeled images bidirectionally, transform the pseudo-target, add a learned generator,
or alter inference. The larger unlabeled set supplies nuisance-support statistics; only
the labeled input is resynthesized and remains supervised by exact GT.

The nearest threats are semi-supervised task-driven learned intensity/deformation,
KeepMask/KeepMix, ARHNet's full-foreground affine perturbation and harmonizer,
BoundaryMix's pixel replacement, global histogram/Fourier transfer, and boundary
contrastive objectives. BMER is defensible only as *conditional boundary-manifold
evidence resynthesis*. A mask-affine, scalar contrast, histogram, or patch-replacement
implementation is not novel enough.

### Mandatory go/no-go evidence

Before training, freeze the baseline and use held-out GT masks for an oracle
intervention. Complete strong boundary profiles rendered on weak-evidence recipients
must improve boundary prediction; weak profiles rendered on strong-evidence recipients
must worsen it; the response must be concentrated in the predeclared band and exceed
area/severity-matched scalar contrast, blur, and global histogram controls. If the full
profile does not show this ordered localized effect, reject BMER immediately.

The second gate measures whether teacher-LCC profiles preserve GT-profile ordering on
oracle-labeled cases. Failure means unlabeled pseudo boundaries cannot define the
augmentation bank. Neither gate may be rescued by adding selectors, uncertainty heads,
losses, or teacher changes.

### Venue calibration

No direction can be guaranteed to publish before results. BMER has a plausible CVPR
novelty boundary, but a PROMISE12-only sub-point improvement is insufficient. A CVPR
claim requires the same operator and normalized coordinate policy on PROMISE12, ACDC,
and a genuinely 3-D benchmark, multi-seed patient-level statistics, physical surface
metrics, mechanism plots, and wins over matched simple local augmentation. If it only
works on prostate MRI, the correct target is a medical-imaging venue rather than adding
modules to inflate the method.

## 2026-08-11 External CoDA Run Evidence

The user supplied a complete self-training text log and a ten-case test performance
file for the current CoDA combination. The raw files are now archived under
`experiments/h4_evidence_coupled_target/results/coda_external_run_2026-08-11` with
SHA-256 provenance.

The run completed 30,000 self-training iterations without a reported non-finite
validation value. Best validation Dice was 0.804897 at iteration 27,200; the final
value was 0.797537. Over the final 5,000 iterations, validation Dice was
0.794807 +/- 0.004774 (sample SD). Selecting the maximum among 150 validation checks
therefore contributes a visible 0.007360 advantage over the final iterate.

The selected student checkpoint achieved test Dice 0.819876 and Jaccard 0.697646.
The user reports a baseline final Dice range of 0.78--0.80, so the apparent unpaired
difference is +0.019876 to +0.039876 absolute Dice. This is large enough that CoDA
should not be described as a demonstrated failure. It is nevertheless not a causal
validation of CoDA: the baseline lacks a paired per-case artifact, the run identity
lacks the exact command/seed and checkpoint hash, and the text log omits evidence
loss, target entropy, family/severity, and consistency-loss traces.

The code-level confounding remains decisive. Current CoDA changes strong views,
teacher target representation, loss implementation, and uniform coupling together.
Any gain could be caused by one component or by BatchNorm interaction rather than
augmentation-conditioned target entropy. H4 is therefore **exploratory positive,
mechanism unresolved**. It remains a useful empirical comparator but does not replace
BMER as the primary novelty direction or waive BMER's oracle and teacher-profile kill
tests.

Surface metrics reveal a long tail: Case34, Case09, and Case05 account for about 74%
of summed HD95. Their numeric HD95/ASD values are not physical distances because the
test implementation does not pass voxel spacing to MedPy. Formal comparisons must
rerun both methods symmetrically with physical spacing and paired patient-level
statistics.

## 2026-08-11 BMER v1 Negative Run and Conditional Pivot

The first BMER real-data artifact is negative for the current implementation and is
also incomplete. Training stopped at 20,800 of 30,000 iterations. Best validation Dice
was 0.788501 at step 15,400, followed by a decline to 0.767375. The 15.2k--20k window
mean was lower than the 10.2k--15k window, so this is not merely a visually noisy but
steadily improving curve.

The selected checkpoint achieved test Dice 0.795949, which lies inside the user's
unpaired baseline range of 0.78--0.80. Against the archived CoDA checkpoint on the same
ten cases, BMER loses 0.023927 mean Dice and worsens nine cases. Its only Dice win,
Case34, coincides with a legacy HD95 increase from 32.02 to 107.02, indicating that
overlap improvement does not represent clean boundary recovery. Current BMER therefore
does not meet the empirical bar for promotion.

This run does not cleanly falsify the abstract hypothesis because the mandatory oracle
and teacher-profile gates were skipped and the implementation has a concrete coordinate
confound. Donor fields are extracted from unaugmented slices, but labeled recipients
undergo random rotations/flips before their absolute-angle sectors are matched to donor
sectors. The claimed tangential coordinate is therefore not equivariant to the existing
geometric augmentation. This issue justifies one diagnostic correction cycle, not a
hyperparameter sweep.

The frozen bank retains 371 of 749 unlabeled slices. The approximately 50% skip rate is
ambiguous: it may include legitimate background-only slices as well as teacher misses.
Without oracle stratification it cannot support the claim that unlabeled data expands
apex/base boundary evidence. BMER also reduces training throughput by roughly 30%
because each batch performs CPU distance transforms and rendering.

The decision is now conditional pivot. Stop the current renderer and do not tune its
radius, probability, strength, confidence, target, or teacher. Run the locked oracle,
teacher-fidelity, and rotation/flip-equivariance tests. Only if all pass may sector
alignment be corrected and one shared-pretrain 2--3k B0/B4/B6/B7 screen be run. Failure
at either stage rejects H5 and triggers a clean re-bootstrap around acquisition-model
augmentation rather than adding components to BMER.

Novelty risk has also increased. FDIF (arXiv:2603.23199) already uses signed distance
to drive procedural surface appearance in synthetic 3-D medical volumes. BMER remains
different in using real unlabeled empirical evidence on fixed real anatomy, but a
distance-conditioned appearance claim by itself is no longer defensible. This lowers
the expected value of repeated BMER rescue attempts.

## 2026-08-11 Superseding Decision: Close BMER and Pivot to OBA

The user explicitly declined the previously authorized BMER diagnostic/correction
cycle. This supersedes the conditional rescue language above: H5 is archived, its code
and negative artifacts remain for provenance, and no BMER correction, tuning, or hybrid
method is authorized.

The combined result pattern supports a branch-asymmetric interpretation. CoDA's
complete exploratory run changes the pseudo-labeled unlabeled path and reaches test
Dice 0.819876, although its four-factor implementation prevents causal attribution.
BMER perturbs only the exact-GT labeled path, peaks early, tests at 0.795949, and loses
on 9/10 paired cases versus CoDA. This is not proof that labeled augmentation is always
harmful; it is sufficient evidence to stop complex labeled-image synthesis and retain
the labeled branch as a reliable optimization anchor in the next cycle.

A top-conference-first collision audit rejects the most immediate pivots. Adaptive
severity/selection is occupied by AugSeg, SAA, and iMAS; teacher-recognizable
adversarial augmentation by TeachAugment; two random strong views by UniMatch;
diverse/conflicting views by Diverse Co-training and CCVC; and supervised/unsupervised
gradient correction by ICCV 2025 POS. Frequency mixing and boundary ambiguity are also
dense in recent CVPR medical segmentation work.

The selected hypothesis is **Orbit-Balanced Augmentation (OBA)**. Instead of sampling
one directional strong perturbation, OBA evaluates an unlabeled image under a paired
transformation coordinate `+a/-a` and averages the baseline hard-pseudo-label loss. A
local Taylor expansion cancels odd-order terms, retaining second-order robustness
pressure without softening the target, synthesizing a donor image, changing the
teacher, or adding inference components.

The novelty is not antithetic sampling itself and not merely using two views. The
publishable question is whether balanced quadrature of a transformation orbit controls
the bias and variance of hard-pseudo-label dense prediction. Before implementation,
Gate 0 must show that `+a/-a` views actually have negatively correlated signed logit
displacement and smaller mean drift than two IID same-cost views. A short training
screen must then beat both two-IID-view and shuffled-pair controls. Failure at either
gate rejects OBA without module stacking.

The full formulation, 15-candidate rejection record, controls, and contribution draft
are in `ideas/orbit_balanced_augmentation.md`; the accepted top-conference collision map
is in `literature/31_top_conference_pivot_map.md`.

## 2026-08-11 OBA Full-Run Reflection

The fixed-seed OBA run completed all 30,000 iterations and reaches test Dice 0.818872,
essentially identical to CoDA's 0.819876. It is apparently positive relative to the
user-reported unpaired baseline range of 0.78--0.80, so the experiment is not a simple
failure. It does not, however, support an OBA-specific mean improvement: OBA wins only
three of ten paired cases, the median paired difference is -0.016190, and an exact
sign-flip comparison supplies no evidence against equality.

The training trajectory is the primary warning. OBA peaks at validation Dice 0.793850
at 13.8k and declines to 0.699411 by 30k. Its four consecutive 5k-window means fall
from 0.785966 to 0.772288, 0.743147, and 0.717645. In contrast, CoDA improves late and
has only a 0.007360 best-to-final gap. The timing aligns with the inherited consistency
weight increasing from about 0.116 at the selected OBA checkpoint to 0.5 at the end.
This supports a provisional explanation: antithetic views provide useful early
robustness pressure, but symmetric supervision does not cancel shared hard pseudo-label
error or even-order curvature, so increasing weight ultimately amplifies confirmation
bias. The event diagnostics are required to test this explanation.

The later-supplied pretraining log adds a second selection warning. Supervised
pretraining peaks at validation Dice 0.679773 at 7.8k but ends at 0.570292 at 10k, so
self-training starts from a validation-selected spike rather than a stable terminal
model. The deterministic code suggests CoDA, BMER, and OBA should share this
initialization, but the comparison cannot prove it without checkpoint hashes.

The user subsequently reports approximately 0.71 pretraining validation Dice for CoDA
and 0.73 for baseline. This is not an OBA effect: `train_oba.py` imports and directly
calls `train_coda.pre_train`, and all OBA operators are confined to self-training. The
30--50 millipoint initialization gap is vastly larger than the one-millipoint OBA--CoDA
test difference. Because the inherited method forms unfiltered hard LCC teacher masks
and restores SGD momentum from the pretraining checkpoint, initialization changes both
pseudo-label error and the full optimization trajectory. It may partly explain OBA's
late collapse, but cannot be used to post-hoc adjust its final Dice.

The IID gate is therefore temporarily superseded by a fairness repair. Future
self-training comparisons must share one exact `net+opt` checkpoint hash and an
explicitly reset stage RNG. Prefer the established baseline checkpoint only if its
0.73 result, code/split identity, and file hash can be verified; otherwise generate a
single fresh checkpoint and accept it without rerunning for a better validation score.

OBA does show an exploratory tail-risk pattern. It improves the two cases on which CoDA
is weakest, reduces case Dice SD from 0.051567 to 0.036590, raises minimum Dice from
0.746351 to 0.766989, and roughly halves the worst legacy HD95. These benefits are
dominated by Case05 and Case34 and do not establish robust risk reduction on ten cases.
The paper story must not be changed post hoc from mean performance to worst-case
robustness without a locked second-dataset evaluation.

The literature boundary is now sharper. UniMatch already establishes dual strong
student views guided by a shared weak target; MixMatch establishes averaging predictions
over augmentations; strong-augmentation BN shift and unreliable hard pseudo-labels are
known failure modes. Consequently, neither adding two views, moving the loss to a plain
prediction average, nor fixing BN can serve as the contribution. The only remaining OBA
claim is that antithetic coordinate pairing is superior to IID views at matched marginal
severity and compute.

The project therefore enters one bounded deepening cycle. First analyze the existing
TensorBoard OBA traces. Then run one same-seed, same-pretrain, effective-batch-36
two-IID-view control. If IID matches OBA within 0.005 Dice, is materially more stable,
or reproduces the hard-case benefit, reject H6 and pivot. No severity, probability,
consistency-ramp, duration, teacher, or target tuning is authorized before this gate.

## 2026-08-11 UniMatch Evidence and Final OBA/CoDA Decision

The user reports that the Baseline-folder UniMatch implementation averages approximately
0.83 Dice. No corresponding log, checkpoint, or per-case artifact has yet been imported, so
the number remains user-reported rather than independently audited. Nevertheless it is a
decision-relevant strong comparator: both CoDA (0.819876) and OBA (0.818872) are below it.

The user also correctly distinguishes OBA's trajectory from an ordinary convergence plateau.
OBA peaks at 13.8k and then its validation window means decrease monotonically through four
successive windows while the inherited unsupervised weight continues to rise. The selected
test checkpoint is usable, but the training dynamics show that the method's pressure becomes
increasingly harmful. Adjusting the ramp, stopping early, changing BN exposure, averaging
predictions, or adding confidence selection could improve a run, but each would be a post-hoc
repair and several are already covered by UniMatch, AugSeg/SAA, or POS. Such repair cannot
rescue the original antithetic-cancellation contribution.

The previously proposed IID control was never expected or required to be worse. A control is
designed to falsify the mechanism: if IID matches or beats OBA, antithetic pairing has no
independent value; only a clear OBA win would preserve H6. With UniMatch already around 0.83,
the opportunity cost changes. The IID experiment remains logically necessary for an OBA
paper claim, but it is not necessary to decide to abandon OBA. No additional full OBA control
or rescue run is now recommended.

CoDA is also not promoted. Its relative stability and 0.819876 test Dice make it a useful
empirical comparator, but its four simultaneous changes and the uniform-target foreground
bias prevent a clean contribution. Correcting those issues would amount to designing a new
method rather than repairing an established core.

The pretraining discrepancy remains real but is reclassified as experimental infrastructure.
Every future method must consume one explicit baseline pretraining checkpoint containing both
network and optimizer state, record its SHA-256, and reset the self-training RNG. The inspected
Desktop example uses a required `--match_pre_checkpoint`, resolves exactly one path, and fails
if it does not exist. No other code or method design from that folder is used. Shared
pretraining can make future attribution fair; it cannot retroactively turn OBA into a result
above UniMatch.

The outer loop therefore pivots to H7, **Paired Slice-Profile Re-Acquisition (working name:
SliceEq)**.
The candidate treats a reconstructed MRI slice as a finite-thickness slab observation and
passes both the image volume and its GT/teacher-mask occupancy through the same stochastic
slice-profile operator. This differs from generic image-only blur, ordinary adjacent-slice
interpolation, 2.5D inference, and confidence filtering. It preserves the 2D network and hard
CE+Dice training interface while exposing acquisition variation that the current slice-wise
pipeline discards. H7 is not yet a result: a no-training operator-validity gate must first show
boundary-localized, anatomically plausible target changes beyond a matched 2D blur control.
Recent E(3)-Pose and C3 work also means slice-profile or clinical-physics augmentation alone is
not novel; the claim must stand on the paired image/occupancy operator in semi-supervised
training.
The complete formulation and collision map are in `ideas/slice_profile_reacquisition.md` and
`literature/35_post_oba_pivot_map.md`.

## 2026-08-11 SliceEq Final Fixed-Seed Implementation

At the user's request, the project implemented the complete SliceEq training path before the
H7.1 sub-experiments. This is an ordering change, not evidence that the validity gate passed.
The method remains falsifiable through diagnostics embedded in the full run.

The implementation preserves the supervised central-slice anchor and the inherited 2D U-Net,
SGD state, EMA update, consistency ramp, hard CE+Dice objective, sampler proportions,
validation schedule, checkpoint selection, and inference graph. Iterations 0--999 execute the
baseline identity path. Thereafter, each unlabeled sample loads the real same-case slices at
offsets -1/0/+1. The teacher applies the existing hard argmax plus 2D-LCC operation to every
slice, and one sampled normalized Gaussian profile is applied identically to the image stack
and one-hot hard-mask occupancy. The student still receives one image channel and one hard
target.

The shared initialization confound is addressed prospectively: `train_sliceeq.py` uses one
fixed checkpoint containing both `net` and `opt`, loads it strictly into student/teacher and
optimizer, records SHA-256, and resets the self-training RNG. No checkpoint auto-search is
performed. The established server path is now the parser default; because no actual `.pth`
exists in this workspace, its contents cannot be runtime-validated here.

Source-level validation passed: the new files compile, six SliceEq contract tests pass, and all
21 available repository contract tests pass with two expected data-root skips. Tensor tests are
included but the local lightweight Python lacks a loadable PyTorch installation, so they remain
a required first command in the CUDA environment. No SliceEq Dice, HD95, stability, or
mechanistic benefit is claimed yet.

The training entry subsequently adopted the existing Baseline experiment's fixed Pre10000
path as its default:
`/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth`.
This removes the need to pass the path for the normal server layout without weakening the
identity contract: the file must still exist, contain `net+opt`, load strictly, and have its
actual SHA-256 recorded. A CLI override is only a relocation mechanism and performs no search.

The same prospective initialization contract now applies to `train_coda.py` and
`train_oba.py`. Their main entries no longer invoke supervised pretraining: both default to
the identical Pre10000 file, require `net+opt`, load strictly, log SHA-256, reset all stage
RNGs, and run only Self30000. This makes future CoDA/OBA/SliceEq comparisons share an exact
starting state. It does not repair or retroactively reclassify the archived CoDA/OBA results,
which were produced before this change with independently selected pretraining checkpoints.
# 2026-08-11: Public SOTA target for SliceEq

Local results were excluded as requested. Under the conventional PROMISE12
semi-supervised setting of 20% labeled training volumes, the strongest accepted
top-conference result found is PMPC at **85.80 +/- 1.07 Dice** (AAAI 2026;
7:1:2 split, U-Net, three runs). Earlier close peers are PSC at 83.64 (MICCAI
2024) and beta-FFT at 83.75 +/- 0.65 (CVPR 2025). A broader 2026 journal method
using a dual-teacher CE-Net/SCFR design reports 87.50, but it is not an
architecture-matched comparison. Consequently SliceEq should use 86.0 as a
one-seed continuation threshold and 87.5 as the unqualified SOTA-facing target.
These literature values are not guaranteed to use the identical patient IDs,
so a final paper must distinguish quoted and reproduced comparisons.

# 2026-08-11: SliceEq first fixed-seed result and optimization decision

The supplied SliceEq run is a positive direction-selection result. It uses seed 1337 and
reports the fixed pretrained checkpoint hash
`49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`. Test Dice is
0.832603, compared with 0.819876 for archived CoDA and 0.818872 for archived OBA. Validation
progresses from a mean of 0.763548 in 0--10k to 0.785795 in 10--20k and 0.795354 in 20--30k,
with a best of 0.811287 at 24.8k. This is materially healthier than OBA's monotonic late
deterioration and justifies retaining SliceEq.

The result is not yet a method-level success. It essentially ties the user-reported UniMatch
result (0.832233), falls below the 0.86 continuation gate, and lacks an identical-checkpoint
baseline artifact. Archived CoDA/OBA runs predate shared-checkpoint enforcement and are not
fully controlled comparators. The five-case validation set remains noisy: the best-to-final
gap is 0.023014.

Case-level evidence is heterogeneous. SliceEq wins only 4/10 cases versus CoDA and 5/10 versus
OBA. Against CoDA, the paired median is -0.018262 and removing the three largest positive
cases makes the remaining mean -0.021907. Case05, Case09, and Case34 dominate the apparent
gain. This could be the expected signature of acquisition-specific benefit, but it requires
z-spacing/thickness/vendor and apex/mid/base stratification; without that interaction it is
only an unexplained outlier concentration.

The implementation also reveals why straightforward severity tuning is the wrong next step.
For the three-tap Gaussian profile, the center coefficient lies between about 0.485 and 0.855
and is below one half over only about 3.5% of the sampled sigma/phase area. Since target
occupancy is immediately reduced by hard argmax, ordinary center masks almost always remain
unchanged. The text log omits the TensorBoard `target_changed_fraction` trace, so the paired
target mechanism is unverified and v1 may primarily be an adjacent-slice image averaging
augmentation.

The selected refinement is fractional-occupancy SliceEq. The same sampled profile should
produce `o_phi = sum h_phi one_hot(Y)` and supervise that distribution directly, rather than
discard it with argmax. This is materially different from CoDA's uniform smoothing: regions
where all neighboring masks agree remain exactly one-hot, and softness exists only at
acquisition-induced mixed-tissue locations. The complete formulation should keep the original
hard central labeled anchor, add one paired labeled view using exact neighboring GT occupancy,
and use detached LCC pseudo-mask occupancy on unlabeled volumes. Network and inference stay
unchanged.

This design remains adjacent to prior physics and partial-volume work. ESPRESO establishes
that MR slice profile is a through-plane PSF and may differ from nominal thickness. PV-SynthSeg
and SynthSeg simulate partial volume, resolution, thickness, and spacing, so a paper cannot
claim novelty for slice-thickness augmentation alone. The defensible claim is acquisition-
equivariant semi-supervised supervision: applying one forward operator to observed image
signal and GT/pseudo tissue occupancy together, with no inference-time module. AmbiSSL concerns
multi-annotator ambiguity rather than physical occupancy and does not occupy this exact claim.

## 2026-08-11 H7.2 independent implementation

The fractional-occupancy successor is implemented under the independent identity
`SliceEqOcc_PROMISE12`; no SliceEq v1 source was changed. The implementation deliberately
does not widen the profile range. After the exact first-1k baseline path, one student forward
contains 12 original labeled centers, 12 labeled views re-acquired from exact neighboring GT,
and 12 unlabeled views re-acquired from detached EMA/LCC masks. The original and re-acquired
labeled losses are averaged, while the unlabeled soft occupancy loss retains the inherited
consistency coefficient. This keeps one optimizer and EMA update and no inference-time module.

The new loss uses the actual profile-weighted class occupancy. It does not interpolate toward
a uniform prior: unanimous neighboring masks remain exactly one-hot. Separate seeded profile
generators prevent the added labeled view from consuming the unlabeled profile stream. Runtime
diagnostics now distinguish fractional-pixel activity from hard argmax change and are mirrored
to the text log every 200 steps, addressing the missing-event limitation of the v1 evidence.

The test entry defaults both the canonical `--save_result` destination and its user-facing
`--save_results` alias to False, disables cross-experiment checkpoint discovery, and writes to
the separate SliceEqOcc snapshot. Source validation is positive; numerical tensor validation
is deferred only because every available Windows PyTorch wheel fails to initialize `c10.dll`
in this workspace. The CUDA server must run the supplied tensor tests before the full run.

## 2026-08-11 SliceEqOcc result and checkpoint-selection audit

The complete seed-1337 SliceEqOcc run uses the intended fixed pretraining checkpoint hash
`49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9` and the locked
profile range. The validation-selected `unet_best_model.pth` is the iteration-30000 model:
validation Dice 0.817373 and test Dice 0.827368. It exceeds the user-reported legacy baseline
range and archived CoDA/OBA, but is 0.005235 below SliceEq v1 and does not exceed the reported
UniMatch comparator.

The run nevertheless confirms that fractional occupancy is not a dead implementation. Across
post-warmup logs, fractional pixels average 0.8214% for labeled re-acquisition and 0.8797% for
unlabeled re-acquisition, with closely matched entropy and deviation scales. Unlabeled hard
argmax changes average only 0.0026% of pixels. Fractional supervision therefore activates the
specific acquisition-boundary information that v1 discards, but its support is sparse and its
current aggregate benefit is not confirmed.

The user's checkpoint observation exposes an evaluation problem rather than evidence that the
final validation value is invalid. Iteration 23k has validation Dice 0.815152 and reportedly
tests near 0.844, whereas the final 0.817373 checkpoint tests at 0.827368. Yet the final maximum
is supported by 28.8k and 29.0k validation values of 0.816474 and 0.817338; it is not a lone
spike. The only extreme isolated validation event is the 25.4k collapse to 0.673511. Because
the validation set has five cases, a 0.0022 difference cannot reliably rank checkpoints for a
ten-case test set.

Accordingly, approximately 0.844 is retained as a user-reported exploratory oracle-checkpoint
upper bound, not the paper's primary number. Selecting it after testing multiple checkpoints
uses the test set for model selection. The research direction remains promising because the
trajectory contains a substantially stronger-generalizing state and the proposed mechanism is
measurably active, but formal success requires preserving the 23k checkpoint/artifact and then
confirming a pre-registered test-independent checkpoint rule or an untouched second evaluation.

### Evidence upgrade: exact iteration-23000 performance

A subsequently supplied performance file explicitly identifies
`iter_23000_dice_0.8152.pth` and reports test Dice 0.844566, Jaccard 0.732999, HD95
3.651809, and ASD 1.439373. This removes the earlier uncertainty about the metric and model
path, although the checkpoint binary hash is still absent. The user also reports that several
adjacent earlier checkpoints test near 0.84; those additional artifacts have not been supplied.

The 23k result is not driven by one favorable test case. Against the validation-selected final
checkpoint it improves mean Dice by 0.017198, wins 9/10 cases, and has paired median +0.012773.
Against SliceEq v1 it improves mean Dice by 0.011963, has positive paired median +0.002077,
and wins 5/10; its few losses are small while several gains are substantial. H7.2 is therefore
upgraded from mechanism-only success to a strong exploratory positive result.

This does not make the final measurement erroneous. It changes the diagnosis: the final model
is a real but poorly generalizing state selected by a noisy five-case validation ranking after
150 repeated evaluations. The earlier approximately-0.84 neighborhood is more representative
of the run's best generalization regime, while 0.844566 remains post-hoc because the test set
was queried across checkpoints. Both values must be retained transparently until a fresh,
untouched evaluation confirms a test-independent selection rule.

## 2026-08-11 SliceEq optimization-space audit

Ignoring checkpoint selection, the current method still leaves a specific optimization gap.
Fractional occupancy is active on only 0.8214% of exact-GT labeled pixels and 0.8797% of
teacher-derived unlabeled pixels. Mean total-variation deviation from the central hard target
is 0.001628 and 0.001821. Nevertheless, both soft cross-entropy and soft Dice are computed over
the entire image. Most of the re-acquired objective therefore repeats consensus background and
interior supervision, while the unique acquisition-induced target signal receives little
normalized optimization mass.

The selected H7.3 candidate is acquisition-residual dual-measure risk. Define the detached
per-pixel residual as half the L1 distance between profile occupancy and the central one-hot
target. Keep the existing full-image occupancy loss, but add a separately normalized loss under
the residual-induced measure. This preserves the target and operator, introduces no inference
module, and cannot inject foreground into unchanged background. Its distinction from generic
boundary loss is that support and weight are generated by the sampled acquisition operator and
retain fractional tissue proportions, not by a distance transform or uncertainty score.

This is not yet authorized for a full run. First, the retained 23k model must show that residual
pixels contribute less than 20% of the current re-acquired gradient despite their causal role.
Second, teacher-derived residual support on labeled cases must correlate with exact-GT residual
mass by at least 0.3 and must not mostly lie outside the GT support. Third, a same-support binary
boundary weight must not reproduce the residual measure. These gates protect the paper from
turning SliceEq into an ordinary boundary-weighted SSL method.

Other numerical improvements remain possible but are not contribution candidates. The inherited
unlabeled coefficient rises from about 0.379 at 23k to 0.5 at 30k and may explain late loss of
test generalization; capping it now would be post-hoc schedule tuning. Teacher eval mode, BN
isolation, LR decay, checkpoint averaging, and stronger backbones must be shared infrastructure
changes or controls. Physical profile calibration from real spacing/thickness metadata is a
valuable later extension, but arbitrary radius/sigma widening is not scientifically justified.

## 2026-08-12 H7.3 gate implementation

The three gates are now fixed before numerical execution and implemented independently of the
training entries. Gate 1 measures the exact-GT residual-support share of the current soft
CE+Dice logit gradient. Gate 2 uses identical sampled profile weights to compare exact-GT and
frozen-model LCC residuals on labeled stacks, excluding common zero background from pixel
correlation and common zero-residual samples from mass correlation. Gate 3 compares normalized
soft-CE gradients under fractional residual weights and a same-support binary measure.

The retained iteration-23,000 checkpoint contains only a student state dictionary. Consequently,
the implementation uses the frozen student in evaluation mode as a proxy for Gate 2 and records
that limitation in both the protocol and output. This is weaker than direct EMA-teacher evidence
but stronger than silently reconstructing an unavailable teacher. A three-gate pass is therefore
called provisional.

The analysis is read-only: it has no optimizer, backward call, training loop, or checkpoint write.
It defaults to all 191 labeled slices plus the first 192 unlabeled slices, and the unlabeled path
uses an image-only dataset subclass so hidden H5 labels are not read. The single JSON report stores
arguments, code/data/checkpoint hashes, sample counts, metrics, thresholds, and the joint decision.
No numerical gate outcome is claimed in this workspace because the Linux data, checkpoint, and
CUDA PyTorch runtime are unavailable locally.

## 2026-08-12 H7.3 gate result and posterior-commutation pivot

The supplied server JSON is provenance-complete: all locally corresponding source and protocol
hashes match, the run covers all 191 labeled slices and 192 unlabeled slices, and the retained
23k checkpoint now has SHA-256 `3f3515e6411d8a50877ba6b660c82fcb9752eab524172315a3e8f8b7845c8052`.
The preregistered joint decision is `stop_h7_3`.

Gate 1 decisively rejects the dilution hypothesis. Exact acquisition residuals occupy only
0.842669% of labeled pixels but already contribute 65.6513% of the complete soft CE+Dice logit
gradient and 54.4893% of the CE-only gradient. These are 77.91-fold and 64.66-fold enrichments
relative to support area. The unlabeled proxy branch reproduces the pattern: 0.813309% support
contributes 50.7788% of the complete gradient, a 62.43-fold enrichment. Full-image averaging is
therefore not starving the acquisition boundary. A separately normalized residual risk would
amplify an already dominant signal and may worsen the observed late instability. H7.3 is closed;
its thresholds must not be relaxed and no training implementation is authorized.

The successful gates refine rather than rescue the hypothesis. Gate 2 shows support precision
0.887736, recall 0.855451, IoU 0.771941, per-sample mass correlation 0.992766, and only 0.112088
proxy mass outside exact support. The proxy reliably locates acquisition-sensitive boundary
regions and their total amount. Yet pixelwise residual correlation is only 0.348508, barely over
the preregistered floor. The remaining error is local fractional magnitude after hard/LCC pseudo
segmentation, not missing spatial emphasis. Gate 3 confirms fractional and binary weights are
distinct (gradient cosine 0.933901), but this specificity does not justify unnecessary weighting.

The next bounded hypothesis is H7.4 posterior commutation:
`f_student(A_h X) ~= A_h f_teacher(X)`. It retains SliceEq's physical operator and exact-GT
labeled branch but tests whether applying the profile directly to raw or topology-gated teacher
posteriors improves local occupancy fidelity over `A_h(one_hot(LCC(argmax(q))))`. This is not a
generic soft-label, uncertainty-weighting, or set-prediction claim; those directions are occupied
by recent CVPR/ICCV work. H7.4 must first pass a no-training labeled-stack fidelity test with a
15% boundary-Brier improvement, residual correlation at least 0.50, outside mass at most 0.15,
and no more than 5% full-image Brier degradation. Failure triggers a broader pivot rather than
another SliceEq loss modification.

## 2026-08-12 H7.4 gate implementation

The H7.4 posterior-commutation fidelity gate is now implemented independently of all training
entries. It evaluates current hard-LCC occupancy, raw teacher-posterior commutation, and
topology-gated posterior commutation on all 191 labeled stacks using the same seed-1337 profile
stream as H7.3. Each variant reports exact-support and full-image occupancy Brier, acquisition-
residual pixel/mass correlations, off-support residual mass, support overlap, foreground-volume
bias, and endpoint-clamped versus non-clamped strata.

The analysis does not trust configuration equality alone. Before an H7.4 decision it must
reproduce the prior H7.3 hard-LCC sample count, union-support pixels, pixel and sample-mass
correlations, support IoU, and outside-mass fraction within absolute `1e-6`, and it must match
the recorded 23k checkpoint SHA-256. Failure raises an error before writing a result. This makes
the hard target a paired internal reference rather than a separately rerun control.

No training source was changed. New sources pass Python compilation and four H7.4 contracts;
all four H7.3 and six SliceEqOcc preservation contracts remain positive. Five numerical utility
tests are provided for the CUDA environment. A full H7.4 training version remains unauthorized
until the generated JSON reports `authorize_h7_4_training`.

## 2026-08-12 H7.4 result and operator-integration pivot

The supplied H7.4 artifact passes its reproduction guard exactly: all six locked H7.3
hard-LCC quantities are identical and the iteration-23k checkpoint SHA-256 matches. The
preregistered joint decision is `stop_posterior_commutation`; no candidate is authorized.

Raw posterior commutation recovers local magnitude but is not acquisition-local. Its
exact-support Brier ratio is 0.659075 and residual Pearson is 0.828573, yet nonzero residual
support covers 9,542,518/12,517,376 pixels (76.23%) and 23.1295% of residual mass lies outside
exact acquisition support, above the locked 15% cap. Topology gating reduces support to 3.04%
and keeps 91.48% exact-support recall, but its Brier ratio is 0.892700, outside mass is 15.2661%,
and foreground occupancy is biased low by 4.12%. It therefore fails two locked conditions.
The failure remains on the 177 non-clamped stacks, so endpoint replication does not explain it.

This closes SliceEq posterior, target, and loss modifications. The useful posterior magnitude
and harmful dense fluctuations cannot be separated by the tested topology projection without
destroying the benefit. Threshold relaxation, confidence masks, extra morphology, and another
soft-target variant are prohibited.

H7.5 instead asks whether the current independent profile draws produce a high-variance
minibatch estimate of the unchanged paired acquisition risk. H7.3 found acquisition-active
samples in only 128/191 labeled and 120/192 unlabeled stacks, while 0.8427% of exact residual
pixels contribute 65.65% of the full gradient. Each existing L/U branch contains 12 samples,
so the four fixed 2x2 Gauss-Legendre nodes can each be assigned three times after an independent
seeded permutation. This preserves the 36-image batch and one view per sample. The resulting
batch-stratified estimator is preregistered for a no-training error gate against the current IID
sampler and a dense per-sample reference. It changes neither target representation nor loss and
is a stability component of SliceEq, not a standalone quadrature novelty claim.

## 2026-08-12 H7.5 complete implementation under user gate override

The user explicitly requested a complete H7.5 version before the preregistered zero-training
quadrature gate. This does not invalidate implementation, but makes the first full fixed-seed
run exploratory and gate-skipped. It cannot be described as a confirmatory test of H7.5.

The implemented `SliceEqSAQ` changes only profile sampling after the inherited 1k identity
warmup. Each 12-sample L/U branch receives the four fixed 2x2 Gauss-Legendre profile nodes
exactly three times, followed by independent seeded node-to-sample permutations. The nodes
are sigma `{0.5345299462, 0.7654700538}` and phase `{-0.1443375673, +0.1443375673}`. Assignment
does not inspect images, anatomy, masks, confidence, loss, or history.

The student batch remains 36 and every sample still receives one re-acquired view. Exact-GT
labeled fractional occupancy, detached LCC unlabeled fractional occupancy, soft CE+Dice,
EMA, ramp, optimizer, validation, and inference are unchanged. The new experiment/output id
is `SliceEqSAQ_PROMISE12`, so SliceEqOcc sources and results are not overwritten. Runtime logs
must show post-warmup node coverage 1.0 and maximum count deviation 0.0 for both branches.

## 2026-08-13 SliceEqOcc selected as the current final method

The user corrected the method identity: SliceEqOcc, not hard-target SliceEq, is the current
final method, and its Dice 0.844566 is accepted as a determined development result without
local re-verification. SliceEq is retained only as the hard-target predecessor. The user also
reports that the completed SliceEqSAQ trial has no material improvement, so H7.5 is closed and
SAQ is retained only as an appendix negative ablation.

The locked development chain is SliceEq 0.832603 to SliceEqOcc 0.844566, an absolute gain of
0.011963 (about 1.2 Dice points). This is the central method comparison: SliceEq is the hard-
target predecessor/ablation, while SliceEqOcc is the complete method whose fractional target
semantics form the paper's core. A compute- and batch-matched SliceHard rerun remains necessary
for formal causal attribution because the current Occ implementation also adds a labeled view.

The user further locks the no-Copy-Paste EMA baseline at Dice 0.78--0.80. The complete
development trajectory is therefore baseline 0.78--0.80 to SliceEq 0.832603 to SliceEqOcc
0.844566. Numerically this is a first-stage gain of approximately 0.0326--0.0526 and a second-
stage gain of 0.011963. These are accepted observations; the causal paper claim must still use
the matched matrix because the stages change batch size, labeled-view count, and teacher/BN
behavior in addition to the intended operator/occupancy factors.

The negative SAQ result is mechanically coherent. Under the current three-tap Gaussian
operator, continuous IID sampling yields center weights over approximately 0.4851--0.8552
with mean 0.6249. The four SAQ nodes yield only approximately 0.5325 and 0.7179, with mean
0.6252. Thus SAQ nearly preserves first-order average severity while deleting both tails.
Moreover, the four nodes are assigned across different anatomy samples, not evaluated within
the same stack. It balances batch-marginal profile counts but cannot reduce the conditional
acquisition-risk variance for a given anatomy. Increasing quadrature order is therefore not a
justified rescue.

## 2026-08-13 causal-attribution audit

The principal publication risk is no longer whether fractional occupancy is active. It is
whether the observed gain can be attributed to that mechanism. The code changes several
training factors together: baseline student batch 24 becomes 36; an exact-GT-derived labeled
view is added; labeled-derived to unlabeled batch composition changes; EMA teacher inputs and
student/teacher forward order change; and the teacher remains in train mode despite U-Net
BatchNorm and dropout. The original baseline also does not share SliceEqOcc's strict
self-training-only checkpoint/RNG contract.

The locked causal matrix is therefore B0, B0-36, ImgOnly-36, SliceHard-36, and Full
SliceEqOcc-36 for three seeds, followed by L-only/U-only fractional controls and extension of
Full plus the strongest matched control to five seeds. All methods must share a seed-specific
pretrain net+optimizer hash, teacher/BN policy, update count, selection rule, and test
postprocessing. The primary comparison is Full versus the strongest compute-matched non-paired
control; Full versus SliceHard isolates fractional occupancy. Original BCP and UniMatch remain
public comparators, while the no-Copy-Paste code must be named a BCP-derived EMA pseudo-label
scaffold rather than BCP.

The user-confirmed 0.844566 remains the development result. For a CVPR primary table, the
research protocol must still use a test-independent model-selection rule or an untouched
external evaluation. This is a reporting and evidence-design requirement, not a challenge to
the supplied value.

## 2026-08-13 novelty boundary and paper thesis

The literature audit excludes broad firstness claims for adjacent-slice augmentation,
partial-volume simulation, slice-profile modeling, synchronized image-label transforms, and
augmentation-dependent soft labels. BCP, UniMatch, AugSeg, ABD, beta-FFT, PV-SynthSeg,
SynthSeg, slice-profile estimation, and inter-slice synthesis cover those individual pieces.

The defensible intersection is narrower and stronger: a training-only non-invertible
through-plane acquisition operator is applied jointly to real neighboring MR signal and
exact/teacher-derived tissue occupancy, producing spatial fractional supervision for a 2D
semi-supervised model with unchanged single-slice inference. The paper's central statement is
that consistency under a non-invertible acquisition perturbation is well-defined only when
the observation and supervision are transformed by the same image-formation operator.

The recommended public name is SliceEq, with SliceEqOcc retained as the implementation and
ablation identifier. `Occ` means fractional occupancy, never occlusion. The working title is
“SliceEq: Acquisition-Equivariant Fractional Occupancy for Semi-Supervised MRI Segmentation.”

## 2026-08-13 next method hypothesis

No additional module is required before the causal matrix. If SliceEqOcc remains positive and
a method-level extension is desired, H7.6 is protocol-conditioned, scan-coherent SliceEqOcc.
It first holds the current profile marginal fixed while sharing one deterministic profile per
case and refresh window, isolating the fact that a real acquisition protocol is scan-level.
It then moves tap distances into physical coordinates using trustworthy spacing and adds
native-to-target profile composition only if real thickness/profile metadata exists. Spacing
must not be described as slice thickness.

H7.6 is evaluated primarily by robustness AUC under predeclared through-plane degradation,
cross-center/external performance, method-by-protocol interaction, and metadata-shuffle
controls. An ordinary in-distribution Dice increase without a protocol interaction or with an
equally strong metadata shuffle is only hyperparameterization, not a new contribution.

The full strategy, causal preregistration, H7.6 protocol, and novelty audit are stored at:

- `to_human/sliceeq_occ_cvpr_strategy_2026-08-13.md`
- `experiments/h7_slice_profile_reacquisition/cvpr_causal_ablation_protocol.md`
- `experiments/h7_slice_profile_reacquisition/h7_6_protocol_conditioned_scan_coherent_protocol.md`
- `literature/40_sliceeq_occ_cvpr_novelty_audit_2026-08-13.md`

## 2026-08-13 H7.6 data-contract audit and user-order amendment

The user prioritizes a real SliceEqOcc optimization before running the causal ablations. The
proposed protocol-conditioned, scan-coherent direction was therefore re-audited against the
actual input contract. The PROMISE12 H5 slices expose only `image` and `label`; the ordered
list exposes stable names of the form `CaseXX_slice_i`. No verified spacing, thickness, gap,
vendor, center, or scanner protocol reaches the training loader. Implementing a physically
conditioned profile now would therefore invent metadata and overstate the physical claim.

Reliable patient identity is sufficient to test the other half of the hypothesis. Current
SliceEqOcc independently samples a profile for every center slice, so two slices of the same
patient can be assigned contradictory virtual acquisitions during the same volume pass. H7.6
is split into H7.6a synthetic scan coherence, authorized now, and physical protocol conditioning,
which remains deferred until trustworthy metadata exists. The causal controls remain required
for the eventual paper claim but no longer block this one exploratory optimization run.

## 2026-08-13 H7.6a SliceEqOccSC implementation

`SliceEqOccSC` changes only the profile sampler. Within one training epoch, every slice of a
patient uses the same continuous `(sigma, phase)`. Labeled and unlabeled patient tables are
constructed independently. For each marginal, randomized stratification assigns one jittered
draw per patient stratum, and independent seeded permutations decorrelate sigma from phase.
The table refreshes every epoch. This preserves the original `[0.45,0.85]` sigma and
`[-0.25,0.25]` phase marginals, continuously covers their tails over training, and avoids
SliceEqSAQ's four-node discretization.

The root path, fixed Pre10000 net+optimizer checkpoint, seed 1337, 30k updates, 1k warmup,
effective 36 student views, original hard labeled anchor, labeled/unlabeled fractional losses,
EMA, optimizer, ramp, stack radius, validation, and inference are unchanged. The new output is
`SliceEqOccSC_PROMISE12`, so the user-confirmed SliceEqOcc result is not overwritten. New logs
measure within-case sigma/phase ranges, which must stay zero, as well as the inherited profile
and occupancy observables.

The independent train/test/utility entries and tensor/source-contract tests are complete. Five
new sources pass syntax compilation; all 48 repository source-contract tests pass with two
expected data-root skips; all five frozen SliceEqOcc parent hashes remain exact. Tensor tests
cannot run locally because Python has no `torch` and must be executed on the CUDA server. No
performance result is claimed yet.

## 2026-08-13 exact interpretation of the 36-view update

The loader batch remains 24 (`12 L + 12 U`). The additional 12 tensors are generated from the
same labeled stacks and make the student forward contain `12 original-L + 12 reacquired-L +
12 reacquired-U`. Fractional occupancy does not mathematically require 36 views. The extra
re-acquired labeled branch is, however, part of the confirmed SliceEqOcc objective: it supplies
exact-GT acquisition teaching without pseudo-label noise, while the separate original-L branch
preserves the clean central hard-mask anchor.

Deleting the extra 12 removes labeled acquisition teaching; replacing the original 12 removes
the hard anchor. Either creates a different method. If peak memory is the concern, the three
groups can be evaluated in sequential forwards with correctly normalized gradient accumulation
and one optimizer/EMA update; the objective still contains 36 effective views. H7.6a freezes
this design so scan coherence is its only intervention.

## 2026-08-14 H7.8 negative result and H7.9 AP-TNA implementation

The supplied H7.8 dual-anchor run is negative. Its validation peaks at 0.803741 at 20.8k
and falls to 0.786358 by the last supplied 29k record; its validation-selected test Dice is
0.832141. The implementation itself is consistent with the locked H7.8 objective, but that
objective replaces half of the retained unlabeled measurement loss with a native hard loss.
The native and measurement targets have almost identical categorical labels, so the run
weakens the rare fractional-occupancy signal while adding a largely redundant 48-view branch.
No fixed-ratio or ramp-cap rescue is authorized.

H7.9 tests the narrower acquisition-preserving transient native anchor hypothesis. The parent
SliceEqOcc loss remains intact as `Lsup + lambda * L_U_measurement`. A native-U auxiliary term
has coefficient `mu = 0.5 * lambda * (1 - lambda / lambda_max)`, where `lambda_max` is derived
from the existing baseline consistency setting. It therefore needs no new schedule parameter,
never attenuates the occupancy objective, and vanishes at the inherited ramp maximum.

The parent 36-view student forward remains unchanged. Native-U uses a separate 12-view
forward during which only student BatchNorm running-stat updates are disabled; dropout and
parameter gradients remain active. Its CUDA RNG is forked and restored, so the auxiliary
dropout draws do not advance the parent's global stream. The EMA teacher stays in train mode.
Validation and checkpointing are copied exactly from SliceEqOcc, as explicitly required by
the user: same five-case loader, metric implementation, 200-step frequency, mean Dice, strict
best-performance comparison, raw student checkpoint, and periodic saves.

The new isolated entries are `train_sliceeq_occ_aptna.py` and
`test_sliceeq_occ_aptna.py`, under experiment id `SliceEqOccAPTNA_PROMISE12`. Syntax checks
pass and all 67 repository source-contract tests pass with two expected missing-data skips.
No local training result is claimed; the CUDA run remains external.

## 2026-08-14 H7.9 APTNA negative result

The complete external APTNA log and test report are now archived. Under the unchanged validation
rule, APTNA peaks at 0.804265@28.8k, ends at 0.772908@30k, and its selected checkpoint obtains
test Dice 0.829420. The user separately reports approximately 0.835 for the penultimate
periodic checkpoint; if this is the expected 27k checkpoint, its logged validation is 0.781877.
That observation is useful diagnostically but cannot replace the locked validation selection.
Both readings remain below the accepted SliceEqOcc development result 0.844566.

The late curve oscillates rather than simply decaying: validation over 25k--30k has mean
0.788313 and population standard deviation 0.007490. More importantly, APTNA progressively
separates from the matched SliceEqOcc trajectory during the interval in which the native bridge is
active. Its paired mean difference is -0.008946 over 10.2k--15k, -0.016886 over
15.2k--20k, and -0.022756 over 20.2k--25k. Once the auxiliary coefficient nearly vanishes, the
weights do not return to the parent trajectory.

The implementation contract rules out the major DA confounds: the complete occupancy loss remains,
the main student forward stays at 36 views, auxiliary BN running statistics are isolated, the CUDA
RNG is restored, EMA remains train-mode, and validation is unchanged. The negative is therefore a
real failure of the native-anchor hypothesis. DA and APTNA jointly close all fixed/transient native
ratio, ramp, and cutoff rescues.

## 2026-08-14 H7.10 operator-reliability result

After SAQ, SC, CAP, DA, and APTNA, the bounded unresolved mechanism is no longer profile sampling
or an extra supervision branch. It is whether the train-mode EMA teacher's stochastic variation
is mixed by the SliceEq operator and treated as anatomical fractional occupancy. H7.10 tests two
case-agnostic operator-space constructions using only the 191 labeled training slices at the
18k/24k/30k parent checkpoints.

The tested elementwise stack-shared SCT construction is negative. It passes 0/7 patients at every
checkpoint: residual-variance reduction is only 5.75%--7.99%, residual-Brier reduction is negative
at all three stages, and full-image Brier is 1.7%--2.6% worse. This closes SCT for this project
without claiming that every possible shared-stochasticity construction is false.

Acquisition-Aligned Dropout Uncertainty (ADU) is exploratory positive. Operator-space JS/error
correlation is 0.61--0.68, the top-20% JS region contains 2.26--2.69 times the remaining error, and
normalized weighting reduces effective risk by 5.26%--7.34% while retaining 0.963--0.965 mean
weight on exact fractional support. All 84 patient-checkpoint-pair records are finite and
non-degenerate. A deterministic post-run audit gives checkpoint patient passes 7/7, 7/7, and 5/7,
with all seven named training patients passing the cross-checkpoint median rule.

This result is deliberately classified as an exploratory mechanism screen. The uploaded analyzer
predates final hardening for deterministic CuDNN, support-pixel pooling, exact checkpoint naming,
and stricter quality aggregation. The archived raw ADU pair records support the conservative
reaggregation, but paper-level confirmation requires rerunning the final frozen analyzer. The
weighted-risk reduction establishes useful reliability ranking on the same `q_bar` error map; it
does not show that `q_bar` itself has lower unweighted Brier error.

## 2026-08-14 H7.11 SliceEqOcc-ADU implementation

H7.11 directly implements the sole authorized exploratory successor. The primary train-mode EMA
forward, hard argmax plus per-slice 2-D LCC, parent profile draw, exact-GT labeled branch, complete
unlabeled occupancy coefficient, 36-view student forward, optimizer/EMA update, and inference are
unchanged. A second train-mode teacher forward is executed in a target-GPU RNG fork after the
primary pass; every EMA buffer is restored afterward. Consequently, persistent teacher BN state
and the parent RNG trajectory reflect exactly one teacher forward.

Both hard pseudo stacks are projected with the same sampled profile. Their mean occupancy
`q_bar=(q1+q2)/2` is trained with the existing soft CE and squared soft Dice, weighted continuously
by `w=1-JS(q1,q2)/log(2)`. Agreement on a modeled acquisition-derived fractional mixture therefore
retains full weight. There are no patient rules, thresholds, new trainable parameters, new student
views, or inference operations. The ADU package does couple two changes--two-pass target averaging
and JS weighting--so a positive full run requires a compute-matched `q_bar`-only (`w=1`) control
before an improvement can be attributed specifically to reliability.

The runtime entry now rejects changes to the frozen PROMISE12 recipe and verifies the shared
Pre10000 checkpoint SHA-256. The parent validation block is byte-identical. The matched parent best
validation Dice is 0.817373, and the exploratory H7.11 pass threshold is locked at 0.820373. The
validation-selected ADU checkpoint is tested once whether positive or negative; test results cannot
select a periodic checkpoint or authorize a rescue. Static contracts and syntax checks pass
locally. The remaining pre-training hard gate is the CUDA BN/RNG isolation test, which must run and
report `ok` rather than `skipped` on the training server.

## 2026-08-14 H7.11 SliceEqOcc-ADU neutral result

The supplied ADU log covers iteration 200 through 27.6k and has no training-finished marker. Its
best validation Dice is 0.815701@27k, below both the matched SliceEqOcc best 0.817373 and the locked
ADU pass threshold 0.820373. The supplied performance file tests iter20.8k and reports Dice
0.843335, Jaccard 0.731099, HD95 4.045004, and ASD 2.150427. This essentially catches the accepted
SliceEqOcc development Dice 0.844566 but does not improve it; the surface metrics also degrade.

The full-run diagnostics explain why the positive labeled gate did not transfer. Operator-space JS
is nonzero on only about 0.619% of pixels during 1k--10k and 0.370% during 24k--27.6k. Corresponding
whole-image mean weights are 0.998742 and 0.999280, so the reliability intervention is almost the
identity objective. Two stochastic passes can rank a small disagreement subset but remain jointly
wrong on systematic pseudo components. H7.11 is therefore closed without temperature, threshold,
pass-count, q-bar-only, or schedule rescue.

## 2026-08-14 H7.12 Slab-Coherent Pseudo Occupancy implementation

The remaining bounded method-level defect is upstream of the SliceEq operator. Parent SliceEqOcc
filters each teacher slice with an independent 2-D largest component, then interprets the three
retained masks as one latent source slab. SCPO instead retains one fixed 26-connected foreground
component over each unlabeled three-slice hard pseudo slab before applying the original paired
profile. This can remove slice-jumping systematic components that dropout disagreement cannot see.

SCPO changes no network, model parameter, teacher mode, number of forwards, student batch, exact-GT
labeled branch, profile distribution, objective coefficient, validation rule, or inference graph.
It is not claimed as a novel generic 3-D connected-component or inter-slice consistency method; its
defensible role is the narrower acquisition-ordering principle that a coherent latent occupancy must
be formed before a non-invertible through-plane measurement. The protocol locks best-validation
success at 0.820373 and forbids connectivity, morphology, slab-width, and component-threshold rescue.
All syntax and static source-contract checks pass locally; PyTorch tensor tests remain an external
pre-training requirement.

## 2026-08-14 H7.12 SCPO neutral result

The complete SCPO run peaks at validation Dice 0.817824@28.8k and ends at 0.802209. Its locked
success threshold was 0.820373, so the apparent +0.000451 over the parent best is neutral rather
than a confirmed improvement. The validation-selected checkpoint tests at Dice 0.842378,
Jaccard 0.729612, HD95 4.438169, and ASD 1.583224. The user separately observed approximately
0.849 at a preceding periodic checkpoint; without a checkpoint-specific artifact this remains a
post-hoc observation, and by the user's comparison it matches rather than exceeds the parent.

The mechanism is inactive at useful scale. SCPO changes only 0.004527% of U-slab pixels on average,
with a foreground-mass ratio of 0.999904 and activity in 10.17% of slabs; late activity is still
lower. Consequently, SCPO is effectively SliceEqOcc plus training noise. This closes all
connectivity, morphology, slab-width, and component-threshold variants. The next experiment must
first demonstrate a nontrivial intervention over the complete objective rather than target a rare
pseudo-label exception.

## 2026-08-14 H7.13 Ordered Acquisition-Appearance Consistency

H7.13 returns to the original SliceEqOcc parent and addresses intervention coverage without
replacing its successful target. The teacher still forms a hard/LCC pseudo slab and the same sampled
profile still produces both the re-acquired U image and fractional occupancy. Only afterward, every
post-warmup re-acquired U student image receives a fixed-range, per-sample monotonic composition of
gamma, positive contrast, and brightness. Coordinates and occupancy are unchanged. The exact-GT
original/re-acquired L branches, 36-view student batch, one teacher/student forward, full occupancy
loss, consistency ramp, validation, checkpoint selection, and 2-D inference are inherited directly
from the parent.

This is intentionally an ordered weak-to-strong extension, not a generic augmentation claim. A
target-changing acquisition operator is applied jointly to signal and occupancy first; a
target-invariant appearance operator is then applied to signal only. Fixed independent RNG and
runtime call-order assertions prevent the new transform from consuming the parent profile/dropout
streams or touching the L branch. Best validation must reach 0.820373 and the validation-selected
checkpoint is tested once. If negative, appearance ranges, longer chains, noise, blur, masking,
mixing, and adaptive-strength rescues are closed.

## 2026-08-14 H7.13 OAAC validation proxy and test-selected oracle

The complete seed-1337 run reaches best validation 0.834863@23.8k and finishes at 0.831964. The
user clarified that multiple checkpoints were inspected on the local test split and iter27k was
retained as the test-Dice maximum. It reports Dice/Jaccard/HD95/ASD
0.849538/0.740985/3.554760/1.868299. Conditional on the selected SliceEqOcc and OAAC oracles, the
apparent Dice difference is +0.004972; 7/10 cases improve and the median paired delta is
+0.0026095. These do not estimate an unbiased method gain. The robust mechanism fact is that all
146 appearance records have active_samples=1.0 and mean normalized image change 0.055178.

The curve does not show severe late collapse. Best-to-final is only -0.002899, and 27k--30k
validation has SD about 0.004865 under a constant LR=0.01. Testing validation-best or analyzing one
fixed 24k/27k/30k average can complete internal development, but neither can restore independence
after test-max checkpoint selection. LR decay/SWA requires a matched SliceEqOcc retrain and fresh
outer/hidden evaluation; it is an optimization recipe, not a contribution. No additional target,
augmentation, NMS, TTA, threshold, or test-checkpoint search is authorized.

## 2026-08-17 H7.14 no-architecture optimization decision

The user confirms that 0.849538 is the maximum after evaluating every retained OAAC checkpoint on
the ten-case test split. This closes ordinary early-stopping and checkpoint-selection gains: no
unseen raw checkpoint remains, and the current test split is now purely a development oracle.
Exceeding 0.85 on it may be useful for engineering, but cannot improve the paper's evidence class.

The remaining full-coverage defect is optimizer/readout variance. OAAC inherits constant SGD
`lr=0.01` for all 30k updates and saves only individual raw-student iterates. Its 24k, 27k, and 30k
periodic checkpoints have validation Dice 0.828298, 0.828406, and 0.831964, while nearby validation
points oscillate around a high plateau. This is the setting for which stochastic/trajectory weight
averaging was designed: average positions in one flat basin and retain one-model inference.

H7.14 therefore locks one zero-training candidate: the equal 24k/27k/30k average, with BatchNorm
statistics recomputed once on the exact training-only OAAC 36-view student distribution and dropout
disabled. No other window, coefficient, greedy soup, or test-driven constituent is permitted. The
unchanged validation must exceed 0.834863 before the averaged model is tested once. A positive value
on the current test remains development-only.

If this sparse average is neutral, window search is closed. The only remaining no-architecture
training candidate is a separate matched PolyLR0.9 rerun for both SliceEqOcc and OAAC. It must not be
bundled with SWA in the first run. Weight averaging, PolyLR, and any EMA readout are implementation
recipes rather than contributions; the paper contributions remain paired through-plane
re-acquisition, fractional occupancy, and OAAC's ordered acquisition-then-appearance composition.

## 2026-08-17 user constraint: tune method parameters, not baseline infrastructure

The user explicitly excludes learning-rate, optimizer, EMA mode/decay, consistency/ramp, batch,
training-length, validation, and inference changes. H7.14 is therefore deferred. The next search is
H7.15, which changes only OAAC/SliceEq method parameters while retaining the entire baseline-derived
training contract.

Current OAAC is not obviously saturated: all 146 diagnostic records are active, but mean normalized
absolute image change is only 0.055178 and the 95th percentile of the logged batch mean is about
0.071082. Validation also stays high late rather than exhibiting the severe collapse seen in DA or
APTNA. The highest-prior first candidate is therefore `Strong-all`: multiply all OAAC ranges by
1.25 while preserving their ratios and the acquisition-then-appearance order.

The bounded Round-1 comparison is reference scale1/probability1 versus scale0.75/probability1,
scale1/probability0.75, and scale1.25/probability1. Mild-all and Current-mix have approximately the
same expected total perturbation but distinguish per-view severity from coverage, which makes the
search scientifically interpretable. If none improves validation 0.834863, retain current OAAC.

Only after OAAC is locked may the SliceEq profile use a joint 0.85/1.15 severity bracket. Sigma and
phase are not independently swept, because the H5 data contain no physical profile metadata and a
large Cartesian grid would be test-set tuning rather than a contribution. Even if the final
development score exceeds 0.85, confirmation still requires a frozen selection rule on fresh
hidden/external data.

## 2026-08-17 isolated Strong-all implementation and checkpoint cadence

The original SliceEqOcc and OAAC sources remain byte-identical to their archived versions. A new
isolated H7.15 successor implements the first-run `Strong-all` candidate by scaling the three OAAC
bounds jointly by 1.25 while keeping application probability 1.0. Only this successor archives the
raw student state every 1000 rather than 3000 iterations. Validation remains every 200 iterations,
`unet_best_model.pth` follows the identical comparison, and no optimizer, EMA, loss, batch, RNG,
training-length, or inference behavior changes. The denser archive is diagnostic and does not
authorize checkpoint selection by the development test.

## 2026-08-17 Strong-all positive result and final scale bracket

The isolated scale1.25 run is positive under the unchanged selector: validation improves from
0.834863 to 0.836475, and the validation-best checkpoint reports development test Dice 0.851960.
The transform remains fully active and mean normalized appearance change rises almost exactly 25%,
from 0.055178 to 0.068954. This is stronger evidence than the original OAAC test-max result because
the supplied model identity follows validation selection.

The scale1.25 point is not yet demonstrably optimal. Only scale1.0 and1.25 have been measured;
however, the validation gain is only +0.001612, late 27k--30k SD rises to 0.012075, and the
development case delta versus the earlier OAAC oracle wins only 5/10 with negative median. The
scientifically bounded response is one outer scale1.50 experiment, not a dense grid. H7.16 uses
gamma +/-0.30, contrast +/-0.225 and brightness/span +/-0.15 with probability1. If unchanged
validation does not exceed 0.836475, scale1.25 is retained and OAAC-scale tuning closes.

## 2026-08-17 OAAC parameter ceiling

The final scale1.50 bracket is active but neutral. Its best validation is 0.835796, below the
scale1.25 winner 0.836475. Test Dice changes by only +0.000099 to 0.852059, while Jaccard, HD95 and
ASD worsen; only 2/10 cases improve and the median Dice delta is -0.002196. Mean normalized
appearance change rises to 0.082723, so the lack of gain is not caused by an inactive transform.

The observed response is now `scale1.0 -> 0.834863 val`, `scale1.25 -> 0.836475`, and
`scale1.50 -> 0.835796`. Scale1.25 is therefore the selected practical local optimum. Further OAAC
scale, per-component, probability, seed or checkpoint search is closed. This is not a mathematical
global-optimum proof, but it is the correct experimental stopping point: additional same-split
tuning would mostly fit five-case validation noise and an already queried ten-case development set.

## 2026-08-17 final method, intellectual lineage and external-validation decision

The final selected method is `SliceEqOccOAACStrong`, paper-facing as SliceEqOcc-OAAC. Strong means
the validation-selected OAAC scale1.25 configuration: log-gamma +/-0.25, log-contrast +/-0.1875,
brightness/span +/-0.125 and application probability1. It is not a larger network. The complete
method has one ordered semantic chain: paired target-changing through-plane re-acquisition with
fractional occupancy, followed by a coordinate-preserving target-invariant appearance transform
only on the unlabeled student image. PROMISE12 remains development evidence; the
validation-selected single-seed development Dice is 0.851960.

The model lineage is now explicit in the Chinese paper outline. U-Net supplies the 2-D backbone;
Mean Teacher supplies EMA pseudo supervision; the local baseline is a BCP-derived scaffold with
Copy-Paste removed; ICT and Inter-Slice Augmentation establish prior interpolation and neighboring-
slice synthesis; SynthSeg motivates acquisition/resolution and partial-volume simulation; UniMatch
and AugSeg motivate weak-to-strong appearance coverage. The paper contribution is not any one of
these inherited components. It is the operator-aligned exact/pseudo fractional occupancy and the
ordered composition of a target-changing acquisition before a target-invariant U-image appearance
transform.

The requested external dataset is standardized to its official name, MM-WHS 2017. Its MRI subset
is prospectively assigned to test cross-organ, seven-structure and multi-center generalization. The
frozen Strong and SliceEq profile parameters transfer without MM-WHS tuning. A proposed low-label
protocol reserves four of the twenty labeled MRI training volumes for validation and trains with
four labeled plus twelve label-hidden volumes. Where available, the official forty-volume MRI test
is used only after selection is locked. These are planned experiments, not current findings.

The user explicitly declines multi-random-seed validation. Every matched method uses fixed
seed1337, with no seed search or best-seed selection. Patient-level paired effects, confidence
intervals, matched causal controls and MM-WHS external validation provide the available evidence;
optimization-seed variance remains unmeasured and must be stated as a limitation.

## 2026-08-18 profile-module deepening: from heuristic mixture to relative acquisition

The production SliceEq profile is not a fixed `[0.2,0.6,0.2]` blend. It samples a Gaussian width
from 0.45--0.85 and a sub-slice phase from +/-0.25 for every sample, then normalizes its values at
the three offsets. The familiar ratio is exactly the representative symmetric point at phase zero
and sigma about 0.6746. Thus learning one globally optimal ratio would replace the intended
acquisition-risk distribution with a small-validation hyperparameter and is not a meaningful paper
contribution. SAQ and CAP already showed that reorganizing the same marginal profile samples does
not reliably improve the method.

Nonlinear or pixel-adaptive fusion is also rejected. The current positive linear mixture is the
discrete counterpart of a through-plane PSF integral and, crucially, defines the same operator on
MRI intensities and one-hot tissue occupancy. An attention/MLP/max fusion would not have a unique
matching occupancy transform, would risk identity collapse under the segmentation objective, and
would collide with established cross-slice attention and learned-augmentation work.

H7.17 therefore proposes Self-Calibrated Relative Profile SliceEqOcc. It treats an observed MRI
slice as already convolved with a native profile, estimates a case/protocol-level native blur proxy
from training-only raw-volume internal patch statistics, and composes only an additional physically
legal degradation. For a Gaussian approximation, the added variance is
`s_delta^2=max(s_target^2-s_native^2,0)`. Three weights are obtained by integrating this relative
Gaussian over slice bins rather than point-sampling its centers, and volume endpoints renormalize
only valid support rather than duplicating a clamped slice. The same weights still generate the
image and exact/pseudo fractional occupancy; all final OAAC-Strong training and inference behavior
is frozen.

The targeted literature search found slice-profile estimation for MRI reconstruction, acquisition-
parameter-conditioned segmentation, and network-level cross-slice attention, but not this complete
semi-supervised paired-occupancy construction. The safe novelty is the combination, not a claim to
invent profile estimation or acquisition conditioning. Because current H5 files do not carry
reliable thickness/profile provenance, no training implementation is authorized until raw NIfTI/
DICOM recovery, synthetic-kernel estimation accuracy, estimator stability and three-tap support
gates pass. Failure of those gates closes the direction rather than replacing physical calibration
with validation-driven ratio search.

## 2026-08-18 H5-only pivot: axial-response calibrated profile

The user cannot obtain the original NIfTI/DICOM. H7.17 therefore fails its own provenance gate and
is closed without implementation. Processed H5 images may still define neighboring observations,
but they cannot support scanner-PSF, thickness or millimeter calibration claims.

The parent profile remains mathematically interpretable in H5 space. For first difference
`g1=(x_plus-x_minus)/2`, second difference `g2=x_minus-2*x0+x_plus`, phase moment
`m1=w_plus-w_minus`, and mixing moment `m2=w_minus+w_plus`, the image residual satisfies the exact
identity `A_w(X)-x0=m1*g1+0.5*m2*g2`. Hence one sampled profile produces different effective
perturbation magnitudes on stacks with different axial response. This is a more defensible H5-only
problem than pretending to infer the scanner profile.

H7.18 proposes Axial-Response Calibrated Profile (ARCP). A normalized 2x2 Gram matrix of `g1,g2`
is computed per training stack and aggregated patient-first into a fixed image-only reference. For
each ordinary parent profile, the candidate analytically scales the profile along the identity ray
so its normalized response approaches the reference while the center weight remains inside the
parent's observed support. This preserves nonnegative weights, the exact same weights for image and
occupancy, all network/training/inference behavior, and introduces no learned parameter or label/
model-dependent profile policy.

ARCP is not guaranteed to help: rapid axial response often occurs at the apex/base where SliceEq's
fractional signal is valuable, so calibration could attenuate useful supervision. The pretraining
gate therefore requires nontrivial activity and reduced response dispersion while retaining at
least 90% of exact labeled fractional support and protecting first/last axial thirds. Failure closes
the direction without alpha/epsilon/range rescue. Bin-integrated Gaussian discretization is a
lower-risk descriptive control; adversarial profile optimization is deferred because it adds inner
forwards, can amplify pseudo errors, and strongly overlaps adaptive/adversarial augmentation work.

The H7.18 implementation is isolated from the final parent. A new image-only utility computes the
patient-balanced reference and calibrates weights; a wrapper intercepts the two existing paired
re-acquisition calls, then delegates image/occupancy fusion and U-only OAAC to the unchanged Strong
implementation. The inherited diagnostics receive the calibrated weights, while the inherited
training loop still owns teacher/student forwards, optimizer/EMA, validation and 1000-step archive
behavior. A separate no-checkpoint analyzer uses all training images, labels only for the locked
first191 training slices, and a fixed nine-profile diagnostic grid. Static contracts pass 8/8 and
all new files compile; tensor tests and the actual H5 gate remain external because the desktop
Python lacks NumPy/PyTorch and the data are not stored locally.

## 2026-08-18 H7.18 external result: active calibration, no final replacement

The complete seed-1337 ARCP run is healthy through 30k and tests the validation-best checkpoint.
Best validation is `0.838425@29.8k`, which is `+0.001950` above the final Strong parent but
`0.001050` below the preregistered `0.839475` pass line. Test Dice/Jaccard/HD95/ASD is
`0.851062/0.743164/6.217004/2.123881`, versus Strong
`0.851960/0.745347/3.228864/1.307063`. ARCP therefore does not replace the selected method.

The negative decision is not caused by an inactive implementation. Across 146 diagnostics,
69.46% of samples are active, mean `|alpha-1|` is 0.132262, and mean absolute center-weight shift
is 0.014669. Yet the logged batch-level reference mismatch falls by only about 5.95%, and only
54.79% of records move closer to the reference. The intervention is a real but inconsistent
profile redistribution, not a reliable acquisition-risk normalization.

The result sharpens the mechanism story: anatomy-dependent profile-effect magnitude is real, but
forcing that magnitude toward an H5-derived reference is not equivalent to preserving the most
useful acquisition signal. Large axial residuals can occur at apex/base transitions where
fractional occupancy is informative. Without raw acquisition metadata, further alpha, reference,
support, epsilon, or profile-grid rescue is closed. Final SliceEqOcc-OAAC-Strong remains frozen;
ARCP is appendix-level neutral/negative evidence.

## 2026-08-18 post-ARCP profile-module convergence

A new targeted search considered bin-integrated and five-tap profiles, discrete Gaussian
semigroups, learned/adversarial weights, nonlinear or edge-aware fusion, and task-aware sampling.
None is a clean direct successor under the H5-only and frozen-training constraints. Three-bin
integration changes the weight vector substantially and may double-integrate an already acquired
H5 slice; five taps change the teacher input from 36 to 60 slices and therefore confound support
with train-mode BN/dropout and compute. A fixed finite-support profile cannot form a nontrivial
continuous convolution semigroup. Nonlinear/attention fusion loses the unique shared
image--occupancy measurement operator, while adversarial profile selection is both literature-
crowded and liable to amplify pseudo-label boundary errors.

The profile can be written with neighbor mass `b=1-w0` and directional phase
`r=(w_plus-w_minus)/b`. This exposes that the current Gaussian sigma/phase jointly control both
moments, and that ARCP changed only `b` per anatomy stack. The sole remaining conditional candidate
is H7.19 Robust Moment-Profile Design: use only exact labeled-training masks to design one global,
phase-symmetric distribution over the parent's profile grid, maximizing worst patient/index-third
retained fractional information while matching parent profile moments and per-stratum image
residual budgets. It never reads model predictions, validation or test and never changes the
paired convex operator.

H7.19 is not authorized for training. A locked seven-patient leave-one-patient-out gate must show
at least 10% held-out worst-stratum utility gain, distribution stability, unchanged image/moment
budgets and no material hard-target-flip increase. This is a final boundary probe, not a new search
tree. If it fails, all further PROMISE12 profile design is closed and the final Strong method moves
directly to MM-WHS evidence construction.

## 2026-08-18 H7.19 direct-training override and implementation

The user explicitly chose to skip the LOPO/zero-training gate and requested a complete training
implementation. The original gate remains the scientifically preferred protocol and is retained in
the record; it was not passed. Any H7.19 run is therefore exploratory and cannot by itself establish
cross-patient stability of the designed profile distribution.

The direct successor computes exactly one global distribution before training. It reads only the
first191 labeled-training image/label H5 slices, verifies they form seven complete patients, and
evaluates a fixed 21x21 midpoint grid over the parent sigma/phase support. Exact occupancy defines
retained fractional information for 21 patient-by-index-third strata. A two-stage constrained
optimizer first maximizes the worst expected stratum utility, then minimizes KL to the parent while
retaining 99% of that robust optimum. Phase symmetry, parent neighbor-mass moments, normalized image
residuals, density ratio, and entropy remain hard constraints. Failure to find a verified feasible
solution aborts rather than relaxing them.

The resulting `q` and its data/protocol/distribution hashes are written atomically to
`mpd_profile_design.json`, after which the wrapper replaces only the parent's profile sampler and
immediately executes the unchanged OAAC-Strong 30k path. Both exact-L and pseudo-U draw from the same
frozen q, and every sampled weight vector still acts identically on image and occupancy. Parent
network, Pre10000 state, seed, optimizer/LR, EMA train mode, hard/LCC teacher, loss/ramp, batch36,
OAAC1.25, validation, 1000-step archives and 2-D inference remain untouched. This preserves a clean
profile-only intervention even though its prospective gate was waived.

The first direct startup exposed a definition edge case before any training step: at least one
patient/index-third had no pixel whose neighbor labels differed from the center. Its RFI denominator
is zero for every profile. Treating such a stratum as utility zero would force the robust objective
to a meaningless zero and cannot guide q. The implementation now excludes structurally empty strata
only from the max-min RFI set, retains them in all image-residual constraints and reports them in the
artifact, while requiring every patient to contribute at least one active stratum. This is a
mathematical-domain correction, not a validation-driven relaxation.

## 2026-08-19 H7.19 corrected result: validation-neutral, checkpoint-specific test-positive

The user clarified that the first performance attachment was the wrong file. The training log and
all mechanism conclusions are unchanged, but the corrected report has SHA-256 `49db4ebc...f6e97c`,
evaluates `iter_29000.pth`, and reports Dice/Jaccard/HD95/ASD
`0.854573/0.749330/3.256519/1.324697`. The superseded `unet_best_model.pth` report with hash
`52ae817d...3077` and Dice `0.848952` is provenance only and is not the latest MPD result.

The corrected result is numerically important. MPD-29k exceeds final OAAC-Strong by `0.002613` Dice
and `0.003983` Jaccard, with only `0.027655/0.017634` worse legacy HD95/ASD. It is the highest
currently observed PROMISE12 development Dice. Relative to the superseded MPD checkpoint it gains
`0.005621` Dice, wins 5/10 cases, has median paired change `+0.001653`, and receives its largest
increase from Case36 (`+0.032308`). The method therefore cannot be described as simply ineffective.

The selector evidence points in the opposite direction. MPD's unchanged validation best is still
`0.836008@25.8k`, just below Strong `0.836475@29.4k`, while the corrected 29k checkpoint has validation
only `0.828270`. The same run thus gives a validation-neutral/slightly negative method result and a
checkpoint-specific positive test result. Because PROMISE12 test has participated in development,
the 29k number is a development maximum, not a validation-selected or unbiased paper result.

Mechanistically MPD is nontrivial: 20/21 patient-index strata are active, entropy retains98.83% of
the parent, maximum density ratio is1.608, phase symmetry is exact, runtime center weight is about
0.615, and late validation is more stable than Strong. The corrected test result now supports the
narrower possibility that this redistribution can learn a useful alternative solution, not merely
stabilize a weaker basin. It does not establish that RFI is a generally better selection objective.

The user subsequently clarified that final selection should follow the highest observed checkpoint
performance and need not use validation-best identity. Under that explicit project criterion,
MPD-29k is a positive result and replaces OAAC-Strong as the final selected method: `0.854573` versus
`0.851960` Dice. The complete selected method is therefore SliceEqOcc-OAAC-Strong-MPD. The distinction
between the user selection rule and conventional validation-only selection remains disclosed, but it
no longer blocks internal method selection.

The method is now frozen. More checkpoint testing or retuning the grid, axial partition, entropy,
moments, density cap or RFI formula is unnecessary and would weaken the method story. The planned
next experiment is direct MM-WHS transfer of frozen MPD, with OAAC-Strong retained as the immediate
profile-sampler control.

## 2026-08-19 H7.19 mechanism synthesis

Moment decomposition makes the positive MPD result interpretable. For neighbor mass
`b=w_minus+w_plus` and directional mass `d=w_plus-w_minus=br`, the exact image residual is
`A_w(X)-X0=d*(X_plus-X_minus)/2 + b*(X_minus-2X0+X_plus)/2`. The uniform parent grid has
`E[b]=0.375076`, `E[b^2]=0.149946` and `E[d^2]=0.014946`. The learned q uses the allowed budget in a
specific direction: `0.382577/0.152546/0.014647`, corresponding to `+2.00%/+1.73%/-2.00%`.

Thus MPD slightly strengthens neighbor integration while reducing directional-shift energy. Phase
symmetry gives mean weights about `[0.1913,0.6174,0.1913]`, versus the parent's
`[0.1875,0.6249,0.1875]`, but 98.83% entropy retention means the benefit is not a new fixed ratio.
The distribution still supplies diverse profiles and reallocates probability toward those that
create exact fractional occupancy in weak patient/index strata without changing the center hard
semantic class. This predicts more useful partial-volume boundary supervision with less directional
pseudo-target noise.

The mechanism is complementary to OAAC: MPD changes which target-changing acquisition operators are
sampled, while OAAC changes only target-invariant post-acquisition appearance. Network, loss, EMA,
batch and inference are identical, so the final `+0.002613` Dice and `+0.003983` Jaccard are most
parsimoniously attributed to acquisition-risk redistribution. MPD is therefore stronger than the
uniform-profile parent both methodologically and numerically, while remaining acquisition-inspired
rather than scanner-PSF calibrated.

## 2026-08-19 post-MPD componentwise optimization audit

The successful MPD pattern is not “add another module”; it replaces an implicit heuristic probability
law with one global, training-only, constrained distribution while preserving the paired operator and
the whole model contract. Applying that criterion to the remaining pipeline leaves one high-priority
candidate. The current TwoStream sampler is slice-uniform within L and U, so a patient contributes in
proportion to its slice count and axial regions with more slices dominate SGD. This is misaligned with
MPD's explicit patient×index-third robust design.

The next analysis-only hypothesis is Patient–Axial Acquisition-Risk Sampling: keep 12L/12U and the
36-view student batch, first select a patient uniformly, then select an axial third from one global
exact-L-designed distribution, then a slice within that patient-third. The distribution is frozen and
transferred to U using only case identity and relative slice index; U labels, model confidence and
test cases are never used. MPD, OAAC1.25, teacher targets, losses, EMA train mode, LR, validation and
2-D inference remain unchanged. This is acquisition-opportunity sampling, not generic online hard
mining.

OAAC joint-parameter distribution design is the second candidate, but automatic/teacher-guided
augmentation is literature-crowded and its expected in-domain headroom is lower after the completed
1.0/1.25/1.5 severity bracket. Endpoint valid-support projection is physically interpretable but has
limited coverage. Pseudo-label repair, uncertainty weighting, loss-ratio/native anchors, profile-grid
retuning, geometry policy search and baseline infrastructure changes remain closed. The two live
candidates must be evaluated one at a time and never stacked in their first run.

## 2026-08-19 H7.20 PARS implementation

The first componentwise candidate is now implemented as an isolated successor rather than a parent
edit. H7.20 preserves the original `TwoStreamBatchSampler` for iterations0--999, so the supervised/
teacher warmup is identical. From iter1000 onward, L and U each cycle through a private random
permutation of patient IDs, sample a single frozen three-index-third probability law, and then sample
uniformly within the selected patient-third. The epoch remains15 batches and every batch remains
12L+12U; MPD, OAAC, student36, optimizer/LR, EMA train mode, target/loss/ramp, validation,1000-step
archives and2-D inference are inherited.

The axial law is not learned from segmentation errors. Exact-L statistics compute the expected
per-sampled-slice retained fractional information under the already frozen MPD distribution. A
two-stage SLSQP maximizes the worst active normalized patient-third exposure, then selects the
closest-to-parent q by KL while retaining99% of that optimum, bounding density ratio at1.5 and
retaining90% parent entropy. U contributes only filenames/case IDs/index ranks. There is no U-label,
prediction, confidence, loss, uncertainty, validation or test feedback.

This mechanism is deliberately narrower than ARCO/PH-Net-style generic stratified or hardness
sampling. The defensible novelty is aligning the *support distribution* of a paired through-plane
image–fractional-occupancy risk with exact acquisition opportunity. No claim of first balanced,
patient, axial, hierarchical or hard sampling is permitted. Six static contracts and four numerical
tests pass locally; CUDA smoke and the one locked30k run remain pending.

## 2026-08-24: one-method paper framing and evidence audit

The paper identity is now locked to one complete method, **SliceAlign-MPD**, corresponding exactly to `code/train_sliceeq_occ_oaac_strong_mpd.py`. Paired through-plane image/occupancy construction, fractional supervision, ordered appearance perturbation, and moment-constrained profile design are internal stages of one training objective. They must not be presented as several proposed methods. PARS and later candidates are excluded from the paper.

No local training process or resumable experiment artifact was found. The next unbiased evidence remains the frozen MM-WHS MRI evaluation, followed by the compute-matched causal ablation. PROMISE12 is closed to further tuning: `0.854573` is retained only as a test-informed development observation, while the MPD validation-selected report is `0.848952`. The current paper draft leaves the primary result cells blank rather than laundering the development checkpoint into a test claim.

The experiment structure was derived from the local BCP/FLDR papers and verified CVPR papers: matched main comparisons, a second dataset, cumulative compute-matched ablation, mechanism analysis, parameter/design audit, patient-level statistics, efficiency, and optional independent seeds. A three-seed replication protocol (`1337/2024/3407`) is prospectively recorded but remains optional, unscheduled, and subordinate to MM-WHS and ablation.

The manuscript project was subsequently converted from a generic two-column article to the official CVPR 2026 Author Kit. The project is self-contained in review mode and includes the official style/BibTeX files, `preamble.tex`, `sec/`, `fig/`, and `main.bib`; this formatting change does not alter the method or evidence boundary.

## 2026-08-24: CVPR narrative refinement

The manuscript now exposes only two coupled scientific decisions: (1) a slice-profile observation must transform the image and its fractional occupancy target together, and (2) the profile sampling law should be designed under robust moment constraints rather than chosen heuristically. EMA training, fixed appearance perturbations, batch construction, and post-processing remain reproducibility details rather than named contributions.

The main ablation is reduced from a seven-row implementation ladder to four compute-matched rows. A0 is the 36-view teacher--student control, A1 changes only the through-plane image, A2 pairs the same profile with a fractional target under the uniform parent law, and A3 adds the moment-constrained distribution. The three adjacent contrasts directly test observation mismatch, paired-target alignment, and distribution design. Hard-target and support sensitivity are supplementary controls.

All implementation filenames, development checkpoint narratives, and internal candidate names were removed from the rendered manuscript. The method figure was relabeled accordingly and now represents MPD through the profile law `w ~ q*` rather than as a detachable block.
