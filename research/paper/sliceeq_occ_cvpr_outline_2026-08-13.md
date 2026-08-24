# SliceEq CVPR paper blueprint

Status: writing scaffold based on the selected method and accepted development results. Statements requiring future multi-seed/external experiments are marked as pending and must not be converted to factual claims before evidence exists.

## Working title

**SliceEq: Acquisition-Aligned Fractional Occupancy for Semi-Supervised MRI Segmentation**

If only prostate MRI is finally evaluated:

**SliceEq: Acquisition-Aligned Fractional Occupancy for Semi-Supervised Prostate MRI Segmentation**

## One-sentence thesis

For a non-invertible through-plane acquisition perturbation, semi-supervised consistency is correctly defined only when the image observation and its exact or pseudo supervision are re-acquired by the same forward operator.

## Abstract draft

Data augmentation is central to semi-supervised medical image segmentation, yet most methods implicitly assume that the segmentation target remains invariant after perturbing the image. This assumption is violated by through-plane MR acquisition: finite slice support and off-center sampling integrate tissues from neighboring anatomical planes, so a re-acquired image is no longer faithfully described by the hard mask of its center slice. We introduce **SliceEq**, a training-only paired re-acquisition augmentation that applies one stochastic slice-profile operator jointly to real neighboring MR slices and their tissue masks. The resulting target is an operator-induced fractional occupancy rather than heuristic label smoothing. We use exact masks for labeled volumes and EMA teacher masks for unlabeled volumes, while retaining the original hard center-slice supervision as an anatomical anchor. SliceEq requires neither a 3D backbone nor neighboring slices at test time and leaves the single-slice 2D inference graph unchanged. On the current PROMISE12 development protocol with seven labeled volumes, the configured hard-target predecessor and fractional-occupancy successor obtain approximately 83.26% and 84.46% Dice, respectively. Because this development step also adds a labeled re-acquisition view and changes the student-view composition, matched controls are required before attributing the entire difference to fractional occupancy. **[After confirmatory experiments: add one sentence reporting multi-seed gains over compute-matched image-only/hard-target controls and one sentence reporting external/protocol-shift robustness.]** These results motivate acquisition-aligned target semantics as an alternative to label-invariant strong augmentation for semi-supervised MRI segmentation.

Do not add `state-of-the-art` unless a protocol-matched, multi-seed table establishes it.

## Introduction argument

### Paragraph 1: problem and dominant paradigm

- Medical segmentation labels are expensive; SSMIS uses unlabeled data through teacher--student consistency.
- Contemporary methods improve the perturbation: Copy-Paste/mixing, weak-to-strong views, adaptive masking, displacement, frequency transforms.
- Shared assumption: perturb the observation while preserving or compositing a hard semantic target.

### Paragraph 2: hidden failure of label invariance

- This assumption is reasonable for many in-plane photometric/geometric transforms.
- It is not correct for a non-invertible through-plane MR acquisition.
- A thick/off-center virtual slice integrates adjacent anatomical planes; its target should represent tissue proportions under the same support.
- Using the center hard mask gives a structurally mismatched training pair precisely near apex/base and partial-volume boundaries.

### Paragraph 3: method intuition

- Access the real neighboring slices only during training.
- Sample one slice profile and apply identical normalized weights to image intensity and one-hot tissue masks.
- Preserve fractional occupancy for both exact labeled masks and teacher-derived unlabeled masks.
- Keep the original hard labeled center as an anchor.
- Test remains an ordinary 2D single-slice model.

### Paragraph 4: evidence and findings

- Locked development trajectory: the BCP-derived no-Copy-Paste baseline fluctuates at 0.78--0.80 Dice, hard-target SliceEq reaches about 0.832603, and full SliceEqOcc reaches about 0.844566.
- This gives two empirical transitions: approximately +3.26 to +5.26 Dice points from baseline to SliceEq, followed by +1.1963 points from hard SliceEq to fractional SliceEqOcc. Only matched controls may turn these transitions into causal component claims.
- Fractional residual is sparse but gradient-active, so it carries high-value acquisition semantics.
- SAQ does not improve performance because balancing profile nodes across different anatomy samples preserves mean severity but removes tails and does not integrate sample-conditional risk.
- **Pending:** summarize matched controls, multi-seed CI, external/protocol robustness, and efficiency.

### Paragraph 5: contributions

1. We identify a target-semantics failure in label-invariant SSL augmentation under a non-invertible through-plane forward operator and formulate acquisition-aligned consistency. This is not group equivariance.
2. We propose SliceEq, a paired slice-profile re-acquisition operator that produces image-aligned fractional occupancy from exact and EMA pseudo masks while retaining a hard anatomical anchor.
3. **Pending evidence:** We establish the mechanism through compute/BN-matched image-only and hard-target controls and demonstrate cross-protocol robustness with unchanged 2D inference cost.

## Related work structure

### Semi-supervised medical image segmentation

Discuss Mean Teacher, BCP, ICT, FixMatch/UniMatch, AugSeg, ABD, beta-FFT, and uncertainty/ambiguity methods. ICT is the closest interpolation-consistency prior and must be conceded explicitly: it already interpolates unlabeled inputs and teacher predictions. Focus the distinction on same-subject anatomical neighbors, the constrained through-plane operator, operator-derived spatial occupancy, and unchanged single-slice inference. Explicitly state that the local baseline is a BCP-derived EMA scaffold because Copy-Paste is removed; original BCP remains a comparator.

### Acquisition-aware augmentation and partial volume

Discuss PV-SynthSeg, SynthSeg, MR slice-profile estimation, resolution/thickness simulation, and Inter-Slice Augmentation (ECAI 2020). Concede that neighboring medical images and their labels have already been interpolated. Establish the narrower gap at the intersection of a constrained re-acquisition operator, exact/pseudo fractional supervision, and training-only neighboring slices in 2D SSL.

### Soft and ambiguous targets

Separate three phenomena:

- heuristic label smoothing;
- annotation/model ambiguity;
- deterministic acquisition-induced fractional occupancy.

SliceEq belongs to the third.

## Method section

### 3.1 Problem setting

Define labeled volumes `D_L`, unlabeled volumes `D_U`, student `f_theta`, and EMA teacher `f_xi`. Clarify that training samples are 2D centers with access to a three-slice stack, while testing uses only the center image.

### 3.2 Slice-profile re-acquisition

Define `h=(sigma, phi)`, normalized weights `w_k(h)`, image operator `A_h(X)`, and tissue occupancy operator `A_h(onehot(Y))`. State the discrete three-tap approximation and endpoint policy. Avoid claiming it is the exact scanner PSF unless H7.6 obtains valid metadata.

### 3.3 Fractional occupancy supervision

Explain why `argmax(A_h(Y))` returns to a nearly invariant hard target and loses partial-volume semantics. Define soft cross-entropy and squared soft Dice. Emphasize that the target distribution is spatial and operator-derived, not uniform label smoothing.

### 3.4 Semi-supervised objective

Define:

- original labeled hard-anchor loss;
- re-acquired labeled exact-occupancy loss;
- re-acquired unlabeled teacher-pseudo-occupancy loss;
- EMA update and consistency ramp.

State that the teacher target is detached. Clearly disclose LCC if retained.

### 3.5 Training and inference complexity

Training after warmup uses 36 student views: 12 original L, 12 re-acquired L, and 12 re-acquired U. Inference uses one 2D image and the unchanged U-Net. Report measured train FLOPs, peak memory, and test latency, not only asymptotic claims.

### Conditional 3.5a: ordered appearance weak-to-strong extension

H7.13 passes its validation proxy (`0.834863` versus threshold `0.820373`), while a post-hoc test-selected 27k checkpoint reaches development Dice `0.849538` after multiple checkpoints were inspected on the local test split. This number is oracle-only and cannot enter the primary table; testing the validation-selected 23.8k checkpoint later does not restore test independence. Include this subsection in the final method only if a frozen validation selector is confirmed on fresh hidden/external evaluation and matched multi-seed controls retain the gain. Define the order
`A_h(image, occupancy)` followed by `G_eta(image)` only: acquisition changes the target and is paired first; the monotonic coordinate-preserving appearance transform leaves the already formed occupancy unchanged. Present OAAC as broader student-view coverage under SliceEq, not as a new weak-to-strong or photometric-augmentation contribution. If H7.13 is neutral, omit it from the method and list it only among appendix design choices.

H7.15 supplies a cleaner development checkpoint-selection result: jointly scaling OAAC severity by 1.25 raises validation from `0.834863` to `0.836475`, and the validation-best checkpoint reaches Dice `0.851960`. Treat this as a positive single-seed development sensitivity result, not a primary result. H7.16 tests one final 1.50 outer bracket; the OAAC scale is frozen at 1.25 unless 1.50 exceeds validation `0.836475`. Any final paper claim still requires fresh test-independent evaluation and matched multi-seed controls.

H7.16 closes the OAAC severity search: scale1.50 reaches validation `0.835796` and validation-best development Dice `0.852059`, failing to replace scale1.25. Its +0.000099 test difference comes with only 2/10 wins and worse Jaccard/HD95/ASD. Report scale1.0/1.25/1.50 as a compact sensitivity table and use scale1.25 in the final method. Do not claim a global optimum; call it the validation-selected local setting.

## Figure plan

### Figure 1: the paper in one image

Three columns:

1. **Conventional strong consistency:** augmented image paired with unchanged center hard mask; highlight a boundary mismatch.
2. **Physical cause:** a through-plane profile spans `z-1,z,z+1`, integrating adjacent tissues.
3. **SliceEq:** identical weights produce the re-acquired image and fractional occupancy; student test path remains a single 2D slice.

The visual should foreground target semantics, not a generic teacher--student diagram.

### Figure 2: training pipeline

- L center hard anchor;
- L exact-mask paired re-acquisition;
- U EMA per-slice hard pseudo-slab paired re-acquisition;
- soft CE+Dice and EMA update;
- dashed training-only neighboring-slice path.

### Figure 3: mechanism visualization

- one apex/base stack;
- center hard mask, hard re-acquired argmax, and fractional occupancy heatmap;
- pixelwise target difference concentrated at acquisition-active boundary;
- optional gradient map showing sparse high-gradient support.

### Figure 4: protocol robustness

Pending H7.6/external evidence: Dice/NSD versus synthetic target thickness or spacing strata for B0-36, ImgOnly, SliceHard, and Full.

## Table plan

### Table 1: main comparison

Rows: supervised-only, BCP-derived scaffold, original BCP, UniMatch, recent public methods, SliceEq.
Columns: PROMISE12 at two label budgets, external MRI dataset, Dice, NSD, physical HD95. Report mean ± SD across seeds.

### Table 2: causal ablation

Rows: B0, B0-36, ImgOnly, SliceHard, Occ-L-only, Occ-U-only, Full.
Columns: effective batch, neighbor image, paired target, fractional L, fractional U, Dice, NSD, training FLOPs.

The central visual comparison is:

`SliceEq hard target 0.832603 -> SliceEqOcc fractional target 0.844566` (development values), followed by confirmatory multi-seed values.

The table caption or accompanying text should also give the full development trajectory:

`baseline 0.78--0.80 -> SliceEq 0.832603 -> SliceEqOcc 0.844566`.

### Table 3: acquisition robustness

Rows: B0-36, generic blur augmentation, ImgOnly, SliceHard, Full, optional H7.6.
Columns: native, mild, medium, severe virtual acquisition, robustness AUC, external center.

### Table 4: design choices

- hard versus fractional occupancy;
- L-only versus U-only versus both;
- profile range sensitivity;
- hard-anchor ratio;
- teacher policy if scientifically useful.

SAQ goes in the appendix as a negative sampling-strategy ablation.

## Required experimental wording

- Use `development result` for 0.844566 until a test-independent checkpoint or untouched evaluation confirms the primary number.
- Use `post-hoc test-selected development oracle` for OAAC 0.849538; do not call it validation-selected, primary, confirmatory, or an unbiased gain.
- Use `BCP-derived EMA pseudo-label scaffold` for the no-Copy-Paste code.
- Use `legacy voxel-index HD95/ASD` for existing distance metrics; do not attach `mm`.
- Use `acquisition-inspired discrete slice profile` for the current implementation.
- Use `spacing-conditioned` only when spacing is read correctly.
- Use `thickness-calibrated/native-to-target PSF` only with trustworthy thickness/profile provenance.

## Claim-evidence ledger

| Proposed claim | Required evidence | Current status |
|---|---|---|
| fractional occupancy improves hard SliceEq | locked 0.832603 to 0.844566 development comparison; matched multi-seed SliceHard | development positive; confirmatory pending |
| gain is not extra batch/GT view | B0-36 and L-only/U-only factorial | pending |
| gain is not neighbor smoothing | ImgOnly and generic blur controls | pending |
| paired operator matters | Full versus ImgOnly and TargetOnly | pending |
| no inference overhead | identical graph plus measured latency/FLOPs | graph verified; measurement pending |
| robust to acquisition shift | synthetic protocol curve and external/center-out test | pending |
| general medical segmentation method | second organ/modality MRI dataset | pending; otherwise narrow title |
| ordered acquisition then appearance improves SliceEq | locked validation proxy, frozen selector evaluated on a fresh hidden/external set, and matched multi-seed photometric controls | validation proxy positive; PROMISE12 oracle positive but test-selected; confirmatory pending |

## Appendix plan

- detailed pseudo-code and operator proof that weights form a valid occupancy distribution;
- profile parameter sensitivity;
- complete per-case/per-seed statistics;
- endpoint clamping analysis;
- SAQ negative result and tail-truncation explanation;
- SCPO negative result: short-slab 26-connected cleanup changes only 0.004527% of U pseudo pixels and does not improve the locked parent;
- rejected residual-weighting and posterior-commutation gates as research diagnostics, not contributions;
- environment, hashes, checkpoint rule, and unlabeled-label firewall test.

## Suggested final narrative

The paper should not read as “we found a better augmentation.” It should read as:

1. existing SSL perturbations assume target invariance;
2. an MR acquisition transform changes the semantics of the target;
3. SliceEq makes the target commute with the acquisition operator through fractional occupancy;
4. the configured hard-target predecessor reaches about 0.832, while the fractional-occupancy successor reaches about 0.844 in the current development trajectory; matched controls are still needed for a causal attribution;
5. the no-Copy-Paste EMA baseline is 0.78--0.80, giving a clear two-stage development trajectory;
6. matched controls and protocol-shift experiments demonstrate that the gain comes from acquisition-aligned semantics rather than smoothing or compute;
7. all volumetric information is training-only, so deployment remains the original 2D model.
