# H1 Protocol: ViSA-MT Hard-but-Valid View Selection

## Status

CONFIRMATORY protocol draft. Lock this protocol in version control before executing.

## Hypothesis

Selecting the most student-challenging augmentation among teacher-stable candidates will outperform random strong augmentation and difficulty-only selection.

## Minimal Method

Let `f_t` be the EMA teacher, `f_s` the student, `w` a weak augmentation, and `{a_k}` a small candidate set.

1. Compute base teacher probability `p0 = f_t(w(x))`.
2. Sample `K=3` candidate strong views. Start with low-cost intensity/physics transformations; use the same weak geometry for all views.
3. For candidate `k`, compute aligned teacher probability `pt_k = align_k(f_t(a_k(w(x))))` and aligned student probability `ps_k = align_k(f_s(a_k(w(x))))`.
4. Validity cost: `V_k = JS(stopgrad(p0), stopgrad(pt_k))`, optionally averaged only over pixels where `max(p0) >= tau_c`.
5. Difficulty benefit: `D_k = JS(stopgrad(p0), ps_k)`.
6. Select `k* = argmax D_k` among candidates with `V_k <= tau_v`. If none are valid, fall back to the weakest candidate.
7. Train with supervised Dice+CE and an unsupervised Dice/CE or KL consistency loss on accepted pixels.

Keep the first implementation sample-wise. Do not add topology modules, contrastive heads, Copy-Paste, or pseudo-label refinement until the core signal is validated.

## Augmentation Bank for the Pilot

- Generic: gamma, brightness/contrast, Gaussian blur, Gaussian noise, cutout.
- MRI-oriented: smooth multiplicative bias field, Rician noise approximation, anisotropic downsample-upsample blur, intensity nonlinearity, mild ghosting/k-space line dropout if implemented correctly.
- Geometry in phase two only: flip, small affine, small elastic deformation, with exact inverse/forward label-coordinate mapping.

## Go/No-Go Experiment H1.1

Treat held-out labeled volumes as unlabeled. For each volume and augmentation candidate:

- compute `V_k` without using the label;
- compute oracle harm `E_k = loss(pt_k, y) - loss(p0, y)` after coordinate alignment;
- report patient-level Spearman correlation between `V_k` and `E_k`, AUROC for detecting harmful augmentations (`E_k > 0` or a clinically meaningful threshold), and calibration by operation/magnitude.

Go if the pooled Spearman correlation is positive and stable across folds, and the selector has useful AUROC. A practical initial target is median patient-level rho >= 0.4 and AUROC >= 0.7. These are engineering gates, not claimed universal thresholds.

## Baselines and Ablations

1. Supervised only.
2. Plain Mean Teacher with weak/weak views.
3. Tuned fixed weak-to-strong Mean Teacher.
4. UniMatch-style dual strong views.
5. Difficulty-only selection: maximize `D_k`.
6. Validity-only selection: minimize `V_k`.
7. ViSA-MT: maximize `D_k` subject to `V_k <= tau_v`.
8. ViSA-MT without base confidence gating.
9. Generic bank versus MRI-oriented bank.
10. One, two, three, and five candidates to measure the accuracy/compute tradeoff.

## Primary Outcomes

- Patient-level Dice and HD95.
- Secondary: ASD, relative volume difference, ECE/Brier score, boundary Dice, connected-component failures.
- Mechanistic: validity-error correlation, selected augmentation strength, acceptance rate, consistency-loss variance, pseudo-label precision/recall on oracle-held-out data.
- Efficiency: training FLOPs, wall-clock time, GPU memory, inference cost (expected unchanged).

## Failure Interpretation

- If `V_k` does not correlate with oracle harm, reject H1 before running a large training sweep.
- If difficulty-only equals ViSA-MT, validity gating is unnecessary.
- If fixed strong augmentation equals ViSA-MT within confidence intervals, the paper should pivot to the negative finding or to the center/style hypothesis.
- If gains appear only on PROMISE12, frame as prostate MRI specialization and target MICCAI/MedIA rather than CVPR.
