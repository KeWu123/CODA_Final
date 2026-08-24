# H7.6a Scan-Coherent SliceEqOcc Implementation Protocol

Status: negative and closed on 2026-08-13; test Dice `0.836219`.  
Date: 2026-08-13.  
Evidence class of the first full run: exploratory optimization run.  
Parent: the user-confirmed SliceEqOcc recipe (`0.844566` development Dice).

## User priority and scope

The user explicitly prioritizes a direct optimization of SliceEqOcc before the causal ablation matrix. This authorizes one independent successor while preserving the current method and fixed paths. It does not authorize generic module stacking or post-hoc test selection.

The initially proposed H7.6 combined protocol conditioning and scan coherence. The current PROMISE12 H5/list contract exposes only images, labels, and names of the form `CaseXX_slice_i`; it contains no verified spacing, thickness, gap, vendor, or acquisition-protocol field. Therefore this implementation is deliberately restricted to **synthetic scan coherence**. It must not be described as metadata-conditioned, thickness-calibrated, or an exact scanner PSF.

## Hypothesis

SliceEqOcc currently samples a different virtual slice profile independently for every center slice. A real acquisition protocol is scan-level, so adjacent samples from the same patient can be assigned contradictory virtual acquisition profiles during one pass through the volume. Sharing one virtual profile within a case for an epoch, while preserving continuous cross-case and cross-epoch coverage, will reduce acquisition-domain label noise and improve the SliceEqOcc development result.

## Sole method change

Replace the two independent per-sample profile RNG streams with two branch-specific scan-protocol tables:

1. collect labeled and unlabeled case IDs from the unchanged ordered `train_slices.list` and locked labeled boundary;
2. at each protocol refresh, assign every case one `(sigma, phase)` pair;
3. every slice from the same case uses exactly that pair until the next refresh;
4. labeled and unlabeled cases are stratified independently so each branch retains its uniform marginal;
5. refresh the tables once per training epoch by default.

For `N` cases in a branch, each marginal uses randomized stratification:

\[
u_i = (\pi(i) + \epsilon_i)/N,\quad \epsilon_i\sim U(0,1),
\]

with independent seeded permutations for `sigma` and `phase`. The values are mapped to the unchanged ranges:

\[
\sigma\in[0.45,0.85],\qquad \phi\in[-0.25,0.25].
\]

This preserves continuous tails and uniform long-run marginals. It is not the four-node SAQ rule.

## Frozen parent recipe

- root path: `/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source`;
- pretrained checkpoint: the existing fixed UniMatch Pre10000 net+optimizer path;
- 30k self-training updates, seed 1337, loader batch 24 and labeled batch 12;
- first 1k identity warmup;
- effective student batch 36 after warmup: 12 original-L + 12 reacquired-L + 12 reacquired-U;
- original hard labeled anchor and exact-GT labeled fractional occupancy;
- detached EMA argmax + 2D LCC unlabeled occupancy;
- soft CE + squared soft Dice, inherited consistency ramp, SGD and EMA;
- three real neighboring slices, same spatial transform and endpoint clamping;
- unchanged 2D validation/inference graph and strict checkpoint loading.

No confidence threshold, posterior target, boundary weight, attention module, new head, loss, schedule, batch, teacher policy, or inference input is added.

## Parameters

- `--sliceeq_protocol_refresh_epochs 1`;
- `--sliceeq_sigma_min 0.45`;
- `--sliceeq_sigma_max 0.85`;
- `--sliceeq_phase_min -0.25`;
- `--sliceeq_phase_max 0.25`.

The refresh parameter must be a positive integer. The first run uses the default and does not tune it.

## Reproducibility and diagnostics

- Use stable case strings, sorted case tables, explicit CPU `torch.Generator` seeds, and no Python `hash()`.
- L/U tables have independent deterministic seed streams.
- Record refresh ID, branch case counts, batch unique-case fractions, and maximum within-case sigma/phase range.
- Within-case ranges must be exactly zero up to floating precision.
- Record branch sigma/phase/center-weight summaries and all inherited occupancy diagnostics.
- The new experiment identity is `SliceEqOccSC_PROMISE12`; no parent source, checkpoint, or output is overwritten.

## Prediction and decision

Primary metric: the same locked Dice metric used for the user-confirmed SliceEqOcc result. Secondary diagnostics are validation trajectory stability and inherited occupancy activity.

- Positive: a repeatable improvement over `0.844566` under the same externally used result-selection convention warrants retaining scan coherence as the next method version, followed by confirmatory seeds/controls.
- Neutral: no material gain closes scan coherence; do not tune refresh duration or add metadata-free protocol heuristics.
- Negative: return to SliceEqOcc and investigate same-anatomy conditional two-profile integration rather than batch/case marginal sampling.

The result must be reported as exploratory because the user requested optimization before ablation and the current PROMISE12 test has served as development feedback.

Observed outcome: best validation Dice `0.816251`, test Dice `0.836219`, and
9/10 paired test cases below SliceEqOcc. The operator was active and the
within-case profile ranges were exactly zero. Therefore refresh-duration
tuning is prohibited by the predeclared negative rule. The next registered
experiment is same-anatomy conditional antithetic integration in
`h7_7_conditional_antithetic_protocol.md`.

## Batch-36 interpretation

The added 12 views are required by the **current SliceEqOcc three-branch objective** because it simultaneously preserves the 12 original hard labeled anchors and adds 12 exact-GT re-acquired labeled occupancy views. They are not mathematically required by fractional occupancy itself. A 24-view variant could replace the original labeled view, omit labeled re-acquisition, or split the design, but it would no longer be the confirmed SliceEqOcc recipe. H7.6a therefore freezes 36 to make scan coherence the only change.
