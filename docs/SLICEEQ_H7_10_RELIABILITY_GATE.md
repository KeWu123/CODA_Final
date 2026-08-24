# SliceEq H7.10: operator-reliability gate

## Outcome of this implementation

This is the next method experiment after the negative APTNA run. It is a
read-only, zero-training gate for two bounded SliceEqOcc successors. It does
not change `train_sliceeq_occ.py`, validation, checkpoint selection, test
inference, or any case-specific behavior.

The scientific question is whether the current acquisition occupancy mixes
useful anatomical partial volume with avoidable teacher-dropout noise. Exact
labels are used only from the locked seven-patient labeled training prefix to
decide whether either mechanism deserves one full run.

| Candidate | Potential training-time change | Extra teacher forwards | New parameters | Inference change | Current evidence |
|---|---|---:|---:|---:|---|
| SCT: Stack-Coherent Teacher Stochasticity | Share each teacher dropout mask across the three slices supporting one re-acquisition | 0 | 0 | None | Exploratory negative; branch closed |
| ADU: Acquisition-Aligned Dropout Uncertainty | Compare two hard-LCC occupancies after the same acquisition operator and use their JS disagreement as reliability | 1 | 0 | None | Exploratory positive; one full run authorized |

SCT aligns teacher stochastic correlation with the support of the existing
modeled acquisition operator. It is not 3-D context, topology post-processing, or a new
profile sampler. ADU does not filter high-entropy fractional occupancy: two
passes that agree on a fractional target have zero disagreement and retain
full weight.

The hypothesis concerns the EMA teacher, but the historical SliceEqOcc files
contain only student state dictionaries. The analyzer therefore loads the
18k, 24k, and 30k students as train-mode proxy teachers. A positive gate only
authorizes one full experiment; it is not a claimed Dice improvement and is
not a reconstruction of the unavailable historical EMA states.

The uploaded numerical gate predates the final analyzer hardening for
deterministic CuDNN, support-pixel pooling, checkpoint-name locking, and
cross-checkpoint quality accounting. Its raw ADU pair records support a
conservative machine-readable reaggregation, but the original execution
cannot be relabeled as a final-hash confirmatory run. The current status is
therefore an exploratory mechanism result that authorizes one exploratory
full run. A paper-level confirmatory claim requires rerunning the final frozen
analyzer.

## Locked remote command

Run from the repository `code` directory on the CUDA machine:

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python analyze_sliceeq_reliability_gate.py \
  --root_path /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source \
  --checkpoint_dir ../model/SliceEqOcc_PROMISE12_7_labeled/self_train/unet \
  --checkpoint_steps 18000 24000 30000 \
  --output_json ../model/SliceEqOcc_PROMISE12_7_labeled/analysis/h7_10_operator_reliability_gate.json
```

The default checkpoint names are `iter_18000.pth`, `iter_24000.pth`, and
`iter_30000.pth`. Validation-save names such as
`iter_18000_dice_0.8000.pth` are also accepted. If such retained files must be
passed explicitly, preserve their locked iteration mapping:

```bash
python analyze_sliceeq_reliability_gate.py \
  --checkpoints /path/to/iter_18000_dice_x.pth /path/to/iter_24000_dice_x.pth /path/to/iter_30000_dice_x.pth \
  --checkpoint_steps 18000 24000 30000 \
  --output_json /path/to/h7_10_operator_reliability_gate.json
```

Do not substitute checkpoints after inspecting the gate output. The analyzer
requires exactly 191 labeled slices, seven patients, two fixed case-mixed
batch schedules, eight stochastic draws, and the three locked checkpoints.

## Data and state isolation

- Only `train.list`, `train_slices.list`, and the first 191 labeled-slice H5
  files are accessible. The dataset subclass rejects any label read outside
  that allowlist.
- Official validation and test data are not loaded.
- Every paired forward restores the same model buffers and runs inside a
  forked RNG scope. Parameters are frozen and hashed before and after.
- SCT preserves the teacher's train mode, input batch composition, and BN
  update rule. Its altered upstream dropout can still change the numerical
  batch statistics, so the analysis does not claim bit-identical BN values.
- Padding of the final 11-sample batch is used only to retain the normal
  12-stack train-mode BN shape and is excluded from all statistics.
- The only output is one atomically written JSON report; there is no optimizer,
  backward pass, or checkpoint write.

## Automatic decision

The JSON field `decisions.joint_decision` has one of three meanings:

- `authorize_slice_eq_occ_sct_training`: implement and run only SCT.
- `authorize_slice_eq_occ_adu_training`: implement and run only ADU.
- `stop_h7_10_small_method_extensions`: neither mechanism has sufficient
  exact-GT evidence; do not rescue it with new thresholds.

Both candidates must pass patient-balanced conditions at two of three
checkpoints and on at least five named patients after cross-checkpoint median
aggregation. SCT must improve non-clamped stacks as well. Every one of the four
ADU pairs must be finite and non-degenerate. If both pass, SCT remains the
default because it changes less; the predeclared two-percentage-point ADU
selection margin is an engineering selection rule, not a novelty claim.

Any authorized full training entry will retain the original SliceEqOcc
validation block verbatim, as requested.

## Publication boundary

These candidates are secondary components under SliceEqOcc's main paired
re-acquisition and fractional-occupancy contribution. Do not claim first
shared dropout, first MC-dropout uncertainty, first JS weighting, or a strict
aleatoric/epistemic decomposition. A safe description is that the method
measures dropout-induced disagreement after the same non-invertible modeled
acquisition operator. This provides an operational reliability proxy that
leaves two-pass-agreed mixture at full weight; it is not an identifiable
decomposition of uncertainty.
