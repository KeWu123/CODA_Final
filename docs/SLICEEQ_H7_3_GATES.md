# SliceEq H7.3 three-gate analysis

Run this diagnostic from the `code` directory on the same CUDA server that
contains the retained SliceEqOcc iteration-23,000 checkpoint:

```bash
python analyze_sliceeq_gates.py
```

The defaults deliberately match the current PROMISE12 root, seed 1337,
three-slice profile distribution, first-191-slice labeled split, and checkpoint:

`/home/aiteam/zhengtaoma/CoDA/model/SliceEqOcc_PROMISE12_7_labeled/self_train/unet/iter_23000_dice_0.8152.pth`

The bounded default run evaluates all 191 labeled slices (16 batches, with a
short final batch) and the first 192 unlabeled slices (16 batches). It performs
no optimization and writes only:

`../model/SliceEqOcc_PROMISE12_7_labeled/analysis/h7_3_gates_iter23000.json`

The terminal prints progress separately for labeled and unlabeled batches and
then prints the three Boolean decisions plus `joint_decision`. A
`provisional_proceed` result means all preregistered thresholds passed. It is
called provisional because the saved checkpoint contains only the student;
Gate 2 necessarily uses that frozen student as an EMA-teacher proxy.

To inspect the final decision:

```bash
python - <<'PY'
import json
p = '../model/SliceEqOcc_PROMISE12_7_labeled/analysis/h7_3_gates_iter23000.json'
with open(p, encoding='utf-8') as f:
    report = json.load(f)
print(json.dumps(report['decisions'], indent=2))
PY
```

Do not change thresholds after seeing the report. Their locked definitions and
kill rules are in
`research/experiments/h7_slice_profile_reacquisition/h7_3_gate_protocol.md`.
