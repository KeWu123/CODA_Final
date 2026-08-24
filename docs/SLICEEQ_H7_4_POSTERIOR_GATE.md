# SliceEq H7.4 posterior-commutation gate

This is a frozen-model, zero-training analysis. Run it from the server `code`
directory after syncing the new files:

```bash
python -m unittest discover -s ../tests -p 'test_sliceeq_posterior.py' -v
python analyze_sliceeq_posterior_gate.py
```

It requires the existing H7.3 report at:

```text
../model/SliceEqOcc_PROMISE12_7_labeled/analysis/h7_3_gates_iter23000.json
```

and the same retained checkpoint:

```text
/home/aiteam/zhengtaoma/CoDA/model/SliceEqOcc_PROMISE12_7_labeled/self_train/unet/iter_23000_dice_0.8152.pth
```

The analysis first reproduces the H7.3 hard-LCC support statistics to an
absolute tolerance of `1e-6`. A mismatch in data order, profile draws,
checkpoint hash, or target construction stops the run before an H7.4 decision.

It then compares:

1. hard-LCC occupancy;
2. raw posterior commutation;
3. topology-gated posterior commutation.

Terminal progress looks like:

```text
[H7.4 labeled] batch 1/16 samples=12
```

The only output is:

```text
../model/SliceEqOcc_PROMISE12_7_labeled/analysis/h7_4_posterior_commutation_gate_iter23000.json
```

Interpret `decisions.joint_decision` as follows:

- `authorize_h7_4_training`: at least one preregistered candidate passed; use
  `selected_candidate` in an independent training version.
- `stop_posterior_commutation`: do not implement another SliceEq target/loss
  variant and broaden the research direction.

This command does not update model parameters or write a checkpoint.
