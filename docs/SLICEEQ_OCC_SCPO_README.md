# SliceEqOcc-SCPO

`SliceEqOcc-SCPO` is the one bounded successor authorized after ADU returned a
neutral result. It keeps the model, student batch36, loss, profile sampler,
train-mode EMA, validation, and 2-D inference unchanged. The sole method change
is that the hard U pseudo masks form one 26-connected component over each
three-slice slab before the existing SliceEq operator.

## CUDA tests

From the repository root:

```bash
python -m unittest tests.test_sliceeq_scpo tests.test_sliceeq_scpo_contract -v
```

The tensor tests must run with PyTorch and must not be skipped. Then train from
the `code` directory:

```bash
python train_sliceeq_occ_scpo.py
```

All recipe arguments and the shared Pre10000 checkpoint SHA-256 are locked by
the entry. Output is isolated under `SliceEqOccSCPO_PROMISE12`.

## Strict evaluation

```bash
python test_sliceeq_occ_scpo.py \
  --checkpoint_path ../model/SliceEqOccSCPO_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

Use only the validation-selected best checkpoint. The matched parent best
validation is `0.817373`; exploratory SCPO success requires `>=0.820373`.
Periodic checkpoint test inspection cannot select the method or iteration.

## Interpretation

SCPO is not a 3-D model and adds no inference operation. Its purpose is to make
the latent pseudo occupancy coherent before a through-plane acquisition is
simulated. The logs report how often SCPO actually changes the parent 2-D-LCC
target. If the run is negative, do not tune connectivity, morphology, slab
width, or component thresholds.
