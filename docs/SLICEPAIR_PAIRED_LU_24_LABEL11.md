# SlicePair paired-L/U 24-view experiment with 11 labels

This is the label-budget counterpart of the 7-label `paired_lu_24` run. It
does not introduce another method or change the optimization recipe.

## Locked comparison

| Item | Label 7 | Label 11 |
|---|---:|---:|
| Labeled training cases | 7 | 11 |
| Labeled slice prefix | 191 | 306 |
| Student views after warm-up | 6 native-L + 6 paired-L + 12 paired-U | same |
| Loader batch | 12 L + 12 U | same |
| Pretrain / self-train | 10,000 / 30,000 | same |
| Warm-up / seed | 1,000 / 1337 | same |
| Split / selector | 35/5/10 / validation-best | same |

The label-11 run requires a label-11 Pre10000 checkpoint containing both
`net` and `opt`. A label-7 checkpoint is rejected.

## Linux run

```bash
cd ~/Documents/CODA_Final
conda activate my

DATA_ROOT="$HOME/Documents/Updated_code/data/PROMISE12_h5" \
PRETRAINED_CHECKPOINT="$HOME/Documents/CoDA-MT-PROMISE12/model/SliceEqOcc_Pre10000_label11_seed1337_11_labeled/pre_train/unet/unet_best_model.pth" \
GPU=0 DETACH=1 \
bash run_slicepair_paired_lu_24_label11_pipeline.sh
```

If the existing label-11 checkpoint is stored elsewhere, change only
`PRETRAINED_CHECKPOINT`. Its path must include `label11`, `11_labeled`, or
`11label`, and the file itself must contain the supervised `net+opt` state.

## Progress and outputs

```bash
tail -f "$(ls -t server_logs/slicepair_paired_lu_24_label11_*.log | head -n 1)"
```

```text
model/SlicePairPairedLU24_35_5_10_Pre10000_Self30000_label11_seed1337_11_labeled/self_train/unet/unet_best_model.pth
model/SlicePairPairedLU24_35_5_10_Pre10000_Self30000_label11_seed1337_11_labeled/self_train/unet/performance.txt
```

The pipeline evaluates only `unet_best_model.pth`. Do not choose a checkpoint
using the ten test cases.
