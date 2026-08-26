# SlicePair paired-L/U 24-view experiment

This experiment is the paper A3 view-budget control. The loader remains fixed
at 12 labeled and 12 unlabeled center slices. After the inherited 1,000-step
identity warm-up, every student step contains exactly 24 views:

```text
6 native labeled + 6 paired re-acquired labeled + 12 paired re-acquired unlabeled
```

The 12 labeled centers are randomly partitioned into two equal sets every
step. Every center is used exactly once: six receive native hard supervision,
and the complementary six receive exact-GT fractional supervision. Their
branch losses retain the full objective's equal weighting:

```text
L_sup = 0.5 * (L_native_hard + L_paired_fractional)
L = L_sup + lambda(t) * L_unlabeled_fractional
```

The partition uses a dedicated CPU RNG seeded with `seed + 3`. It does not
consume the unlabeled profile, labeled profile, data-loader, or appearance RNG
streams. The unlabeled branch and OAAC-Strong policy are unchanged from the
uniform-profile 36-view parent.

## Linux run

```bash
cd ~/Documents/CODA_Final
conda activate my

DATA_ROOT="$HOME/Documents/Updated_code/data/PROMISE12_h5" \
PRETRAINED_CHECKPOINT="$HOME/Documents/Updated_code/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth" \
GPU=0 DETACH=1 \
bash run_slicepair_paired_lu_24_pipeline.sh
```

The pipeline trains for 30,000 self-training iterations and then evaluates
only `unet_best_model.pth` selected on the five validation cases.

## Progress

```bash
cd ~/Documents/CODA_Final
tail -f "$(ls -t server_logs/sliceeq_occ_slicepair_paired_lu_24_pipeline_*.log | head -n 1)"
```

## Output

```text
model/SliceEqOccIncremental_paired_lu_24_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/self_train/unet/unet_best_model.pth
model/SliceEqOccIncremental_paired_lu_24_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/self_train/unet/performance.txt
```

Do not select another iteration from the test set. If the experiment directory
already exists, move it aside; use `ALLOW_EXISTING=1` only when intentionally
retesting an existing validation-selected checkpoint.
