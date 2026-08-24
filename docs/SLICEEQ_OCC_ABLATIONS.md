# SliceEqOcc incremental ablations

> 本文件保留早期 M0--M3 增量链。论文主表应使用 compute/view-matched 的 C0--C4
> 设计与 AFO 2x2 因子消融，见
> [`SLICEEQ_OCC_PAPER_ABLATION.md`](SLICEEQ_OCC_PAPER_ABLATION.md)。

All runs keep the locked 7-label PROMISE12 protocol: 35/5/10 split, the first
7 labeled cases (191 slices), seed 1337, loader batch 24/12, Self30000, and the
same label-7 Pre10000 `net+opt` checkpoint.

## Baseline identity

M0 is the inherited **EMA hard-pseudo-label self-training baseline**. It is an
existing training framework, not a contribution of SliceEqOcc and not
PosteriorOcc. It uses:

- a 2-D U-Net Student and EMA Teacher;
- original center slices for both Student and Teacher;
- hard Teacher masks produced by argmax plus 2-D largest-component cleanup;
- hard CE+Dice for labeled GT and unlabeled pseudo-labels;
- the inherited 1k unlabeled-loss delay, consistency ramp, and EMA update.

SliceEqOcc contributes paired through-plane image/occupancy re-acquisition,
acquisition-derived fractional targets, and the exact-GT re-acquired labeled
anchor. PosteriorOcc is a later full-method variant that preserves Teacher
posteriors before profile integration; it is not a baseline or an ablation row.

## Incremental chain

Each main row adds exactly one mechanism to the preceding row after the common
1k identity warmup.

| Stage | `OCC_ABLATION` | Re-acquired U image | Neighbor-aligned fractional U target | Re-acquired labeled GT anchor |
|---|---|---:|---:|---:|
| M0 | `baseline` | no | no; center hard Teacher mask | no |
| M1 | `image_only` | yes | no; center hard Teacher mask | no |
| M2 | `aligned_occ` | yes | yes | no |
| M3 | `full` | yes | yes | yes |

M0 and M1 run the EMA Teacher on center slices only, so M1 changes only the
unlabeled Student image. M2 additionally evaluates real neighboring slices to
construct the aligned fractional occupancy. M3 adds the exact-GT labeled view.

`hard_targets` is an additional mechanism control, not a cumulative stage. It
uses the full M3 view layout but hardens the aligned occupancies to one-hot
targets. Comparing it directly with `full` isolates fractional versus hard
target supervision under otherwise matched conditions.

`no_labeled_reacq` remains accepted as a backward-compatible alias for
`aligned_occ`.

New runs use the `SliceEqOccIncremental_*` experiment prefix. This avoids
mixing the corrected definitions with old `SliceEqOccAblation_*` checkpoints.
In particular, the legacy `image_only` included a labeled re-acquired view and
is not a clean M1 result.

## Train one stage

```bash
cd ~/Documents/CoDA-MT-PROMISE12
conda activate my

PRE7="$HOME/Documents/Updated_code/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth"

DATA_ROOT="$HOME/Documents/Updated_code/data/PROMISE12_h5" \
PRETRAINED_CHECKPOINT="$PRE7" \
OCC_ABLATION="baseline" \
GPU=0 DETACH=1 \
bash run_sliceeq_occ_ablation.sh
```

Run the main chain one experiment at a time in this order: `baseline`,
`image_only`, `aligned_occ`, `full`. Run `hard_targets` afterward as the
mechanism control. Every row starts from the same PRE7 checkpoint; M0 does not
rerun supervised pretraining.

## Test one stage

```bash
cd ~/Documents/CoDA-MT-PROMISE12/code
conda activate my

STAGE="baseline"
EXP="SliceEqOccIncremental_${STAGE}_35_5_10_Pre10000_Self30000_label7_seed1337"

CUDA_VISIBLE_DEVICES=0 python test_sliceeq_occ.py \
  --root_path "$HOME/Documents/Updated_code/data/PROMISE12_h5" \
  --exp "$EXP" \
  --checkpoint_path "../model/${EXP}_7_labeled/self_train/unet/unet_best_model.pth" \
  --labelnum 7 --gpu 0 --save_result False --nms 0 \
  --auto_find_checkpoint False
```

Report M0--M3 in one table. Compare `hard_targets` only against M3/full; do not
present it as another step in the cumulative chain.

## Run M0 and M1 automatically

This is the primary ablation workflow currently needing a new run. It trains
M0 and tests its validation-selected checkpoint before starting M1, then prints
both `performance.txt` files.

```bash
cd ~/Documents/CoDA-MT-PROMISE12
conda activate my

DATA_ROOT="$HOME/Documents/Updated_code/data/PROMISE12_h5" PRETRAINED_CHECKPOINT="$HOME/Documents/Updated_code/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth" GPU=0 DETACH=1 bash run_sliceeq_occ_m0_m1_pipeline.sh
```

Follow it with:

```bash
tail -f server_logs/sliceeq_occ_m0_m1_pipeline_*.log
```

## Run M2 and M3 automatically

The existing pipeline runs only M2 and M3 in strict sequence. Each stage is
tested immediately after training; M3 starts only if M2 training and testing
both succeed.

```bash
cd ~/Documents/CoDA-MT-PROMISE12
conda activate my

PRE7="$HOME/Documents/Updated_code/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth"

DATA_ROOT="$HOME/Documents/Updated_code/data/PROMISE12_h5" \
PRETRAINED_CHECKPOINT="$PRE7" GPU=0 DETACH=1 \
bash run_sliceeq_occ_m2_m3_pipeline.sh
```

Follow the pipeline with:

```bash
tail -f server_logs/sliceeq_occ_m2_m3_pipeline_*.log
```

By default the pipeline refuses to reuse an existing M2 or M3 experiment
directory. Set `ALLOW_EXISTING=1` only when reusing that directory is
intentional. Set `DETACH=0` to keep the complete pipeline in the foreground.
