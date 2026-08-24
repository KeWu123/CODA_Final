# CODA Final：跨机器复现实验手册

## 1. 实验套件包含什么

一键脚本 `run_coda_final_paper_suite.sh` 按以下顺序执行。所有训练实验固定使用 PROMISE12 `35/5/10` 划分、7 个标注病例、同一个 Pre10000 checkpoint、Self30000、seed 1337 和验证集选择的 `unet_best_model.pth`。

| ID | 阶段 | 检查的问题 |
|---|---|---|
| B0 | MT-24 | 历史 24-view EMA scaffold 参照 |
| C0 | ViewMatch-36 | 仅增加一个普通标注视图，控制 batch、计算量和 BN 组成 |
| C1 | SRA-Image-36 | 混合三层图像但仍使用中心层 target，检查单纯图像增强 |
| C2 | SRA-Hard-36 | 图像和 target 同步变换，但 occupancy 立即硬化 |
| F10 | AFO-L-only | 只在标注分支保留 fractional occupancy |
| F01 | AFO-U-only | 只在未标注分支保留 fractional occupancy |
| C3 | SliceEqOcc | 两个分支均使用配对 fractional occupancy |
| C4 | SliceEqOcc + OAAC-S1.25 | 在 C3 上加入坐标保持的强度外观扰动 |
| C5 | SliceEqOcc + OAAC-S1.25 + MPD | 最终方法：训练集约束优化得到全局 profile 分布 |

脚本还会运行：

- 两张三切片重采集示意图；
- MPD 网格、分箱、容差、约束和 LOPO 离线审计；
- 方法合同与数值单元测试；
- 每个阶段的严格 test；
- 最终 CSV 和 Markdown 大表。

## 2. 另一台 Linux 电脑准备

```bash
cd ~/Documents
git clone https://github.com/KeWu123/CODA_Final.git
cd CODA_Final
conda activate my
pip install -r requirements.txt
```

数据和模型权重不上传 GitHub。假设它们仍在：

```text
~/Documents/Updated_code/data/PROMISE12_h5
~/Documents/Updated_code/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

## 3. 一行启动完整实验

```bash
cd ~/Documents/CODA_Final && conda activate my && DATA_ROOT="$HOME/Documents/Updated_code/data/PROMISE12_h5" PRETRAINED_CHECKPOINT="$HOME/Documents/Updated_code/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth" GPU=0 DETACH=1 bash run_coda_final_paper_suite.sh
```

它会在后台连续执行。查看最新进度：

```bash
cd ~/Documents/CODA_Final
tail -f "$(ls -t server_logs/coda_final_paper_suite_*.log | head -n 1)"
```

检查进程和 GPU：

```bash
pgrep -af 'train_sliceeq|run_coda_final_paper_suite'
nvidia-smi
```

## 4. 中断后继续

默认 `SKIP_COMPLETED=1`：存在非空 `performance.txt` 的阶段会跳过；存在 validation-best checkpoint 但尚未测试的阶段只补 test。

直接重新运行第 3 节命令即可。若某一阶段只留下不完整目录，先把该实验目录移动到备份位置，再重新启动。不要把 `ALLOW_EXISTING=1` 当成精确断点恢复，当前训练器不保存完整 RNG/DataLoader resume 状态。

## 5. 只跑部分阶段

例如只补主消融和最终 MPD：

```bash
DATA_ROOT="$HOME/Documents/Updated_code/data/PROMISE12_h5" PRETRAINED_CHECKPOINT="$HOME/Documents/Updated_code/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth" STAGES="baseline_36 image_only_36 hard_targets full mpd" GPU=0 DETACH=1 bash run_coda_final_paper_suite.sh
```

合法阶段名为：

```text
baseline baseline_36 image_only_36 hard_targets
occ_l_only occ_u_only full oaac_strong mpd
```

已有 C3/C4 结果、不想重复训练时，可以让默认跳过机制复用同名目录中的 `performance.txt`。

## 6. 只生成两张演示图

```bash
python code/visualize_sliceeq_reacquisition.py \
  --root_path "$HOME/Documents/Updated_code/data/PROMISE12_h5" \
  --output_dir "$PWD/paper_figures/sliceeq_demo"
```

输出：

- `sliceeq_image_reacquisition.png/pdf`：三张 MRI 加权合成虚拟切片；
- `sliceeq_paired_occupancy.png/pdf`：相同权重合成 fractional occupancy；
- `sliceeq_demo_manifest.json`：所选 train-only 样本及权重记录。

脚本默认在前 191 张标注训练切片中选择相邻层标签变化最明显的样本，便于展示 partial-volume 现象。也可以用 `--center_slice CaseXX_slice_N` 手动指定标注训练切片。

## 7. 只做 MPD 离线审计

```bash
python code/analyze_sliceeq_mpd_robustness.py \
  --root_path "$HOME/Documents/Updated_code/data/PROMISE12_h5" \
  --output_dir "$PWD/mpd_offline_audit"
```

完整说明见 `docs/SLICEEQ_MPD_OFFLINE_AUDIT_README.md`。该过程不加载模型、不访问 val/test，也不产生 Dice。

## 8. 输出位置

```text
model/<experiment>_7_labeled/self_train/unet/   checkpoints + performance.txt
server_logs/                                    pipeline/test logs
paper_results/coda_final_paper_results.csv      统一数值表
paper_results/coda_final_paper_results.md       可直接阅读的大表
paper_figures/sliceeq_demo/                     两张演示图
mpd_offline_audit/audit_summary.md              MPD 审计摘要
```

数据、权重、日志、结果和数据派生示意图均由 `.gitignore` 排除，避免把大文件或患者图像推到公开仓库。
