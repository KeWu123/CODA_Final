# SliceEq MPD 离线鲁棒性审计

## 1. 这段代码检查什么

`code/analyze_sliceeq_mpd_robustness.py` 是独立于训练流程的离线审计工具。它检查 MPD 分布是否严重依赖人为离散或某一个直接指定的常数，主要包括：

1. `11x11 / 21x21 / 31x31` profile 网格收敛性；
2. `2 / 3 / 4` 个轴向位置分箱的敏感性；
3. moment budget、image residual budget 和 utility optimum fraction 的单因素敏感性；
4. 去除 entropy floor 或 density cap 后，分布及目标是否明显变化；
5. 七个标注病例的 leave-one-patient-out（LOPO）稳定性；
6. 每条约束的余量和活跃状态；
7. 可选地与已有 `mpd_profile_design.json` 逐项比较。

这不是新的训练方法，也不会产生新的 Dice。它的作用是回答审稿人很可能提出的问题：MPD 是否只是依靠 `21`、`3`、`2%`、`5%` 和 `99%` 这些人为参数才成立。

## 2. 严格的数据边界

审计器调用原 MPD 的锁定数据读取函数，只读取：

- `train_slices.list` 的前 191 张切片；
- 这 191 张切片所属的 7 个完整标注病例；
- 对应的 image 和 ground-truth label。

它不会读取未标注病例的 label，不会构建 val/test 数据集，也不会加载模型权重、预测、loss、confidence 或 Dice。因此结果可以作为 train-only 的方法设计审计，而不是验证集调参。

## 3. Linux 运行命令

在项目根目录执行：

```bash
cd ~/Documents/CoDA-MPD-Final
conda activate my

python code/analyze_sliceeq_mpd_robustness.py \
  --root_path "$HOME/Documents/Updated_code/data/PROMISE12_h5" \
  --output_dir "$PWD/mpd_offline_audit"
```

如果已有训练时冻结的 MPD artifact，可以额外检查默认审计器是否精确复现它：

```bash
python code/analyze_sliceeq_mpd_robustness.py \
  --root_path "$HOME/Documents/Updated_code/data/PROMISE12_h5" \
  --output_dir "$PWD/mpd_offline_audit" \
  --reference_artifact "/path/to/mpd_profile_design.json"
```

该脚本只使用 CPU。默认会重复求解多个凸优化问题，所需时间主要由 SciPy SLSQP 决定。

## 4. 快速检查与完整检查

先做一个不运行 LOPO 的快速检查：

```bash
python code/analyze_sliceeq_mpd_robustness.py \
  --root_path "$HOME/Documents/Updated_code/data/PROMISE12_h5" \
  --output_dir "$PWD/mpd_offline_audit_quick" \
  --grid_sides 21 \
  --axial_bins 3 \
  --moment_tolerances 0.02 \
  --residual_tolerances 0.05 \
  --utility_fractions 0.99 \
  --skip_lopo
```

快速检查通过后，再运行第 3 节的默认完整命令。

## 5. 输出文件

| 文件 | 用途 |
|---|---|
| `audit_summary.md` | 最适合直接阅读的汇总 |
| `grid_convergence.csv` | 网格收敛性 |
| `axial_bins.csv` | 轴向分箱敏感性 |
| `tolerance_sensitivity.csv` | 各约束常数的单因素敏感性 |
| `lopo_stability.csv` | 留一病例稳定性 |
| `constraint_activity.csv` | 约束余量与活跃状态 |
| `baseline_design.json` | 默认 21x21、3-bin 设计及完整诊断 |
| `distribution_comparison.json` | 各审计变体的概率分布 |
| `audit_results.json` | 完整机器可读结果 |
| `audit_config.json` | 参数、数据哈希及 data firewall 声明 |

CSV 使用 UTF-8 BOM，Excel 可以直接打开。

## 6. 如何解释结果

- 网格变化后，期望 neighbor mass、worst RFI 和 entropy fraction 接近，说明结论不是 21x21 网格的偶然产物。
- 轴向分箱和容差轻微变化时，`JS to reference` 较小且 worst gain 方向一致，说明分布对人为常数不敏感。
- 某个约束长期不活跃且删除后结果几乎不变，可以在论文中诚实说明它是安全护栏；若需要简化最终方法，可在冻结新协议后再删除，而不是直接改当前实验。
- LOPO 检查的是七个标注病例中是否由单个病例主导设计。它不是泛化性能或统计显著性的替代品。
- `historical gate` 只复现先前登记的诊断阈值，不用于重新挑选训练配置。

## 7. 单元测试

```bash
python -m unittest discover -s tests -p 'test_sliceeq_mpd_audit.py' -v
```

测试覆盖可变 profile 网格、相位镜像、动态轴向分箱、约束可行性，以及默认审计优化器对原 MPD 优化器的数值复现。
