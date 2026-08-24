# CoDA-MT for PROMISE12

## CODA Final reproducibility entry

For the final label-7 paper ablations, OAAC-Strong/MPD increments, paired
re-acquisition figures, train-only MPD audit, strict tests and automatic result
tables, use `run_coda_final_paper_suite.sh`. The complete cross-machine guide is
in [docs/CODA_FINAL_RUNBOOK.md](docs/CODA_FINAL_RUNBOOK.md).

## Current SliceEqOcc-OAAC method

The current selected line is **SliceEqOcc-OAAC**: paired through-plane
re-acquisition with fractional occupancy, followed by a monotonic
coordinate-preserving appearance perturbation on unlabeled student views.

- [Detailed method principles, literature provenance, contributions, results,
  limitations, and CVPR paper plan](docs/SLICEEQ_OCC_OAAC_MODEL_AND_PAPER_GUIDE.md)
- [OAAC training and evaluation instructions](docs/SLICEEQ_OCC_OAAC_README.md)
- [CVPR outline](research/paper/sliceeq_occ_cvpr_outline_2026-08-13.md)
- [Reviewed OAAC GitHub synchronization manifest](docs/GITHUB_OAAC_SYNC_MANIFEST.md)

The local baseline is a BCP-derived EMA scaffold with Copy-Paste removed; it
must not be described as the complete original BCP method. Data, checkpoints,
patient images, and external training logs are intentionally not versioned.

## SliceEqOcc independent optimization version

The fractional-occupancy successor uses independent entries
`code/train_sliceeq_occ.py` and `code/test_sliceeq_occ.py`; SliceEq v1 is not
overwritten. See `docs/SLICEEQ_OCC_README.md`. The new test entry defaults to
`--save_result False` (with `--save_results` as an alias) and disables
checkpoint auto-search.

## Archived earlier CoDA-MT research line

The material below records an earlier hypothesis and its implementation
history. It is retained for provenance but does not describe the currently
selected SliceEqOcc-OAAC method or its completed development run.

CoDA-MT（Corruption-aware Data–Target Coupling Mean Teacher）是一个面向
PROMISE12 半监督前列腺 MRI 分割的研究实现。它建立在去除 Copy-Paste 后的
BCP/Mean Teacher baseline 上，保持原有 2D U-Net、数据划分、采样器和训练日程，
只修改无标注样本的强增强与伪目标构造。

> 当前状态：代码、数据预检和 Windows 多进程加载已经验证；尚未产生可用于论文
> 结论的完整训练结果。仓库中的方法主张均应视为待实验验证的研究假设。

## 核心研究问题

常规 weak-to-strong 一致性在图像经过降采样或噪声破坏后，仍使用与原图同等确定的
硬伪标签。CoDA-MT 检验以下假设：强增强移除了多少局部证据，伪目标就应在对应位置
降低多少确定性，从而减少增强导致的过度自信和错误监督。

当前实现包含：

- 分辨率退化与高斯噪声强视图；
- 基于局部图像证据损失的空间软化系数；
- 与退化强度耦合的软伪目标；
- soft cross-entropy 与 soft Dice 自训练损失；
- 保留 teacher 前景最大连通域先验；
- 严格的 PROMISE12 固定划分预检。

完整创新论证、相关工作、失败判据及 B0–B6 消融方案位于 [`research/`](research/)。

## 仓库结构

```text
code/                 当前可运行 Baseline 与 CoDA-MT 源码
tests/                数值、梯度、原代码完整性及部署契约测试
research/             文献综述、研究思路、实验协议和研究日志
docs/CODA_README.md    桌面部署与运行说明
```

`train_baseline.py`、`test_baseline.py` 和 `dataloaders/dataset.py` 保留为原始
baseline；新方法通过 `train_coda.py`、`test_coda.py` 和 `utils/coda.py` 并行提供。

## 数据

数据集不会上传到本仓库。默认路径为：

```text
E:/Desktop/PROMISE12
```

目录需要包含固定的 `train.list`、`train_slices.list`、`val.list`、`test.list`，
以及 `data/` 下对应的 HDF5 文件。可通过 `--root_path` 指定其他位置，但预检仍会
严格验证既定划分：35 个训练病例、940 个训练切片、前 7 个病例对应前 191 个
有标注切片、5 个验证病例和 10 个测试病例。

## 环境与测试

建议复用已验证的 Conda 环境 `ct_projector_py310`：

```powershell
conda activate ct_projector_py310
pip install -r requirements.txt
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m unittest discover -s tests -p "test_*.py" -v
```

当前预期为 14 项测试全部通过。`KMP_DUPLICATE_LIB_OK` 只用于当前 Windows 测试
进程，不需要写入系统环境变量。

## 训练与测试

```powershell
Set-Location code
python train_coda.py
python test_coda.py
```

训练默认配置与 baseline 对齐：2D U-Net、二分类、256×256、10k 次监督预训练、
30k 次自训练、batch 24（12 labeled + 12 unlabeled）、SGD 0.01、EMA 0.99、
seed 1337。Windows 下训练和验证 DataLoader 使用可序列化的顶层 worker 回调，
仍保持 `num_workers=4` 及 `seed + worker_id` 行为。

## 论文实验路线

实验必须按预注册顺序执行：

1. B0：复现去除 Copy-Paste 的 Mean Teacher baseline；
2. B1：只加入固定强增强，验证硬伪标签的过度自信/负迁移曲线；
3. B2：仅使用 CoDA 强视图但仍采用硬目标；
4. B3：固定 label smoothing；
5. B4：仅随全局退化强度变化的软目标；
6. B5：图像感知但不与实际增强耦合的静态软目标；
7. B6：完整的局部证据耦合 CoDA-MT。

主指标为 Dice 和 HD95，同时报告 NLL、Brier score、ECE，并进行多随机种子、
退化强度曲线和机制相关性分析。只有 B1 明确暴露失效机制且 B6 同时优于 B3–B5，
才支持将 CoDA-MT 作为论文核心创新。

## 可复现性与边界

- PROMISE12 列表内容和顺序不得修改；LF/CRLF/BOM 差异不视为样本变化。
- 不提交数据、模型权重或包含患者信息的文件。
- 当前没有声称任何 Dice/HD95 提升；结果目录中的内容是代码验证记录。
- 原 Baseline 来源和文件哈希记录见 [`research/baseline/audit.md`](research/baseline/audit.md)。
