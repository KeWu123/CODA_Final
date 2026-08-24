# SliceEqOcc-OAAC-Strong-MPD：直接完整训练版

## 1. 版本定位

该版本只深化最终方法中的三切片 profile 采样模块。它从已经选定的
`SliceEqOcc-OAAC-Strong` 独立派生，不修改任何父文件。用户明确要求不先做 LOPO/零训练
gate，因此训练入口会在启动时完成一次全局 profile 设计，写出设计 artifact，随后立即执行
原来的 30k self-training。该运行应标记为 exploratory direct run，而不是已经通过外推稳定性
验证的 confirmatory run。

保持不变的内容包括：U-Net、7 labeled/191 slices、共享 Pre10000 `net+opt`、seed1337、
SGD 与 LR0.01、EMA0.99 train mode、teacher hard argmax + 2-D LCC、exact-L/pseudo-U
fractional occupancy、soft CE + squared Dice、consistency ramp、loader24/student36、三切片
support、OAAC Strong scale1.25、30k、每200 iter验证、原 best-model 规则、每1000 iter普通权重
保存，以及单切片2-D推理。

唯一改变：父方法从连续均匀的
`sigma~U(0.45,0.85), phase~U(-0.25,0.25)` 采样，改为从训练前冻结的全局离散分布
`q` 采样。每次得到的同一个三权重仍同时作用于图像和 exact/pseudo occupancy。

## 2. 为什么不是寻找一个固定的 0.2/0.6/0.2

生产代码并不是固定比例。对权重
`w=(w_minus,w0,w_plus)`，定义：

```text
b = w_minus + w_plus = 1 - w0
r = (w_plus - w_minus) / b
```

`b` 表示邻层总质量，`r` 表示方向偏移。代表性的 `[0.2,0.6,0.2]` 只是
`phase=0, sigma≈0.6746` 的一个点。MPD 也不把它替换成另一个固定比例，而是在父 Gaussian
支持内设计一个具有足够熵的分布；这样保留多种合法重采集状态，同时使不同训练病人和轴向索引
区域获得更均衡的有效 fractional-occupancy 信息。

## 3. 一次性全局设计

训练启动时仅读取 `train_slices.list` 的前191张切片，即完整的7个 labeled-training 病人。
不会读取 U label、validation/test、checkpoint、prediction、confidence 或 loss。

候选是锁定的 `21×21` midpoint 网格，共441个 `(sigma,phase)` profile。对每个 labeled stack
和 profile，先以 exact GT 得到 occupancy `Q=A_w(onehot(Y))`。只在邻层与中心层存在标签差异、
且 `argmax(Q)` 仍等于中心语义的位置，累计 occupancy entropy，形成 Retained Fractional
Information（RFI）。先按病人和轴向索引三等分聚合为21个 strata。

优化分两步：

1. 最大化21个 strata 中最差的期望 RFI；
2. 在达到第一阶段最优值99%的分布中，最小化 `KL(q||p0)`，选择最接近父均匀分布的解。

硬约束为：phase 镜像对称；`E[b]`、`E[b²]`、`E[(br)²]` 相对父分布在±2%；每个
patient-stratum 的归一化图像 RMS residual 在±5%；`q/p0≤3`；
`H(q)≥0.70H(p0)`。这些不是训练超参数，不允许根据 validation/test 再调。

如果某个 patient-stratum 的所有相邻 exact labels 均完全相同，则其 RFI opportunity 分母为0，
任何 profile 在该区域的 RFI 都只能是未定义而不是“性能为0”。实现会把这种结构性空 stratum
仅从 RFI max-min 目标中排除，但仍把它保留在图像 residual 预算和 artifact 报告中；每个病人仍
必须至少提供一个有效 RFI stratum。该规则不读取模型、validation 或 test，也不按结果调阈值。

优化器若不可行或未收敛，入口会停止；它不会自动放宽约束。成功时在模型目录原子写入：

```text
mpd_profile_design.json
```

其中包含441维概率、distribution SHA-256、训练数据内容 hash、每个 stratum 的父/设计 RFI、
moment/residual/entropy/density 约束诊断，以及“用户跳过 LOPO gate”的证据等级说明。

## 4. 代码结构

- `code/utils/sliceeq_mpd.py`：exact-training 统计、两阶段 SLSQP 设计、artifact 校验、冻结采样器；
- `code/train_sliceeq_occ_oaac_strong_mpd.py`：独立入口，只注入全局 sampler，然后调用父训练；
- `code/test_sliceeq_occ_oaac_strong_mpd.py`：严格指定 checkpoint 的原2-D测试入口；
- `tests/test_sliceeq_mpd.py`：数值、优化器与 RNG 合同；
- `tests/test_sliceeq_mpd_contract.py`：父文件 hash、配方、数据防火墙和测试入口静态合同。

父训练 `train_sliceeq_occ_oaac_strong.py` 和1000-step基座
`train_sliceeq_occ_h7_15_base.py` 均保持 byte-identical。MPD wrapper 不复制 optimizer、EMA、
validation 或 checkpoint 代码。

## 5. 训练命令

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u train_sliceeq_occ_oaac_strong_mpd.py \
  --pretrained_checkpoint /home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

输出目录：

```text
../model/SliceEqOccOAACStrongMPD_PROMISE12_7_labeled/self_train/unet/
```

普通周期权重仍是 `iter_1000.pth` 到 `iter_30000.pth`；验证 best 保存规则未变。

训练机具备依赖时建议先执行以下实现检查。按照用户本次要求，它们是上线检查而不是决定是否允许
训练的方法 gate：

```bash
cd /home/aiteam/zhengtaoma/CoDA
python -m unittest tests.test_sliceeq_mpd tests.test_sliceeq_mpd_contract -v
```

## 6. 指定 iteration 测试

例如测试 `iter_27000.pth`：

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u test_sliceeq_occ_oaac_strong_mpd.py \
  --root_path /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source \
  --checkpoint_path ../model/SliceEqOccOAACStrongMPD_PROMISE12_7_labeled/self_train/unet/iter_27000.pth \
  --auto_find_checkpoint False \
  --save_result False
```

## 7. 论文表述边界

可以表述为：

> a moment-resolved, training-only robust design of the paired through-plane
> image–occupancy profile distribution.

不能表述为 scanner PSF/thickness 恢复、物理标定、首次 MixUp/DRO/optimal augmentation，或
已经证明“全局最佳融合比例”。该组件的贡献是把 heuristic profile sampling 改写为在 exact
training occupancy、轴向 moment 和图像扰动预算下求得的全局稳健设计；主创新仍是 SliceEqOcc
的 paired re-acquisition 与 fractional occupancy，OAAC 仍是 acquisition 后的 U-only
target-invariant appearance consistency。

由于本轮跳过 LOPO gate，阳性结果也只能说明在当前训练/验证协议下具有探索性收益。论文主张仍需
MM-WHS 等未参与该 profile 设计的数据验证其迁移性。
