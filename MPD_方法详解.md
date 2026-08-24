# MPD（Moment Profile Design）方法详解

> 完整名称：SliceEqOcc-OAAC-Strong-MPD
> 代码模块：`code/utils/sliceeq_mpd.py`
> 定位：对最终方法「SliceEqOcc-OAAC-Strong」中**三切片 profile 采样模块**的增强，其余部分保持不变。

---

## 1. 一句话定位

MPD 不是一个新网络、新损失或新数据增强。它做的事情只有一件：

> 把原来在 `sigma ~ U(0.45, 0.85)`、`phase ~ U(-0.25, 0.25)` 上**人工均匀采样**的三切片融合权重，
> 换成由**标注训练集的精确 occupancy 驱动、受物理形态约束**求解出的一个**全局冻结离散分布 `q`**。

训练时仍然随机抽不同 profile，只是“抽哪些 profile 的概率”从均匀变成了稳健设计后的 `q`。

网络、损失、EMA、OAAC、batch、学习率、验证规则、推理方式**全部不变**，因此性能变化可归因于 profile 风险的重新分配。

---

## 2. 背景：它改进了什么

最终方法演化链（由简到强）：

| 方法 | 相邻图像 | 成对目标 | 分数占据 | OAAC |
|------|:---:|:---:|:---:|:---:|
| BCP 衍生 EMA | 否 | 否 | 否 | 否 |
| SliceEq hard | 是 | 是（argmax 硬标签） | 否 | 否 |
| SliceEqOcc | 是 | 是 | 是 | 否 |
| OAAC-Strong | 是 | 是 | 是 | 是（scale 1.25） |
| **OAAC-Strong-MPD** | 是 | 是 | 是 | 是（scale 1.25） |

MPD 与 Strong 唯一区别：profile 分布来源不同。Strong 用均匀采样，MPD 用离线稳健设计。

---

## 3. 三切片 profile 的基本测量算子

对中心切片 `z` 及其相邻两层，令：

```
X⁻ = X_{z-1},  X⁰ = X_z,  X⁺ = X_{z+1}
```

一个三抽头 profile 是：

```
w = (w₋, w₀, w₊),   wₖ ≥ 0,  Σ wₖ = 1
```

重新采集的图像为：

```
X̃_w = w₋·X⁻ + w₀·X⁰ + w₊·X⁺
```

**同一组权重同时作用于监督目标**：对 exact GT（labeled）或 teacher 伪标签（unlabeled）的 one-hot 表示：

```
Q̃_w = w₋·Q⁻ + w₀·Q⁰ + w₊·Q⁺
```

关键点：`Q̃_w` 不是类别不确定度，而是**模拟采集算子诱导的组织分数占据（fractional occupancy）**。图像的层间混合与监督目标的组织混合严格成对，避免“图像已变但目标仍是中心硬标签”的 mismatch —— 这正是 SliceEq hard 被淘汰的原因。

---

## 4. 用两个轴向矩理解三切片融合

定义：

```
b = w₋ + w₊ = 1 - w₀           # 邻层总质量
r = (w₊ - w₋) / b              # 方向比
g₁ = (X⁺ - X⁻) / 2
g₂ = X⁻ - 2X⁰ + X⁺
```

则融合残差可精确写成：

```
X̃_w - X⁰ = (b·r)·g₁ + (b/2)·g₂
```

含义：

- `b` → 邻层混合 / 轴向平滑的总体强度
- `b·r = w₊ - w₋` → 向前或向后的方向偏移
- `b²` → 增强强度的二阶能量
- `(b·r)²` → 方向位移的二阶能量

MPD 的所有矩约束都建立在这两个量 `b` 和 `b·r` 上。

---

## 5. 为什么不用固定 `[0.2, 0.6, 0.2]`

生产代码不是固定比例。`[0.2, 0.6, 0.2]` 只是 `phase=0, sigma≈0.6746` 的一个点。

MPD 也不把它替换成另一个固定比例。它是在父 Gaussian 支持内，设计一个**有足够熵的分布**，保留多种合法重采集状态，同时让不同病人、不同轴向区域的分数占据信息更均衡。

---

## 6. MPD 优化对象：一个分布，而非一组权重

MPD 在父采样支持内构造 `21 × 21 = 441` 个候选 profile（midpoint grid，`sigma` 与 `phase` 各取 21 个中点）。父分布 `p₀` 对这 441 个点均匀采样。

MPD 求解一个全局离散分布 `q`：训练时仍然用 `torch.multinomial` 按 `q` 随机抽取 profile，而不是把所有切片固定成某个比例。

`sigma` 范围锁定 `[0.45, 0.85]`，`phase` 范围锁定 `[-0.25, 0.25]`。

---

## 7. RFI：Retained Fractional Information

MPD 的目标函数基于一个叫 RFI 的量。对训练标注栈和某个 profile `w_g`，先算：

```
Q_{n,g} = A_{w_g}(onehot(Y_n))
```

逐像素判定：若某像素的相邻层语义与中心层**不同**，但融合后的 harg-max 类别**仍等于中心层语义**，则该像素的 occupancy entropy 被计入 RFI：

```
opportunity:  相邻层标签 ≠ 中心层标签（真实存在轴向语义变化的位置）
retained:      argmax(Q) == 中心层 semantic
U_{n,g} = Σ_v [ opportunity(v) · retained(v) · H(Q_{n,g}(v)) ] / (Σ_v opportunity(v) + ε)
```

RFI 奖励的是同时满足三个条件的 profile：

1. 确实产生了分数占据 / 部分容积；
2. 不把中心层的主要语义翻转；
3. 信号出现在**真实存在轴向标签变化**的位置。

分子分母按「病人 × 轴向 index-third」聚合，共 `7 × 3 = 21` 个 strata（每个病人的切片按轴向位置三等分）。

---

## 8. 两阶段稳健优化

用 SLSQP 求解。第一阶段（max-min RFI）：

```
max_q  min_s  Σ_g q_g · u_{s,g}
```

即最大化**最弱**病人-轴向区域的期望 RFI，避免 profile 风险主要服务某个病人或某个密集区域。

第二阶段（KL 投影）：

```
min_q  D_KL(q ‖ p₀)
s.t.   Σ_g q_g · u_{s,g} ≥ 0.99 · t*   （保留第一阶段 99% 最优值）
```

让最终 `q` 尽量靠近父均匀分布，只做「解决最弱区域所必需的最小重分配」。

---

## 9. 硬约束清单（不可调）

这些是锁死的设计约束，不是训练超参数，不允许按 validation/test 再调：

| 约束 | 值 |
|------|----|
| Phase 镜像对称 | `q(σ, φ) = q(σ, −φ)` |
| 一阶矩 `E[b]` | 相对父分布 ±2% |
| 二阶矩 `E[b²]` | 相对父分布 ±2% |
| 二阶矩 `E[(b·r)²]` | 相对父分布 ±2% |
| 每个 patient-stratum 的图像 RMS residual | 相对父分布 ±5% |
| 单点密度上限 | `q/p₀ ≤ 3` |
| 熵下界 | `H(q) ≥ 0.70 · H(p₀)` |

镜像对称通过一个 `_mirror_projection` 投影矩阵实现，把相位对称的候选对捆成一个变量。

---

## 10. 实际求得的分布说明了什么

父分布与 MPD 设计分布的关键矩（来自已冻结的 artifact）：

| 统计量 | 父分布 | MPD | 相对变化 |
|---|---:|---:|---:|
| `E[b]` | 0.375076 | 0.382577 | **+2.00%** |
| `E[b²]` | 0.149946 | 0.152546 | +1.73% |
| `E[(b·r)²]` | 0.014946 | 0.014647 | **−2.00%** |

由于相位严格对称，平均权重约为：

```
E_q[w] ≈ [0.1913, 0.6174, 0.1913]
E_p0[w] ≈ [0.1875, 0.6249, 0.1875]
```

求解器选择了一个**清晰但克制**的策略：

- 略微增加邻层总贡献与二阶轴向混合；
- 同时减少方向偏移能量（更对称）；
- 实际熵保留父分布的 **98.83%**；
- 最大密度比仅 **1.608**，远低于上限 3。

所以 MPD 不是“更强地模糊”，而是把增强风险向「稍强、更对称、在最弱病人/轴向区域仍能产生有效分数占据」的 profile 重新分配。

---

## 11. 为什么能提高分割结果（4 个机制）

1. **提高有效边界监督，而非增加普通样本数量**：让更多采样概率落在能产生 fractional occupancy 且不翻转中心语义的 profile 上，网络看到的是更稳定的亚像素/部分容积边界监督。
2. **降低无意义的方向噪声**：`E[(b·r)²]` 下降，减少“向上一层或下一层偏移”的随机位移能量，避免把相邻层伪标签错误带入中心层。
3. **防止长体积/中间区域支配采样风险**：max-min 目标让最弱 patient/index-third 也获得足够 RFI。
4. **与 OAAC 互补而非重复**：MPD 改「采集域+监督占据分布」，OAAC 改「重采集后 U 图像的 gamma/contrast/brightness，不碰空间坐标和 occupancy target」，两者覆盖采集变化与外观变化，可叠加。

---

## 12. 为什么增益不会特别大

MPD 被刻意限制在父分布附近：矩 ±2%、扰动 ±5%、熵保留 98.83%。它是在已达到 `Dice 0.851960` 的强方法上重新分配训练风险，而非增加网络容量。

开发结果：

- OAAC-Strong：Dice `0.851960`
- OAAC-Strong-MPD（iter 29000）：Dice `0.854573`
- 增益：`+0.002613` Dice、`+0.003983` Jaccard

小的结构化调整量对应小的、一致的增益，符合预期。

---

## 13. 代码结构

| 文件 | 作用 |
|------|------|
| `code/utils/sliceeq_mpd.py` | 精确训练统计、两阶段 SLSQP 设计、artifact 校验、冻结采样器 |
| `code/train_sliceeq_occ_oaac_strong_mpd.py` | 独立入口：只注入全局 sampler，然后复用父训练 |
| `code/test_sliceeq_occ_oaac_strong_mpd.py` | 严格指定 checkpoint 的 2-D 推理入口 |
| `tests/test_sliceeq_mpd.py` | 数值、优化器、RNG 合同测试 |
| `tests/test_sliceeq_mpd_contract.py` | 父文件 hash、配方、数据防火墙、测试入口静态合同 |

父训练 `train_sliceeq_occ_oaac_strong.py` 与 1000-step 基座保持 **byte-identical**。MPD 唯一的运行时注入：

```python
strong.parent.sample_slice_profiles = sampler
```

其余（optimizer、EMA、validation、checkpoint）全部复用。

关键实现细节：

- **数据防火墙**：设计阶段只读 `train_slices.list` 前 191 张切片（7 个 labeled 病人 image+label），不读 U label、validation、test、checkpoint、prediction、confidence、loss。
- **私有 RNG**：`FrozenProfileSampler` 用独立 `torch.Generator`，不消耗父训练的全局 RNG 流（通过 smoke test 校验）。
- **可复现性**：artifact 里记录 `distribution_sha256`、训练数据内容 hash、所有约束的诊断结果；`atomic_json_dump` 保证原子写入。

---

## 14. 训练与测试命令

训练（在训练机，先自动写入 `mpd_profile_design.json`，再执行 30k self-training）：

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u train_sliceeq_occ_oaac_strong_mpd.py \
  --pretrained_checkpoint /home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth
```

输出目录：

```
../model/SliceEqOccOAACStrongMPD_PROMISE12_7_labeled/self_train/unet/
```

其中包含 `mpd_profile_design.json`、`iter_1000.pth`…`iter_30000.pth`、`unet_best_model.pth`、`log.txt`。

指定 iteration 测试：

```bash
cd /home/aiteam/zhengtaoma/CoDA/code
python -u test_sliceeq_occ_oaac_strong_mpd.py \
  --root_path /home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source \
  --checkpoint_path ../model/SliceEqOccOAACStrongMPD_PROMISE12_7_labeled/self_train/unet/iter_27000.pth \
  --auto_find_checkpoint False \
  --save_result False
```

测试入口严格 `strict=True` 加载 state_dict，并调用原 `test_coda` 的 2-D 推理得到 Dice / Jaccard / HD95 / ASD。

---

## 15. 论文表述边界

**可以说的**：

> a moment-resolved, training-only robust design of the paired through-plane image–occupancy profile distribution.

即：把 heuristic 的 profile sampling 改写为在 exact training occupancy、轴向 moment 和图像扰动预算下求得的全局稳健设计。

**不能说的**：

- 恢复真实 scanner PSF / 层厚；
- 物理标定；
- 首次 MixUp / DRO / optimal augmentation；
- 已证明“全局最佳融合比例”；
- 已证明 profile 设计可跨病人/跨站点泛化（需 MM-WHS 等未参与设计的数据验证）。

主创新仍是 SliceEqOcc 的 paired re-acquisition + fractional occupancy；MPD 是 profile-design 组件。

---

## 16. 最终方法链总结

1. 用 exact labeled-training occupancy 离线设计并冻结全局 MPD 分布；
2. 从 MPD 抽取三切片 profile；
3. 对 L 图像/GT 和 U 图像/teacher pseudo mask 使用同一 profile；
4. 保留连续 fractional occupancy，用 soft CE + squared Dice 监督；
5. 仅对重采集后的 U 学生图像施加 OAAC-Strong（scale 1.25）；
6. 使用原 EMA、U-Net、一致性 ramp 完成训练；
7. 推理时丢弃所有三切片 / MPD / OAAC 路径，仍用原 2-D 单切片 U-Net。

一句话逻辑链：

> **设计采集风险 → 成对模拟采集 → 保留组织占据 → 扩展外观覆盖 → 零推理开销。**