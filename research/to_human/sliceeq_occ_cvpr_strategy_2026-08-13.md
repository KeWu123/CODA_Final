# SliceEqOcc 的 CVPR 构建与下一步优化路线

日期：2026-08-13  
当前主方法：`SliceEqOcc`（论文中建议统一写作 **SliceEq**，其中 `Occ` 明确指 fractional occupancy，而不是 occlusion）  
已接受的开发阶段结果：PROMISE12 Dice = **0.844566**。本文档按用户要求接受该结果，不再在本机复核训练日志或 checkpoint。

锁定的核心方法增量为：硬目标前身 `SliceEq ≈ 0.832603`，完整 fractional-occupancy 方法 `SliceEqOcc ≈ 0.844566`，即约 **+0.011963 absolute Dice（+1.2 个百分点）**。因此论文的方法主角是 SliceEqOcc；SliceEq 应作为最关键的 hard-target ablation。

锁定的完整开发结果链为：

| 阶段 | Dice | 相对前一阶段 |
|---|---:|---:|
| BCP-derived EMA baseline（去 Copy-Paste） | 0.78--0.80 | — |
| SliceEq（hard target 前身） | 约 0.832603 | +0.0326--0.0526 |
| SliceEqOcc（完整主方法） | 约 0.844566 | +0.011963 |

这形成两级论文证据：paired slice-profile re-acquisition 带来第一阶段提升，fractional occupancy 带来第二阶段提升。由于 baseline 与 SliceEqOcc 的实际 batch/view 设计不同，第一阶段的正式归因必须由 B0-36、ImgOnly 和 SliceHard 等 matched controls 支撑，不能仅凭数值差直接断言全部来自 re-acquisition。

## 1. 结论

当前不应先做大规模超参数搜索，也不应继续给 SliceEqOcc 叠加 uncertainty、boundary weighting、attention、posterior softening 或 SAQ。最优顺序是：

1. 保留 SliceEqOcc 为主方法，先补齐严格的因果对照与公平训练协议；
2. 用 3--5 个优化 seed 和至少一个未参与开发的第二 MRI 数据集确认收益；
3. 只在验证集上做一个小而预先声明的稳定性调参；
4. 如果仍需要论文级方法增强，优先做 **protocol-conditioned, scan-coherent SliceEqOcc**，而不是增加通用模块。

只要 matched controls 证明完整方法稳定优于 image-only、hard-target 和 compute-matched controls，SliceEqOcc 本身已经具备一条干净的论文主线。后续模块不是投稿的先决条件。

## 2. 论文的核心问题

主流半监督分割通常假定增强是 label-invariant 的：图像被强扰动后，中心切片的硬 GT 或伪标签仍可作为监督。但 MRI 的 through-plane acquisition 是非可逆的。有限层厚和层面相位会把相邻解剖组织积分到同一个观测平面，因此增强后的正确监督不再是原中心切片的硬标签，而应是由同一成像算子诱导的 fractional tissue occupancy。

SliceEq 的核心命题应写成：

> Consistency under a non-invertible acquisition perturbation is meaningful only when both the observation and its supervision are transformed by the same image-formation operator.

中文表述：

> 对非可逆采集扰动，只有当观测与监督语义经过同一成像前向算子时，一致性学习才是定义正确的。

## 3. 方法形式化

对中心位置 `z` 的三层图像栈和分割栈，记为

\[
X_z=\{x_{z-1},x_z,x_{z+1}\},\qquad
Y_z=\{y_{z-1},y_z,y_{z+1}\}.
\]

采样 slice-profile 参数 `h=(sigma, phi)`，得到归一化权重

\[
w_k(h)=\frac{\exp[-(k-\phi)^2/(2\sigma^2)]}
{\sum_{j=-1}^{1}\exp[-(j-\phi)^2/(2\sigma^2)]},
\quad k\in\{-1,0,1\}.
\]

同一个算子同时产生虚拟观测与组织占据率：

\[
\widetilde x_z=\sum_k w_k x_{z+k},\qquad
\widetilde y_z=\sum_k w_k\,\operatorname{onehot}(y_{z+k}).
\]

未标注分支将 `y` 替换为 EMA teacher 的离散伪分割。当前实现的目标为

\[
\mathcal L=
\frac{1}{2}\left[
\ell(f(x_z^L),y_z^L)+
\ell(f(\widetilde x_z^L),\widetilde y_z^L)
\right]
+\lambda(t)\ell(f(\widetilde x_z^U),\widetilde y_z^U),
\]

其中 `ell` 是适用于 fractional occupancy 的 soft cross-entropy 与 soft Dice 均值。测试时仍只输入单张二维切片，没有额外网络、邻层输入或推理开销。

## 4. 为什么 SliceEqOcc 有效，而 SAQ 没有效果

### 4.1 SliceEqOcc 的有效信号

- 图像与目标共享同一个 profile，避免了“邻层混合图像 + 中心硬标签”的语义错配；
- fractional occupancy 保留 through-plane partial volume，而 SliceEq v1 的 `argmax` 几乎把该信息抹掉；
- exact-GT reacquired labeled view 提供无伪标签噪声的 acquisition-equivariance teaching；
- 原始中心硬标签分支保留 clean anchor，避免模型只学习模糊占据率；
- 已有 gate 显示 acquisition residual 虽只覆盖约 0.8427% 像素，却贡献约 65.65% 的完整梯度，说明信号稀疏但并未被全图损失稀释。

对应的开发结果链为 `0.832603 -> 0.844566`。这约 +0.012 Dice 的增量正好与“argmax 丢弃 fractional occupancy、完整 Occ 保留采集诱导组织比例”的机制相对应。正式论文仍需用 matched `SliceHard` 对照和多 seed 证明归因，但叙事上应把这一对比放在最核心位置。

### 4.2 SAQ 的负结果是可解释且有价值的

当前 SAQ 在一个 12-sample branch 中把四个 Gauss--Legendre 节点分给不同 anatomy sample。它平衡的是 batch marginal profile count，而不是同一 anatomy 条件下的 acquisition risk。因此，它不能消除真正的 sample-conditional 梯度方差。

数值上，连续 IID sampler 的中心权重约覆盖 `0.4851--0.8552`，均值约 `0.6249`；SAQ 只产生约 `0.5325` 和 `0.7179` 两档，均值仍约 `0.6252`。SAQ 基本保留一阶平均 severity，却删除了轻/重扰动 tails，并把连续 profile 离散成两档。因此“无提升”更支持：

1. 当前瓶颈不是 batch-level profile 均值估计；
2. 极端 profile 覆盖或 anatomy-conditional risk 比低阶跨样本 quadrature 更重要；
3. 不应继续增加 SAQ 节点或调 quadrature 阶数。

SAQ 可作为 appendix 的负消融，用来说明 SliceEqOcc 的收益不是简单的采样均匀化。

## 5. 能守住的新颖性边界

不能声称：

- 首次使用相邻切片；
- 首次模拟 MRI partial volume 或 slice thickness；
- 首次联合变换图像和标签；
- 首次从增强得到 soft label；
- 首次使用 slice profile；
- 首次处理模糊边界或不确定性。

可辩护的窄而清晰的创新是：

> 在二维半监督 teacher--student 分割中，用同一个 through-plane slice-profile 前向算子重采集真实邻层图像和 exact/pseudo tissue occupancy，以 acquisition-aligned fractional supervision 学习，同时保持单切片二维推理。

它与已有工作的区别为：

- BCP/ABD/MOST 等改变图像内容或 patch 组成，监督仍主要是拼接的硬 GT/伪标签；
- UniMatch/AugSeg 关注通用 weak-to-strong perturbation，并未定义非可逆采集后的目标语义；
- PV-SynthSeg/SynthSeg 使用生成模型获得跨分辨率鲁棒性，但不是针对未标注真实图像的 EMA pseudo-occupancy consistency；
- AmbiSSL 等处理 annotation ambiguity，而 SliceEq 处理 acquisition-induced measurement ambiguity。

## 6. 建议的论文标题、motivation 和 contributions

### 6.1 推荐标题

首选：

> **SliceEq: Acquisition-Equivariant Fractional Occupancy for Semi-Supervised MRI Segmentation**

更保守：

> **SliceEq: Paired Slice-Profile Re-acquisition for Semi-Supervised MRI Segmentation**

若最终只有前列腺数据，应把 `MRI Segmentation` 收窄成 `Prostate MRI Segmentation`；只有在第二器官 MRI 数据集也成立后再使用更广标题。

### 6.2 Motivation 三步链条

1. 医学 SSL 依赖强增强，但通常默认增强后标签语义不变；
2. MRI through-plane acquisition 会混合相邻组织，该假设不成立，硬中心标签与重采集图像存在结构性错配；
3. 因而增强不只要“医学上合理”，还必须同步改变监督语义。SliceEq 用同一 operator 产生重采集图像和 fractional occupancy。

### 6.3 建议 contributions

1. 提出 acquisition-equivariant consistency 视角：指出非可逆层面采集扰动下的 label-invariance 失效，并给出 paired image--target forward operator；
2. 提出 SliceEq，通过 slice-profile 重采集 exact masks 与 EMA pseudo masks，生成由成像算子确定、而非人工平滑的 fractional occupancy supervision；
3. 在不改变二维推理网络和推理成本的前提下，通过 compute-matched、target-matched、跨采集协议和跨数据集实验验证其有效性。

Introduction 中可以先报告上述开发轨迹 `0.78--0.80 -> 0.832 -> 0.844`，随后马上说明完整论文使用 matched controls 拆分 re-acquisition、额外 view 与 fractional occupancy 的贡献。这样既保留清晰的经验动机，又避免过度归因。

第三条只有在相应实验完成后才能写成完成时。SAQ、residual weighting 和 posterior commutation 不列为贡献。

## 7. 发表前必须先做的公平性修复

这些是所有方法共同的 research infrastructure，不是 SliceEq 的新模块：

1. **Baseline 命名**：当前去掉 Copy-Paste 后应称 `BCP-derived EMA hard-pseudo-label scaffold`，不能称为 BCP；实验表另列原版 BCP。
2. **Batch/compute matched**：baseline student batch 是 24，SliceEqOcc 是 36，且多一个 exact-GT derived view；必须有 36-view baseline。
3. **Teacher mode matched**：当前 EMA teacher 未调用 `eval()`，而 U-Net 包含 BatchNorm 和 dropout；baseline 与 SliceEqOcc 的 teacher batch、前向顺序也不同。统一 teacher policy 后所有核心方法重跑。
4. **Shared initialization**：每个 seed 的所有 self-training 方法加载同一个 `net+optimizer` pretrain hash，并在 self-training 起点重置相同 RNG。
5. **Unlabeled-label firewall**：未标注 loader 不应读取或搬运 GT；增加 sentinel test，证明替换 U labels 后 loss/gradient/update 不变。
6. **Optimizer 修复**：加载 optimizer state 会覆盖 CLI `base_lr`；若调 LR，必须显式重设 param groups 并记录实际值。
7. **完整 checkpoint**：保存 student、EMA、optimizer、scheduler、RNG、iteration 和配置，支持恢复与机制分析。
8. **物理指标**：HD95/ASD 使用真实 spacing，补 NSD；统一 empty-mask policy。现有 legacy HD95/ASD 不应写成毫米。

## 8. 最小因果实验矩阵

所有方法使用同一 seed-specific pretrain、相同更新次数、teacher mode、batch/BN policy、验证频率和 checkpoint 规则。

| ID | 方法 | 目的 |
|---|---|---|
| B0 | 当前 EMA scaffold，batch 24 | 复现历史参考 |
| B0-36 | 额外普通/重复 labeled view，batch 36 | 匹配 compute、BN composition 和监督 view 数量 |
| ImgOnly-36 | `A_h(image)` + 中心硬 GT/pseudo target | 测试邻层平滑，不同步改变语义 |
| SliceHard-36 | 图像与 mask 同 profile，但 occupancy 立即 argmax | 隔离 fractional occupancy 的价值 |
| SliceEqOcc-36 | 完整方法 | 主方法 |

最低先做 5 方法 × 3 seeds = 15 次 self-training。随后把 Full 和最强 matched control 补到 5 seeds。机制消融再增加：

- L-only fractional occupancy；
- U-only fractional occupancy；
- original BCP；
- UniMatch；
- 普通 through-plane blur/mean target-invariant control。

主比较预注册为 `SliceEqOcc vs strongest compute-matched non-paired control`；机制比较为 `SliceEqOcc vs SliceHard`。若多个核心比较，使用 Holm 校正。

## 9. 调参：可以做，但只能作为第二阶段

调参可能提高 Dice，但不能构成 contribution。只在冻结 test 后进行一个小网格，并以多 seed 验证均值和方差，而非选择单个最高 checkpoint。

建议顺序：

1. supervised anchor 权重 `alpha`：原始中心 / reacquired labeled 在 `{0.75/0.25, 0.5/0.5, 0.25/0.75}` 中选；
2. unlabeled 最大权重：当前最大约 0.5，比较 `{0.25, 0.5}`；
3. ramp duration：只比较 `{当前值, 更短的一个预声明值}`；
4. profile range 最后调，并优先依据 spacing/thickness，而不是扩大随机区间。

停止规则：若某参数只提高单 seed 或单 checkpoint、没有改善 validation-selected 多 seed 均值，则不采用。不要同时扫 `sigma × phase × alpha × lambda × ramp`。

## 10. 下一项最值得尝试的论文级方法增强

### Protocol-Conditioned, Scan-Coherent SliceEqOcc

当前 profile 是固定 slice-unit 区间内逐样本 IID 采样，存在两个科学缺口：

1. 不同病例的 slice spacing/thickness 不同，相同 `sigma` 表示不同毫米尺度；
2. 真实 acquisition protocol 是 scan-level 的，而不是相邻切片各自拥有独立 profile。

建议的新假设：

> 将 SliceEq 的 profile 从逐切片 IID heuristic 升级为以物理坐标和病例 acquisition metadata 条件化、并在同一扫描内保持一致的 virtual acquisition，可提高跨层厚/跨中心鲁棒性，同时保持相同的 paired occupancy 机制与零额外推理成本。

实现优先级：

1. **Scan coherence**：由 `(seed, epoch/refresh, case_id)` 决定一个 profile，同一病例在该 refresh window 内共享；保持边际分布与 SliceEqOcc 一致，先单独测试 correlation structure；
2. **Spacing-conditioned weights**：在毫米坐标计算 tap 距离；若只有 spacing，明确称 spacing-conditioned synthetic profile；
3. **Thickness-aware native-to-target composition**：只有获得可信 thickness/profile metadata 后才使用，且只模拟 target profile 不窄于 native profile；不能把 spacing 冒充 slice thickness；
4. **Metadata-shuffle control**：随机打乱病例 metadata，排除仅改变 severity distribution 的解释。

主要终点不应只看同分布 Dice，还应看 synthetic thickness robustness AUC、method × thickness interaction、跨中心/外部集，以及 apex/mid/base。

该扩展只有在前述因果 controls 证明 SliceEqOcc 核心成立后再实现。若 metadata 不完整，保留当前 SliceEqOcc，使用 `acquisition-inspired discrete profile` 的收窄表述即可。

## 11. 不推荐作为下一步的方向

- 继续 SAQ 或提高 quadrature 阶数；
- acquisition residual/boundary 再加权；已有 gate 已证明该区域梯度很强；
- raw posterior 或 topology-gated posterior；已有 fidelity gate 失败；
- uncertainty filtering、attention、对比学习、通用 gradient surgery；创新空间拥挤且会模糊主线；
- 直接改成 3D backbone；会失去“训练利用邻层物理、推理仍为轻量 2D”的核心卖点；
- 在已看过的 PROMISE12 test 上继续选 sigma、phase 或 checkpoint。

## 12. CVPR 级评价协议

- 接受 0.844566 为已确定的开发结果，但正文无偏主表应来自预注册的 test-independent checkpoint 规则或未查询外部测试；
- 至少 3 个 optimization seeds，Full 与最强控制建议 5 个；
- optimization seed 与 labeled-subset draw 分开报告；第二标注预算至少再做一个；
- 每 run 保存逐病例 Dice、Jaccard、physical HD95、physical ASD、NSD 和预测；
- 统计单位是 patient，不是 slice；报告 paired difference、median difference、win rate、95% hierarchical bootstrap CI 及配对 permutation/Wilcoxon；
- 增加第二个 anisotropic MRI 数据集，优先选择能恢复原始 spacing/thickness 的 whole-organ segmentation 数据；
- 报告训练 FLOPs、峰值显存、时间；强调 SliceEq 推理 FLOPs 与 baseline 完全相同。

## 13. Go / No-Go 判据

### SliceEqOcc 可作为 CVPR 主方法

同时满足：

1. Full 对 ImgOnly 和 SliceHard 的多 seed 平均差为正，至少主要比较的 95% CI 不跨 0；
2. 对 B0-36 的提升仍存在，排除 batch/额外 GT view 解释；
3. 第二数据集或跨 protocol 分层方向一致；
4. teacher/BN 与 initialization 公平性修复后结论不消失；
5. zero inference overhead 的主张有测量支持。

### 降级为增强 recipe，而非强 CVPR contribution

出现任一核心反证：

- Full 与 ImageOnly/Hard control 打平；
- 提升完全由 B0-36 或 extra labeled view 解释；
- 只在一个 seed、一个 checkpoint 或单一数据集出现；
- acquisition shift 下没有可重复的 interaction；
- 物理表述依赖不存在或不可信的 thickness metadata。

## 14. 最终建议

现在最值得投资的不是再追一个单点 `+0.3 Dice`，而是把 `0.844566` 转化为可信、可归因、可泛化的论文证据。先完成因果矩阵；如果 SliceEqOcc 在严格条件下仍胜，它本身就是主方法。若还要升级，再做 protocol-conditioned、scan-coherent 版本。这一方向既可能继续提升结果，也直接强化 motivation、contribution 和跨协议实验，而不是把论文变成多个通用模块的拼接。
