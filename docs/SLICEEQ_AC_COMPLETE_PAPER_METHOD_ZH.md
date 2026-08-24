# SliceEq-AC：完整论文方法与实验论证

> 论文工作名：**SliceEq-AC: Acquisition-Aligned Semi-Supervised MRI Segmentation with Fractional Occupancy**
>
> 代码实现名：`SliceEqOcc-OAAC-Strong-MPD`
>
> 核心原则：**当增强模拟 MRI 层间采集并改变图像中的组织组成时，监督目标必须通过同一个采集算子同步变化。**

## 1. 方法定位

SliceEq-AC 不是一个新分割网络，也不是把多个现成增强模块堆在一起。它重新定义了半监督 MRI 分割中“增强图像与监督目标应该如何配对”这一训练问题。

普通 weak-to-strong consistency 通常假设增强只改变外观，因此增强后的 Student 图像仍可使用中心切片的硬标签或 Teacher 伪标签。这个假设适用于亮度、对比度等坐标保持变换，但不适用于 through-plane re-acquisition：一张有限层厚的 MRI 切片可以包含相邻层的组织信号，重采集后的图像语义不再严格等于中心切片的二值标签。

SliceEq-AC 将一次训练样本写成一个成对测量：

\[
(\widetilde x, q)=\bigl(\mathcal A_w(x),\mathcal A_w(y)\bigr),
\]

其中同一个 slice profile \(w\) 同时作用于图像信号和组织占据目标。由此，图像与答案始终描述同一次虚拟采集。

## 2. 统一方法视角

最终方法只有一个主线，可分为四个相互依赖的定义，而不是四个互不相关的插件：

| 符号 | 论文名称 | 在统一方法中的职责 |
|---|---|---|
| SRA | Slice-profile Re-Acquisition | 定义虚拟 MRI 层间采集算子 |
| AFO | Aligned Fractional Occupancy | 定义该采集算子下正确的监督语义 |
| OAAC | Ordered Acquisition-Appearance Consistency | 在目标已经对齐后扩展坐标保持的外观域 |
| MPD | Moment-constrained Profile Design | 仅用训练数据设计采集算子的抽样分布 |

SRA 和 AFO 是核心贡献；OAAC 与 MPD 分别解决“采集后的外观覆盖”和“采集 profile 如何取样”。它们都不改变网络结构或推理路径。

## 3. 问题定义

设有标注体积为 \(\mathcal D_L=\{(X_i,Y_i)\}\)，未标注体积为 \(\mathcal D_U=\{X_j\}\)。训练网络为 Student \(f_\theta\)，Teacher \(f_\xi\) 由 EMA 更新：

\[
\xi\leftarrow \alpha\xi+(1-\alpha)\theta.
\]

模型仍是普通 2D U-Net。对于轴向位置 \(z\)，训练阶段读取连续三层 \(z-1,z,z+1\)，推理阶段只输入原始中心切片 \(x_z\)。

## 4. SRA：切片剖面重采集

### 4.1 合法 profile

对偏移 \(k\in\{-1,0,1\}\)，使用归一化高斯 slice profile：

\[
w_k(\sigma,\phi)=
\frac{\exp\left[-\frac{(k-\phi)^2}{2\sigma^2}\right]}
{\sum_{j=-1}^{1}\exp\left[-\frac{(j-\phi)^2}{2\sigma^2}\right]},
\qquad \sum_k w_k=1,\;w_k\ge 0.
\]

其中 \(\sigma\) 控制层间支持宽度，\(\phi\) 表示采集中心相对离散切片的位置偏移。当前支持范围固定为：

\[
\sigma\in[0.45,0.85],\qquad \phi\in[-0.25,0.25].
\]

### 4.2 图像重采集

三层图像使用同一组权重线性组合：

\[
\widetilde x_z=\mathcal A_w(X_z)
=\sum_{k=-1}^{1}w_kx_{z+k}.
\]

这不是逐像素随机 MixUp。一个样本内所有像素共享同一 slice profile，因此它模拟的是一次全切片采集响应。

## 5. AFO：采集对齐的 fractional occupancy

### 5.1 有标注分支

将每层离散标签转为 one-hot 组织占据，再应用与图像完全相同的权重：

\[
q_z^L=\sum_{k=-1}^{1}w_k\operatorname{onehot}(y_{z+k}^L).
\]

边界像素可得到例如 \([0.35,0.65]\) 的背景/前景占据，而不是被立即压回 0 或 1。它表达重采集像素中各组织的比例，不是标签平滑超参数。

### 5.2 未标注分支

Teacher 先分别预测三层，并保留已有的二维最大连通域约束：

\[
\widehat y_{z+k}^U=\operatorname{LCC}
\left(\arg\max f_\xi(x_{z+k}^U)\right).
\]

随后使用同一个 profile 构造未标注图像和伪占据目标：

\[
\widetilde x_z^U=\sum_{k=-1}^{1}w_kx_{z+k}^U,
\qquad
q_z^U=\sum_{k=-1}^{1}w_k\operatorname{onehot}(\widehat y_{z+k}^U).
\]

Teacher 的三层预测不是三个额外 Student 样本；它们只是生成一对 `re-acquired-U image / pseudo occupancy` 所需的目标来源。

### 5.3 软分割损失

对预测概率 \(p=\operatorname{softmax}(f_\theta(x))\) 和 occupancy \(q\)，使用：

\[
\mathcal L_{\mathrm{SCE}}(p,q)
=-\frac{1}{|\Omega|}\sum_{v\in\Omega}\sum_cq_{v,c}\log p_{v,c},
\]

\[
\mathcal L_{\mathrm{SDice}}(p,q)
=\frac{1}{C}\sum_c
\left(1-\frac{2\sum_vp_{v,c}q_{v,c}+\epsilon}
{\sum_vp_{v,c}^2+\sum_vq_{v,c}^2+\epsilon}\right),
\]

\[
\mathcal L_{\mathrm{occ}}
=\tfrac12(\mathcal L_{\mathrm{SCE}}+\mathcal L_{\mathrm{SDice}}).
\]

## 6. OAAC：有序采集-外观一致性

层间重采集会改变组织组成；亮度、对比度和 gamma 只改变外观。两者不能乱序处理。SliceEq-AC 先完成 SRA+AFO，使图像和目标语义一致，再仅对未标注 Student 图像施加坐标保持的单调强度变换：

\[
\widehat x_z^U=G_\eta(\widetilde x_z^U),
\qquad q_z^U\text{ 保持不变}.
\]

实现顺序为 gamma、contrast、brightness。三个变换均不改变空间坐标，不裁剪目标，也不接收 target 参数。当前 Strong 配置使用无量纲范围：

\[
\log\gamma\in[-0.25,0.25],\quad
\log c\in[-0.1875,0.1875],\quad
b/(x_{\max}-x_{\min})\in[-0.125,0.125].
\]

`1.25x` 是相对早期 OAAC 范围的固定开发设置，不应被描述为自动学习得到。论文必须补充强度敏感性，并把 MPD 与 OAAC 强度选择明确区分。

## 7. MPD：矩约束 profile 分布设计

### 7.1 为什么需要 profile 分布

均匀采样 \((\sigma,\phi)\) 默认每种合法 profile 同等重要，但不同 profile 在不同病例及轴向区域中产生的 fractional information 不同。MPD 不学习新网络，也不根据当前 loss、置信度或 test Dice 动态调权；它在训练开始前，仅利用 7 个有标注训练病例设计一个全局、冻结、病例无关的 profile 分布。

### 7.2 候选集合与 retained fractional information

将 \(\sigma\times\phi\) 离散为 \(21\times21=441\) 个候选 profile。对病例与归一化轴向三分区形成的 stratum \(s\)，在相邻层标签不一致的机会像素上定义：

\[
u_{s,g}=\mathbb E_{v\in\mathcal B_s}
\left[H(q_{g,v})
\mathbf 1\{\arg\max q_{g,v}=y_{z,v}\}\right].
\]

它奖励“产生非退化 fractional occupancy、但不翻转中心硬语义”的 profile。MPD 同时由三层图像的一阶/二阶轴向差分计算归一化图像残差，限制设计不能产生失控的图像扰动。

### 7.3 两阶段鲁棒设计

第一阶段最大化最差训练 stratum 的效用：

\[
t^*=\max_{q\in\mathcal Q}\min_s\sum_gq_gu_{s,g}.
\]

约束集合 \(\mathcal Q\) 保持：

- profile 相位镜像对称；
- 父分布的邻层质量及方向矩；
- 训练图像残差预算；
- 最低分布熵与单点密度上限。

第二阶段在保留至少 \(0.99t^*\) 效用的前提下，选择距离均匀父分布最近的解：

\[
q^*=\arg\min_{q\in\mathcal Q}
D_{\mathrm{KL}}(q\|p_0),
\quad
\text{s.t.}\;\min_s\sum_gq_gu_{s,g}\ge0.99t^*.
\]

因此 MPD 是保守的风险重分配，而不是寻找一个固定的 `[0.2, 0.6, 0.2]`。训练时仍然随机抽 profile，只是概率从均匀 \(p_0\) 变为冻结的 \(q^*\)。

## 8. 完整训练目标与 36-view

每个 loader batch 为 24 张中心切片：12 labeled + 12 unlabeled。自训练前 1000 iterations 只使用原始有标注监督。之后构造：

1. 12 个 `Native-L`：\((x_z^L,y_z^L)\)；
2. 12 个 `Re-acquired-L`：\((\widetilde x_z^L,q_z^L)\)；
3. 12 个 `OAAC-U`：\((G_\eta(\widetilde x_z^U),q_z^U)\)。

Student 因而一次前向接收 36 个配对视图。完整目标为：

\[
\mathcal L=
\tfrac12\left[
\mathcal L_{\mathrm{hard}}(x_z^L,y_z^L)
+\mathcal L_{\mathrm{occ}}(\widetilde x_z^L,q_z^L)
\right]
+\lambda(t)\mathcal L_{\mathrm{occ}}
\left(G_\eta(\widetilde x_z^U),q_z^U\right).
\]

其中 \(\lambda(t)\) 沿用父训练的 consistency ramp。Teacher 只由 EMA 更新，不参与梯度反传。

## 9. 推理

所有新增操作仅用于训练。测试时：

\[
\widehat y_z=\arg\max f_\theta(x_z).
\]

不读取相邻切片，不执行 SRA、OAAC 或 MPD，不增加网络参数和推理延迟。这个性质也是方法可迁移到其他 2D 分割骨干的关键。

## 10. 论文贡献表述

建议将贡献收束为三点：

1. **采集对齐的一致性定义。** 指出 through-plane 重采集同时改变图像观测和目标语义，并提出共享 slice profile 的成对图像/目标算子。
2. **半监督 fractional occupancy。** 将真实标签和 Teacher 三层伪标签统一映射为采集对应的软组织占据，以替代与新图像不一致的中心硬目标。
3. **训练数据限定的鲁棒采集风险。** 在物理可解释 profile 家族内，以矩、图像残差、熵和密度约束设计冻结分布，同时保持普通单切片推理。

OAAC 应作为完整目标中的外观域项，而不是单独宣称第四个核心创新。

## 11. 公平消融：每一行只回答一个问题

| 行 | 代码 stage | 36-view | 图像 SRA | target 同 profile | fractional target | OAAC | MPD | 回答的问题 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | `baseline` | 否 | 否 | 否 | 否 | 否 | 否 | 原始 24-view Mean Teacher 水平 |
| C0 | `baseline_36` | 是 | 否 | 否 | 否 | 否 | 否 | 额外视图、计算量和 BatchNorm 是否本身涨点 |
| C1 | `image_only_36` | 是 | 是 | 否 | 否 | 否 | 否 | 只改变图像、仍用中心 target 是否产生语义错配 |
| C2 | `hard_targets` | 是 | 是 | 是 | 否 | 否 | 否 | target 同步但硬化后，是否仍丢失部分体积信息 |
| C3 | `full` | 是 | 是 | 是 | 是 | 否 | 否 | SRA+AFO 这一核心定义是否有效 |
| C4 | `oaac_strong` | 是 | 是 | 是 | 是 | 是 | 否 | 目标对齐后，采集后外观覆盖是否进一步有效 |
| C5 | `mpd` | 是 | 是 | 是 | 是 | 是 | 是 | profile 风险设计是否优于均匀父分布 |

`occ_l_only` 与 `occ_u_only` 是 AFO 的分支归因实验，不属于累计主表：

| 行 | 代码 stage | 作用 |
|---|---|---|
| F10 | `occ_l_only` | 只保留 labeled fractional occupancy |
| F01 | `occ_u_only` | 只保留 unlabeled pseudo fractional occupancy |

最关键的差值解释：

- `C0-B0`：36-view/计算量影响；
- `C1-C0`：只有图像 SRA 的影响；
- `C2-C1`：同步 target 的影响；
- `C3-C2`：保留 fractional occupancy 的影响；
- `C4-C3`：采集后外观覆盖的影响；
- `C5-C4`：训练限定 profile 分布设计的影响。

## 12. 论文实验表应如何组织

### Table 1：主要结果

- PROMISE12 35/5/10，7-label；
- 第二数据集建议使用具有连续 3D MRI 切片的任务；
- Mean Teacher、UniMatch、最接近的半监督医学分割方法；
- Dice、Jaccard、HD95、ASD；
- 所有方法统一验证集选择 checkpoint，test 只评一次。

### Table 2：标注效率与泛化

- 7-label 与 11-label；
- 至少一个不同骨干或不同 SSL scaffold；
- 同一 SliceEq-AC 训练算子迁移，推理网络保持不变。

### Table 3：公平累计消融

报告 C0-C5；B0 单列为原始计算量未匹配参考。F10/F01 放在独立的分支归因小表中，避免把 factorial control 伪装成累计组件。

### Table 4：参数与设计稳定性

- OAAC 强度敏感性；
- MPD 网格 11/21/31；
- 轴向分箱 2/3/4；
- moment/image-residual/entropy/density 约束移除；
- 7 个 labeled case 的 leave-one-patient-out 设计稳定性。

这些实验由 `code/analyze_sliceeq_mpd_robustness.py` 离线完成，不读取 validation/test 或模型 Dice。

### Figure 1：方法图

使用三条训练路径：Native-L、Re-acquired-L、OAAC-U；右侧统一为 36-view Student 前向和 EMA Teacher；底部强调推理仍为原始单切片 2D U-Net。

### Figure 2：机制图

由 `code/visualize_sliceeq_reacquisition.py` 生成：

- 三层图像按同一 profile 合成为 re-acquired image；
- 三层标签按同一 profile 合成为 fractional occupancy；
- 展示硬中心标签无法表达、但 occupancy 能保留的边界部分体积。

### Figure 3：机制分析

建议按目标面积和归一化轴向位置报告 slice-wise Dice，并单独统计前景边界带。这个分析用于检验收益是否集中在小目标及体积两端切片，不能用于重新选择 checkpoint。

## 13. 参数来源与审稿风险

| 参数 | 类型 | 论文中应如何描述 | 必须补什么证据 |
|---|---|---|---|
| 三层支持 `[-1,0,1]` | 结构选择 | 与当前 2D+邻层 H5 数据相容的最小对称支持 | 更大半径敏感性或限制说明 |
| `sigma/phase` 范围 | 合法 profile 家族 | 归一化 slice-unit 中的训练采集支持 | MPD 网格/范围稳定性 |
| OAAC `1.25x` | 开发设置 | 固定、无量纲外观范围，不是学习参数 | 预先列出的强度敏感性 |
| MPD 2%/5%/70%/3x/99% | 鲁棒设计安全约束 | 防止 moments、图像扰动和分布支持塌缩 | tolerance sweep、constraint removal、LOPO |
| loss 中 `0.5` | 等风险平均 | Native-L 与 Re-acquired-L 的算术平均 | 不应包装成学习权重 |

没有 DICOM/NIfTI 采集元数据时，不能声称恢复了真实 scanner slice profile。准确表述应是：**在一个可解释、受约束的 through-plane profile 家族中进行 acquisition-aware training**。

## 14. 当前结果的合规表述

当前可用于方法开发叙述的关键结果为：

| 方法 | checkpoint 规则 | Test Dice | 解释 |
|---|---|---:|---|
| OAAC-Strong 1.25 | validation-selected | 0.851960 | 当前更干净的单次正向结果 |
| MPD | validation-selected | 0.848952 | 未稳定超过 OAAC-Strong |
| MPD | 指定 iter29000 | 0.854573 | 当前最高开发观察，不能称为独立无偏 test 主结果 |

因此论文在新增独立数据验证前，应写：MPD 显示了达到 `0.854573` 的开发潜力，但严格 validation-selected 结果尚未证明其稳定优于 OAAC-Strong。不能只报告指定 test checkpoint。

## 15. 完整论文实验判定标准

达到可投稿状态至少需要：

1. 完成 C0-C5 公平消融，并使用同一 validation checkpoint 规则；
2. 在第二个连续 3D MRI 数据集复现主趋势；
3. 补 7-label/11-label 或另一标注比例；
4. 报告至少 3 次独立训练的均值和标准差；若固定同一 seed 重跑，只能称 implementation repeat，不能称随机种子鲁棒性；
5. 完成 OAAC sensitivity 与 MPD 离线稳定性审计；
6. 报告逐病例、体积端部/中部、小/中/大前景及边界分析；
7. 报告训练时间、显存和推理开销；
8. 冻结 PROMISE12 后再做外部验证，避免继续用其 test 指导方法。

## 16. 一段可直接用于汇报的 Method

> 我们研究的是半监督 MRI 分割中的采集-目标错配。普通一致性训练默认增强只改变外观，因此增强图像仍使用中心切片标签；但 through-plane 重采集会混合相邻层组织，使图像语义本身发生变化。SliceEq-AC 用同一个 Gaussian slice profile 同时重采集连续三层图像和 one-hot 标签，在有标注分支得到 exact fractional occupancy，在未标注分支由 EMA Teacher 的三层伪标签得到 pseudo fractional occupancy。完成语义对齐后，我们只对未标注 Student 图像施加坐标保持的单调外观扰动。最后，MPD 仅使用有标注训练病例，在保持 profile moments、图像扰动预算、分布熵和相位对称的条件下，设计一个全局冻结的 profile 分布。所有操作只在训练时使用，测试仍是普通单切片 2D U-Net。

## 17. 代码对应

| 内容 | 文件 |
|---|---|
| SRA | `code/utils/sliceeq.py` |
| AFO 与 soft loss | `code/utils/sliceeq_occ.py` |
| OAAC | `code/utils/sliceeq_oaac_strong.py` |
| MPD 设计与采样 | `code/utils/sliceeq_mpd.py` |
| 公平消融训练 | `code/train_sliceeq_occ_ablation.py` |
| OAAC portable 训练 | `code/train_sliceeq_occ_oaac_strong_portable.py` |
| MPD portable 训练 | `code/train_sliceeq_occ_oaac_strong_mpd_portable.py` |
| 三切片合成示意图 | `code/visualize_sliceeq_reacquisition.py` |
| MPD 离线审计 | `code/analyze_sliceeq_mpd_robustness.py` |
| 连续论文实验 | `run_coda_final_paper_suite.sh` |

