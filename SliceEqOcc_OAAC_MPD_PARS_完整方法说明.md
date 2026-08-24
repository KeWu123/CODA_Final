# SliceEqOcc-OAAC-Strong-MPD/PARS 完整方法说明

> 整理对象：`CoDA-MPD-Final` 当前代码、历史实验记录、MPD 结果清单与 PARS 冻结协议  
> 整理日期：2026-08-23  
> 当前结论：**已有完整结果的最终版本是 SliceEqOcc-OAAC-Strong-MPD；PARS 是在其上仅改变训练切片采样分布的后继实验，代码已完成，但尚无正式 30k 结果。**

---

## 0. 先把文件名和方法关系说清楚

用户给出的路径中，`...mpd_pars.pytest...mpd` 实际上是两个文件名连在了一起：

1. `code/test_sliceeq_occ_oaac_strong_mpd.py`
   - 测试已经完成训练的 **MPD** 权重。
2. `code/test_sliceeq_occ_oaac_strong_mpd_pars.py`
   - 测试将来训练得到的 **MPD+PARS** 权重。

它们都是**测试入口**，并不实现训练方法。真正的方法分别位于：

- `code/train_sliceeq_occ_oaac_strong_mpd.py`
- `code/utils/sliceeq_mpd.py`
- `code/train_sliceeq_occ_oaac_strong_mpd_pars.py`
- `code/utils/sliceeq_pars.py`

两个测试入口的代码几乎完全相同，主要区别只是默认实验名不同。测试时，它们都建立同一个 2D U-Net，严格加载指定权重，然后在 10 个测试病例上逐体积计算 Dice、Jaccard、HD95 和 ASD。

因此一定要区分：

> **MPD/PARS 改变的是训练过程；测试过程仍然只是原始单切片 2D U-Net 推理。**

---

## 1. 一句话概括整个研究

我们不是简单把相邻 MRI 切片混在一起做增强，而是把 MRI 的层间采集过程视为一个会同时改变**图像观测**和**组织占据比例**的算子：训练时用同一切片剖面权重成对重建图像与目标，再用 MPD 从有标注训练数据中设计更合理的剖面采样分布；PARS 则进一步尝试让病例和轴向位置获得更公平的训练机会，而推理仍保持普通 2D U-Net。

更短的论文式表述是：

> **SliceEqOcc-OAAC-Strong-MPD learns acquisition-aligned fractional occupancy under a training-only, moment-constrained through-plane profile distribution, without changing the inference network.**

---

## 2. 最终方法到底由什么组成

当前有结果的完整方法可以写成：

```text
EMA Mean Teacher 基座
        +
成对切片剖面重采集（Slice-profile Re-acquisition）
        +
分数占据目标（Fractional Occupancy）
        +
有序采集-外观一致性（OAAC-Strong 1.25）
        +
矩约束剖面分布设计（MPD）
```

PARS 是候选后继版本：

```text
最终 MPD 方法
        +
训练后 1000 iter 的病例均衡轴向采样（PARS）
```

为了汇报时容易讲，可以把核心部分简称为：

| 汇报简称 | 建议全称 | 实际含义 |
|---|---|---|
| SRA | Slice-profile Re-Acquisition | 用同一三抽头剖面重建相邻层图像和目标 |
| AFO | Acquisition-aligned Fractional Occupancy | 不把混合目标重新硬化，而保留每个像素的前景占据概率 |
| EOA | Exact Occupancy Anchor | 有标注重采集分支使用真实相邻标签形成精确占据目标，同时保留原始中心切片硬标签锚点 |
| OAAC | Ordered Acquisition-Appearance Consistency | 先完成会改变目标的采集变换，再只改变 U 图像外观 |
| MPD | Moment Profile Design | 在物理剖面支持和矩约束下设计全局 profile 概率分布 |
| PARS | Patient-Axial Acquisition-Risk Sampling | 在病例均衡前提下设计统一的轴向三段采样概率 |

说明：SRA/AFO/EOA 是适合汇报的功能名称；代码中真正的主方法名仍是 `SliceEqOcc-OAAC-Strong-MPD`。

---

## 3. 为什么会走到这一步：完整研究演化

### 3.1 基座：去除 Copy-Paste 的 EMA Mean Teacher

本项目基座来自 BCP 风格代码，但去除了 Copy-Paste，仅保留：

- 2D U-Net Student；
- EMA Teacher；
- 有标注监督损失；
- Teacher 对无标注图像生成伪标签；
- Student 使用伪标签进行一致性训练。

它不能被称为“完整原始 BCP”，更准确的名字是：

> **BCP-derived no-Copy-Paste EMA baseline**

当前固定协议：

| 项目 | 配置 |
|---|---|
| 数据集 | PROMISE12 |
| 划分 | 35 train / 5 val / 10 test |
| 标注预算 | train 前 7 个完整病例，共 191 slices |
| 网络 | 2D U-Net，二分类 |
| 输入大小 | 256 × 256 |
| Pretrain | 10000 iterations，仅标注数据 |
| Self-train | 30000 iterations，其中前 1000 iter 是监督 warm-up |
| Loader batch | 24 = 12 labeled + 12 unlabeled |
| 优化器 | SGD，LR 0.01，momentum 0.9，weight decay 0.0001 |
| EMA | decay 0.99，训练期 teacher 为 train mode |
| Seed | 1337 |
| 验证 | 每 200 iter |
| 周期权重 | 每 1000 iter 保存 raw Student |

用户此前确认的 baseline Dice 是 `0.804387`。仓库较早的研究记录只把旧 baseline 记作约 `0.78--0.80`，因此正式论文中应以重新统一复现的结果为准，不要混用两个口径。

### 3.2 早期探索为什么没有成为最终方法

| 路线 | 核心思路 | Test Dice | 结论与经验 |
|---|---|---:|---|
| CoDA | 根据图像破坏程度软化伪目标 | 0.819876 | 有探索性增益，但强视图、目标表达、损失和耦合同时变化，难做因果归因 |
| BMER | 重合成边界证据 | 0.795949 | 9/10 病例相对 CoDA 下降，说明局部边界重合成容易引入错位或分布偏差 |
| OBA | 成对反向增强抵消增强漂移 | 0.818872 | 与 CoDA 基本相同，后期验证明显退化，说明降低增强均值漂移不是主要瓶颈 |
| SliceEq | 相邻三层图像和目标成对重采集，但最终仍硬化目标 | 0.832603 | 首次明显证明“采集过程和目标必须一起变”是有价值方向 |
| SliceEqOcc | 保留 fractional occupancy | 0.827368（val-selected）；0.844566（iter23k 开发观察） | 核心机制成立，但小验证集无法稳定选到 test 最优 checkpoint |
| SliceEqOcc-SC | 同病例共享 scan-level profile | 0.836219 | 降低了样本级 profile 多样性，未超过 SliceEqOcc |
| APTNA | 新的目标/不确定性处理 | 0.829420 | 目标侧附加处理没有带来稳定收益 |
| ADU | disagreement 权重 | 0.843335 | 接近但不超过 SliceEqOcc，表明额外伪标签可靠性权重不是主要增益来源 |
| SCPO | 3D slab 连通伪标签 | 0.842378 | 伪标签拓扑处理基本中性 |
| OAAC 1.00 | 采集后对 U 图像做有序外观增强 | 0.849538（test-selected 开发观察） | 说明目标正确形成后，扩大 U 外观覆盖是有效的，但该 checkpoint 由 test 搜索得到 |
| OAAC-Strong 1.25 | 联合放大 gamma/contrast/brightness 范围 | 0.851960（val-selected） | 验证选择下的可靠正向结果，成为 MPD 的直接父方法 |
| OAAC 1.50 | 继续增强外观强度 | 0.852059（val-selected） | Dice 只高 0.000099，但验证、更完整表面指标和病例胜率更差，因此不替代 1.25 |
| ARCP | 按每个图像栈的轴向响应调 profile | 0.851062（val-selected） | 非平凡但没有超过 Strong，说明“逐栈统一残差强度”不是正确目标 |
| MPD | 训练集 exact occupancy 驱动的全局 profile 分布设计 | 0.854573（iter29k 开发观察） | 当前最高开发结果；但不是验证集选中的 checkpoint |
| PARS | 病例均衡 + 全局轴向三段机会分布 | 尚无 | 已实现，等待一次冻结的 30k 运行 |

从这些实验得到的最重要经验不是“模块越多越好”，而是：

1. 真正有效的主线是**采集变化与目标变化必须成对建模**。
2. 把 fractional occupancy 再硬化，会丢掉重采集产生的部分容积信息。
3. 在目标正确之后，适度增加 U 图像外观覆盖有效，但更强并不一定更好。
4. 后续最值得优化的是同一 acquisition-risk 的**分布**，而不是继续添加伪标签阈值、边界 loss 或网络分支。
5. PROMISE12 只有 5 个 val 病例，validation 与 test checkpoint 排名不一致是当前最严重的实验问题之一。

---

## 4. 核心动机：为什么普通增强在这里会错

### 4.1 普通增强的隐含假设

多数半监督分割方法采用：

```text
Teacher 在原图/弱增强图上产生伪标签
                ↓
Student 在强增强图上学习同一个伪标签
```

这隐含假设是：增强只改变图像外观，不改变该像素对应的组织语义。

对 gamma、brightness 等坐标不变的外观变换，这个假设通常成立；但对 MRI 层间重采集并不成立。

### 4.2 MRI 切片不是无限薄平面

一张 MRI 切片可以包含有限厚度范围内的信号。用离散三层近似时，新的虚拟切片可写成：

\[
\widetilde X_z=w_{-1}X_{z-1}+w_0X_z+w_{+1}X_{z+1},
\qquad \sum_k w_k=1,\quad w_k\ge 0.
\]

这里每个输出像素都按相同权重混合三张切片的对应像素。例如某一点：

```text
前一层像素 = 0.1
中心层像素 = 0.7
后一层像素 = 0.4
权重        = [0.2, 0.6, 0.2]

新像素 = 0.2×0.1 + 0.6×0.7 + 0.2×0.4 = 0.52
```

问题是：图像已经包含三层信号，如果仍使用中心层硬标签 \(Y_z\)，图像和答案不再严格配对。

### 4.3 正确做法：同一个算子同时作用于图像和目标

把二分类标签写成 one-hot occupancy \(Q\)，则：

\[
\widetilde Q_z=w_{-1}Q_{z-1}+w_0Q_z+w_{+1}Q_{z+1}.
\]

如果某像素在三个标签中分别为前景、前景、背景，且权重为 `[0.2,0.6,0.2]`，则该位置的前景占据率是：

\[
q_{fg}=0.2\times1+0.6\times1+0.2\times0=0.8.
\]

这不是含糊的错误标签，而是“该虚拟切片对应的采集厚度中约 80% 为前景组织”的软监督。

这就是 SliceEqOcc 的核心：

> **图像如何被重采集，目标就以完全相同的权重表示组织占据率。**

---

## 5. 三抽头 slice profile 是怎么来的

方法不是固定使用 `[0.2,0.6,0.2]`。这个比例只是一个便于解释的示例。

实际 profile 由高斯剖面的宽度 \(\sigma\) 和相位 \(\phi\) 生成：

\[
\widetilde w_k=
\exp\left[-\frac12\left(\frac{k-\phi}{\sigma}\right)^2\right],
\qquad k\in\{-1,0,+1\},
\]

\[
w_k=\frac{\widetilde w_k}{\sum_j\widetilde w_j}.
\]

父方法的采样范围为：

\[
\sigma\in[0.45,0.85],\qquad
\phi\in[-0.25,0.25].
\]

- \(\sigma\) 越大，相邻切片信号比例通常越高；
- \(\phi>0\) 表示虚拟采集位置向 \(z+1\) 偏；
- \(\phi<0\) 表示向 \(z-1\) 偏；
- \(\phi=0\) 时剖面左右对称。

`[0.2,0.6,0.2]` 大约对应 `phase=0, sigma≈0.6746` 的一个剖面点，而不是固定训练参数。

---

## 6. 三条训练路径如何组成 36-view Student batch

### 6.1 前 1000 self-train iterations

前 1000 iter 是监督 warm-up：

- 使用原始中心切片；
- 只计算有标注 hard CE + Dice；
- 不启用 SliceEq；
- 不启用 fractional occupancy；
- 不启用 OAAC；
- 不计算无标注一致性损失。

它的作用是让从 Pre10000 加载的 Student/Teacher/optimizer 在新的 self-train 阶段稳定进入训练状态。它包含在 30000 self-train iterations 内，不是额外再加 1000。

### 6.2 iter 1000 之后

Loader 仍然只取 24 个样本：

```text
12 labeled center slices + 12 unlabeled center slices
```

但每个有标注样本同时形成原始视图和重采集视图，所以 Student 一次前向输入：

```text
12 × Native-L
12 × Re-acquired-L
12 × Re-acquired-U + OAAC
= 36 views
```

#### 路径 A：Native-L 硬语义锚点

\[
(X_z^L,Y_z^L)
\]

保留原始中心切片与硬真实标签，防止训练完全漂向虚拟采集分布。

#### 路径 B：Re-acquired-L 精确分数占据

读取同一病例的 \(z-1,z,z+1\) 图像和真实标签，并用同一组 profile 权重形成：

\[
(\widetilde X^L, q^L).
\]

这是“Exact Occupancy Anchor”：它为模型提供没有伪标签噪声的采集对齐软监督。

#### 路径 C：Re-acquired-U 伪分数占据

无标注病例没有真实标签，因此先由 EMA Teacher 分割三张相邻切片：

\[
X^U_{z-1},X^U_z,X^U_{z+1}
\xrightarrow{Teacher}
\widehat Y^U_{z-1},\widehat Y^U_z,\widehat Y^U_{z+1}.
\]

Teacher 输出经 argmax 和逐切片 2D 最大连通域处理。随后，同一组 profile 权重同时混合图像栈和伪标签栈：

\[
\widetilde X^U=\sum_k w_kX^U_{z+k},
\qquad
q^U=\sum_k w_k\widehat Q^U_{z+k}.
\]

每个无标注样本最后仍只产生**一个**重采集 Student 视图，而不是三张 Student 输入。三张相邻切片只用于 Teacher 构造对应的 occupancy。

### 6.3 Teacher 与 Student 的前向数量

- Teacher：对 `12 U × 3 neighbor slices = 36` 张 2D 图像产生伪掩码；
- Student：对 `12 Native-L + 12 Re-acquired-L + 12 Re-acquired-U = 36` 张图像做一次拼接前向；
- 推理：每次仍输入一张原始 2D 切片。

因此所谓 `student36` 是 36 个 Student views，不是 loader 的 `batch_size=36`。命令行 batch 仍固定为 24。

---

## 7. Fractional occupancy 如何计算损失

### 7.1 原始有标注硬损失

\[
\mathcal L_{hard}
=\frac12\left(\mathcal L_{CE}^{hard}+\mathcal L_{Dice}^{hard}\right).
\]

### 7.2 重采集分支的 soft CE

设 Student 概率为 \(p_{ic}\)，occupancy 目标为 \(q_{ic}\)：

\[
\mathcal L_{CE}^{soft}
=-\frac1N\sum_{i,c}q_{ic}\log p_{ic}.
\]

### 7.3 squared soft Dice

\[
\mathcal L_{Dice}^{soft}
=1-\frac1C\sum_c
\frac{2\sum_i p_{ic}q_{ic}+\epsilon}
{\sum_i p_{ic}^2+\sum_iq_{ic}^2+\epsilon}.
\]

\[
\mathcal L_{soft}
=\frac12\left(\mathcal L_{CE}^{soft}+\mathcal L_{Dice}^{soft}\right).
\]

### 7.4 总损失

标注分支先平均原始硬监督和重采集精确 occupancy 监督：

\[
\mathcal L_L
=\frac12\left[
\mathcal L_{hard}(f_\theta(X_z^L),Y_z^L)
+\mathcal L_{soft}(f_\theta(\widetilde X^L),q^L)
\right].
\]

无标注分支为：

\[
\mathcal L_U
=\mathcal L_{soft}(f_\theta(\widehat X^U),q^U),
\]

其中 \(\widehat X^U\) 是经过 OAAC 的重采集无标注图像。

总目标：

\[
\mathcal L=\mathcal L_L+\lambda(t)\mathcal L_U.
\]

这里 \(\lambda(t)\) 是原有 consistency ramp。`0.5 × (original + reacquired)` 是两个有标注风险项的等权平均，不是通过 test 调出来的技巧。

---

## 8. OAAC-Strong 1.25 到底做了什么

OAAC 全称是 **Ordered Acquisition-Appearance Consistency**。

它的重点不是发明 gamma、contrast、brightness，而是固定正确顺序：

```text
先：A_h 同时改变图像和 occupancy，形成正确配对
再：G_eta 只改变 U Student 图像外观，occupancy 不再改变
```

即：

\[
(X^U,Q^U)\xrightarrow{A_h}(\widetilde X^U,q^U)
\xrightarrow{G_\eta}(G_\eta(\widetilde X^U),q^U).
\]

OAAC 依次执行：

1. gamma；
2. contrast；
3. brightness。

Strong 1.25 的冻结范围为：

| 变换 | 范围 |
|---|---|
| log-gamma | `[-0.25, 0.25]` |
| log-contrast | `[-0.1875, 0.1875]` |
| brightness / image span | `[-0.125, 0.125]` |
| 应用概率 | 1.0 |

这些变换不改变空间坐标，gamma 和 contrast 系数保持为正，因此目标 \(q^U\) 可以不变。

代码还使用独立 appearance RNG，避免 OAAC 消耗 profile、Student/Teacher dropout 或全局随机数流。

需要注意：36 个视图在一次共享 BatchNorm 的 Student 前向中计算，所以 OAAC 强 U 图像会影响整个 batch 的 BN 统计。可以说“L 输入、GT 和损失定义不变”，但不能说“L 分支内部 activation 完全不受 OAAC 影响”。

---

## 9. MPD：为什么不是继续调一个固定权重

### 9.1 MPD 解决什么问题

SliceEqOcc 父方法在 \((\sigma,\phi)\) 范围内近似均匀随机采样 profile。它确保了随机性，却没有保证：

- 每个有标注病例都能获得足够的 fractional occupancy 信息；
- 轴向不同位置都能获得相对均衡的 acquisition opportunity；
- 采样分布不会偏向“产生大量变化但语义已经翻转”的 profile。

MPD 不增加网络模块，也不把 profile 固定成另一组 `[a,b,c]`。它优化的是：

> **441 个合法 profile 在训练中分别应被采到多少次。**

### 9.2 候选空间

在原始支持内使用 `21 × 21 = 441` 个 midpoint 网格点：

\[
(\sigma_g,\phi_g),\qquad g=1,\ldots,441.
\]

父分布为 \(p_0\)，MPD 设计全局离散分布 \(q\)。训练时每个样本仍随机抽一个 profile，只是抽样概率从 \(p_0\) 改成 \(q\)。

### 9.3 profile 的矩解释

定义总邻层质量与方向偏置：

\[
b=w_{-1}+w_{+1}=1-w_0,
\]

\[
r=\frac{w_{+1}-w_{-1}}{b}.
\]

图像重采集残差可写成：

\[
\widetilde X-X_z=(br)g_1+\frac b2g_2,
\]

其中：

\[
g_1=\frac{X_{z+1}-X_{z-1}}2,
\qquad
g_2=X_{z-1}-2X_z+X_{z+1}.
\]

- \(b\)：混入多少相邻层信号；
- \(br\)：向前/向后偏移的强度；
- \(b^2\) 与 \((br)^2\)：控制总体混合和方向扰动的二阶规模。

因此 MPD 约束：

\[
E_q[b],\quad E_q[b^2],\quad E_q[(br)^2]
\]

保持在父分布对应矩的 ±2% 内。这样改变的是 profile 的风险分配，而不是偷偷把增强整体变强或变弱。

### 9.4 RFI：Retained Fractional Information

MPD 只使用前 7 个有标注训练病例的 exact labels。对每个 profile，它关注：

1. 相邻层标签确实不同，因此该位置存在层间组织变化机会；
2. 加权 occupancy 的 argmax 仍等于中心层语义，避免把样本直接变成另一个切片的标签；
3. 在前两项成立时，occupancy 仍保留多少非零熵，即多少有意义的部分容积信息。

可写成：

\[
U_{s,g}=
\frac{
\sum_v
\mathbf1[\text{neighbor differs}]
\mathbf1[\arg\max Q_g(v)=Y_z(v)]
H(Q_g(v))
}{
\sum_v\mathbf1[\text{neighbor differs}]+\epsilon
}.
\]

其中 \(s\) 是 patient × axial index-third stratum，\(g\) 是 profile。

直观上，RFI 奖励的是：

> **既不推翻中心切片主要语义，又能产生有信息量的 fractional occupancy。**

### 9.5 21 个 strata

7 个有标注病例分别按切片索引排序后分成 3 个等秩区间：

\[
7\text{ patients}\times3\text{ index thirds}=21\text{ strata}.
\]

这里必须称为 `index thirds` 或“轴向索引三等分”，不能直接称为 apex/mid/base。没有解剖标注或物理方向元数据时，把三段命名成 apex/base 会形成过度主张。

### 9.6 两阶段 SLSQP

第一阶段最大化最差 stratum 的期望 RFI：

\[
\max_q\min_s\sum_gq_gU_{s,g}.
\]

第二阶段在保留第一阶段 99% 最优值的条件下，寻找最接近父分布的解：

\[
\min_qD_{KL}(q\Vert p_0).
\]

约束包括：

- phase 镜像对称；
- 三个关键 profile moments 保持在父分布 ±2%；
- 每个 patient-stratum 的归一化图像 RMS residual 在父预算 ±5%；
- `q/p0 ≤ 3`；
- `H(q) ≥ 0.70 H(p0)`；
- 所有概率非负且和为 1。

这些约束的意义是：MPD 只能在原始方法附近重新分配概率，不能退化为只选择少量极端 profile。

MPD 的原始协议还设计了 leave-one-patient-out（LOPO）门槛：每次用 6 个有标注病例设计分布，在第 7 个病例上检查最差 stratum 的 RFI 是否改善。但 2026-08-18 用户明确要求跳过预筛选，直接使用全部 7 个有标注训练病例设计一次全局分布并完成 30k 训练。因此当前 MPD 必须标记为：

> `exploratory_user_override_without_lopo_gate`

这不表示实现错误，但意味着“设计分布对未见病例是否泛化”还没有由 LOPO 或外部数据证明。

### 9.7 实际冻结设计

当前 MPD artifact 的关键统计：

| 统计项 | 数值 |
|---|---:|
| 候选 profile | 441 |
| Active RFI strata | 20 / 21 |
| 结构性空 stratum | `Case08:index-third-2` |
| Worst designed RFI | 0.47033756 |
| 设计熵 / 父分布熵 | 6.01794513 / 6.08904488 |
| 熵保留率 | 98.83% |
| 最大 density ratio | 1.607772 |
| Phase mirror error | 0 |
| `E(b)` | 0.38257712 |
| `E(b²)` | 0.15254627 |
| `E((br)²)` | 0.01464702 |
| 近似平均 `[w-,w0,w+]` | `[0.1913, 0.6174, 0.1913]` |

父分布的对应矩为：

```text
E(b)       = 0.375076
E(b²)      = 0.149946
E((br)²)   = 0.014946
mean w     ≈ [0.1875, 0.6249, 0.1875]
```

这表明 MPD 没有把方法变成固定 `[0.2,0.6,0.2]`，也没有大幅改变平均增强强度；它主要调整了不同 profile 的出现频率。

### 9.8 MPD 在代码里只替换一个点

训练入口最终只注入：

```python
strong.parent.sample_slice_profiles = sampler
```

网络、loss、EMA、OAAC、batch、验证和测试逻辑全部继承父方法。因此 MPD 相对 OAAC-Strong 的差别有较清楚的归因边界。

---

## 10. PARS：它与 MPD 有什么区别

### 10.1 MPD 与 PARS 优化的是两个不同随机变量

| 方法 | 回答的问题 | 改变什么 |
|---|---|---|
| MPD | 当前样本应该使用哪一种 through-plane profile？ | 441 个 profile 的概率 \(q_g\) |
| PARS | 下一次训练应该从哪个病例、哪个轴向索引段取 slice？ | 病例和 3 个 index-thirds 的采样过程 |

MPD 设计的是**算子分布**；PARS 设计的是**数据支持分布**。

### 10.2 PARS 的动机

原 `TwoStreamBatchSampler` 在切片索引集合上采样，因此：

- 切片较多的病例自然获得更多更新；
- 某些切片密集的轴向区域可能主导 SGD；
- 这与 MPD 的 patient × index-third 最坏风险设计并不完全对齐。

PARS 的目标不是寻找 loss 最大的难例，而是让训练机会与 acquisition-derived fractional occupancy opportunity 对齐。

### 10.3 PARS 使用哪些数据

PARS 设计阶段只使用：

- `train_slices.list` 中的病例名和切片索引；
- 前 191 个有标注训练切片；
- MPD 在 exact occupancy 上得到的训练机会统计。

它明确不使用：

- 无标注标签；
- validation/test；
- 模型预测、confidence、uncertainty；
- 当前 loss 或历史 Dice。

因此 PARS 不是 online hard mining，也不是伪标签置信度采样。

### 10.4 PARS 的 opportunity

MPD 的 utility 只在存在层间标签变化的 active slices 上平均。为了对应运行时“抽到任意一张 slice”的机会，PARS 将其乘以 active slice 比例：

\[
u_{p,t}
=\overline{U}^{active}_{p,t}
\times
\frac{N^{active}_{p,t}}{N_{p,t}}.
\]

然后按每个病例在父轴向分布下的期望机会归一化：

\[
\bar u_{p,t}
=\frac{u_{p,t}}
{\sum_{j=1}^{3}p_j u_{p,j}}.
\]

这样避免某个天然有更多边界变化的病例在优化中完全压过其他病例。

### 10.5 PARS 的两阶段设计

PARS 只设计一个所有病例共享的三维概率向量：

\[
q=(q_1,q_2,q_3).
\]

第一阶段最大化最差 active patient-third exposure：

\[
\max_q\min_{p,t}q_t\bar u_{p,t}.
\]

第二阶段保留 99% 最优值，并最小化与父轴向分布的 KL：

\[
\min_qD_{KL}(q\Vert p_z).
\]

约束：

- \(q_t\ge0\)，且 \(\sum_tq_t=1\)；
- `q_t / p_t ≤ 1.50`；
- `H(q) ≥ 0.90 H(p)`；
- 三个 index thirds 均必须 active。

### 10.6 PARS 运行时怎么取样

L 和 U stream 分别执行：

1. 用随机排列轮流访问病例，使病例计数最多只差 1；
2. 根据冻结的同一个三段概率 \(q\) 选择 index third；
3. 在该病例的该 third 内均匀抽一张 slice；
4. 合成原来的 `12L + 12U` batch。

PARS 使用私有 NumPy generator，seed 为 `1341`，不会推进全局 NumPy、Torch 或 CUDA RNG。

Sampler 的 epoch 长度仍锁定为：

\[
\left\lfloor\frac{191}{12}\right\rfloor=15\text{ batches},
\]

所以 validation、iteration 和 checkpoint 的时间尺度不因 PARS 改变。

### 10.7 为什么前 1000 iter 不启用 PARS

`iter 0--999` 是父方法的纯监督 warm-up。PARS 保留原 `TwoStreamBatchSampler`，从第一个 acquisition-active iteration 才替换采样。

这样能确保 PARS 的唯一作用发生在真正启用 SliceEqOcc/MPD 的阶段，而不是通过改变 warm-up 轨迹引入额外混杂。

### 10.8 PARS 在代码里只替换第二个点

PARS 训练入口注入：

```python
mpd.strong.parent.sample_slice_profiles = profile_sampler
mpd.strong.parent.TwoStreamBatchSampler = sampler_factory
```

第一行继承冻结 MPD，第二行才是 PARS 的唯一新增变化。

保持不变的内容包括：

- U-Net；
- shared Pre10000；
- seed 1337；
- OAAC-Strong 1.25；
- SGD/LR 0.01；
- EMA train mode；
- Teacher hard argmax + 2D LCC；
- exact-L / pseudo-U fractional occupancy；
- soft CE + squared Dice；
- consistency ramp；
- 12L + 12U loader、Student 36 views；
- 30000 self-train；
- 每 200 iter validation；
- 每 1000 iter checkpoint；
- 2D 单切片 inference。

### 10.9 PARS 当前状态

截至本文件整理时：

- 代码已实现；
- 本地静态与数值测试已通过；
- 仍需训练机 CUDA startup smoke；
- 正式 seed1337、30k 训练尚未完成；
- 因此没有 PARS Dice，不能用 MPD 的 `0.854573` 冒充 PARS 结果。

PARS 启动时会重新生成并校验两个冻结 artifact：

- `mpd_profile_design.json`；
- `pars_sampling_design.json`。

如果数据边界、源文件 hash、优化约束、采样器可重复性或 CUDA smoke 任一失败，程序应在正式训练前终止。

冻结停止规则是：只跑一次。如果最高测试 checkpoint 不超过 MPD 的 `0.854573`，则关闭 PARS，不继续调 thirds、density cap、entropy floor、机会公式或 sampler seed。

---

## 11. 完整前向过程

```text
                         TRAINING ONLY

Labeled patient
  X[z-1], X[z], X[z+1] -----------+
                                     +--> same profile A_h --> X~L
  Y[z-1], Y[z], Y[z+1] -- one-hot -+                         qL

  X[z], Y[z] -----------------------------------------------> Native-L anchor

Unlabeled patient
  X[z-1], X[z], X[z+1]
          |
          +--> EMA Teacher --> hard argmax --> slice-wise 2D LCC
          |                                      |
          +---------------- same A_h ------------+--> X~U, qU
                                                     |
                                                     +--> OAAC G_eta --> X^U

Student single concatenated forward:
  12 Native-L + 12 Re-acquired-L + 12 OAAC-U = 36 views

Loss:
  L = 1/2 (Lhard-native + Lsoft-exact-occ) + lambda(t) Lsoft-pseudo-occ

EMA update:
  Teacher <- alpha Teacher + (1-alpha) Student

                         INFERENCE

Native single slice X[z] --> same 2D U-Net --> prediction Yhat[z]
```

MPD 决定图中的 `same profile A_h` 怎样采样；PARS 决定每一步的 `Labeled/Unlabeled patient + index-third` 怎样采样。

---

## 12. 两个测试脚本究竟做了什么

### 12.1 共同流程

`test_sliceeq_occ_oaac_strong_mpd.py` 和 `_pars.py` 都会：

1. 检查 PROMISE12 根目录及固定 test list；
2. 构建二分类 2D U-Net；
3. 要求明确提供 `--checkpoint_path`；
4. 使用 `torch.load(..., map_location=device)` 读取权重；
5. 从 checkpoint 中抽取 state dict；
6. 去除可能存在的 `module.` 前缀；
7. 使用 `strict=True` 加载，网络键不一致会直接报错；
8. 调用 `net.eval()`；
9. 对每个 test volume 逐 2D slice 推理，再恢复成 3D prediction；
10. 计算每例及十例平均 Dice、Jaccard、HD95、ASD；
11. 写出 `performance.txt`，按参数决定是否保存预测。

### 12.2 默认参数

| 参数 | 默认值 |
|---|---|
| model | `unet` |
| num_classes | 2 |
| labelnum | 7 |
| stage_name | `self_train` |
| patch_size | 256 256 |
| nms | 0 |
| save_result | False |
| auto_find_checkpoint | 只允许 False |

`auto_find_checkpoint` 被故意禁用，避免脚本在多个 checkpoint 中自动挑一个看起来最好的文件。正式测试应始终明确写出 `--checkpoint_path`。

### 12.3 测试阶段没有什么

测试阶段不使用：

- EMA Teacher；
- \(z-1,z+1\) 相邻切片；
- SliceEq 重采集；
- fractional occupancy；
- OAAC；
- MPD profile sampler；
- PARS data sampler。

因此最终方法不增加测试参数量，也不要求 3D 输入。它利用 3D 邻层信息改进训练，但部署仍是普通 2D 分割。

### 12.4 HD95/ASD 的重要限制

当前结果清单把 HD95/ASD 标为 `legacy voxel-index distance`。如果测试代码没有使用真实 spacing，则这些值不是严格物理毫米距离。

论文正式报告时应：

- 从原始医学图像元数据恢复 spacing；
- 用毫米计算 HD95/ASD；
- 或清楚标注当前值为 voxel-index units，不能写成 mm。

### 12.5 当前本机权重不是可直接测试的模型文件

当前 `CoDA-MPD-Final/model` 下检查到的 `.pth` 文件均只有 `132` 字节，包括：

- MPD `iter_29000.pth`；
- MPD `unet_best_model.pth`；
- OAAC-Strong、OAAC、SliceEqOcc 和 SliceEq 的归档权重路径。

这类大小对应 Git LFS pointer，而不是真实 PyTorch checkpoint。测试脚本本身没有问题，但必须先在训练机放入实际的数 MB 模型权重；否则 `torch.load` 无法把 LFS 文本指针解析成 state dict。

---

## 13. 当前结果应该怎样正确表达

### 13.1 主要结果

| 方法 | Checkpoint 选择方式 | Dice | Jaccard | HD95 | ASD | 证据定位 |
|---|---|---:|---:|---:|---:|---|
| SliceEq | 已归档结果 | 0.832603 | 0.715429 | 4.548882 | 1.746296 | 完整开发运行 |
| SliceEqOcc | validation-selected `unet_best_model` | 0.827368 | 未在本表统一 | 未在本表统一 | 未在本表统一 | 协议选择结果 |
| SliceEqOcc | test-inspected iter23000 | 0.844566 | 0.732999 | 3.651809 | 1.439373 | 事后开发观察 |
| OAAC 1.00 | test-inspected iter27000 | 0.849538 | 0.740985 | 3.554760 | 1.868299 | 事后开发 oracle |
| OAAC-Strong 1.25 | validation-best | 0.851960 | 0.745347 | 3.228864 | 1.307063 | 当前最干净的正向单次结果 |
| OAAC 1.50 | validation-best | 0.852059 | 0.745145 | 3.294424 | 1.553528 | 未通过验证替换规则 |
| ARCP | validation-best | 0.851062 | 0.743164 | 6.217004 | 2.123881 | 负/中性，已关闭 |
| MPD | validation-best `unet_best_model` | 0.848952 | 0.740351 | 3.312648 | 1.383423 | 被后续指定 checkpoint 报告取代，但仍是协议上更干净的 MPD 结果 |
| MPD | 指定 iter29000 | **0.854573** | **0.749330** | 3.256519 | 1.324697 | 当前最高开发结果，非 validation-selected |
| MPD+PARS | 尚未运行 | 待测 | 待测 | 待测 | 待测 | 不得提前写结果 |

注：MPD validation-best 的 Jaccard/HD95/ASD 由结果清单中“iter29k 相对旧报告的差值”反推，与旧 performance artifact 一致；正式引用时仍应优先直接附原始 performance 文件。

### 13.2 MPD 相对 OAAC-Strong

MPD iter29k 相对 OAAC-Strong validation-selected checkpoint：

| 指标 | 变化 | 方向 |
|---|---:|---|
| Dice | +0.002613 | 更好 |
| Jaccard | +0.003983 | 更好 |
| HD95 | +0.027655 | 略差 |
| ASD | +0.017634 | 略差 |

这说明 MPD 的区域重叠指标更好，但表面距离没有同步改善。不能只凭 Dice 就声称边界质量全面提升。

### 13.3 为什么 `0.854573` 不能直接作为论文无偏主结果

MPD 训练中的最佳验证 Dice 是：

```text
0.836008 @ iter25800
```

被测试的 iter29000 在验证集上只有：

```text
0.828270
```

两者相差 `-0.007738`。而 `0.854573` 来自后来指定 iter29000 做 test，因此它属于：

> **checkpoint-specific development test result**

PROMISE12 test 已参与开发和 checkpoint 判断，不能再作为独立无偏 test。对老师可以说“当前观察到的最高开发结果为 0.854573”，但论文主表不能写成“我们在独立测试集上达到 0.854573”而不加限定。

更稳妥的当前结论：

1. OAAC-Strong 1.25 的 validation-selected `0.851960` 是目前更干净的单次结果；
2. MPD 表现出达到 `0.854573` 的潜力，并改善 Dice/Jaccard；
3. MPD 的 validation/test 排名不一致，需要外部数据或锁定 selector 验证；
4. PARS 目前只是待验证假设。

---

## 14. 论文贡献应该怎样写

### 14.1 可以辩护的核心贡献

1. **成对采集对齐监督**  
   对会改变组织组成的 through-plane operator，不再复用中心切片硬标签，而是对图像和目标施加完全相同的重采集算子。

2. **Fractional occupancy 表达**  
   不把重采集目标重新 argmax，而是保留采集过程自然产生的部分容积监督。

3. **有序采集-外观组合**  
   把 target-changing acquisition 与 target-invariant appearance 明确分开，先形成正确目标，再增强 U Student 外观。

4. **训练集限定的矩约束 profile 风险设计**  
   MPD 不根据 validation/test 或模型置信度调权重，而是在父 profile moments、图像扰动预算、熵和密度约束下最大化最差训练 stratum 的 retained fractional information。

5. **不改变推理图**  
   邻层、Teacher、OAAC、MPD/PARS 都是 training-only，最终仍是单切片 2D U-Net。

### 14.2 PARS 若阳性，可以增加的贡献

只有 PARS 完整运行并在冻结规则下稳定超过 MPD 后，才可以加入：

> A frozen patient-balanced axial support law aligns the data sampling risk with acquisition-derived fractional occupancy opportunity.

如果 PARS 不涨点，它应该作为负结果或附录，而不是硬塞进最终方法。

### 14.3 不能声称的内容

- 不能说发明了 gamma、contrast、brightness；
- 不能说发明了 MixUp、soft label 或 patient-balanced sampling；
- 不能说恢复了真实 scanner PSF 或 slice thickness；
- 不能把 index thirds 直接叫 apex/mid/base；
- 不能把 PARS 叫 hard mining 或 uncertainty sampling；
- 不能说 MPD 的 `0.854573` 是独立无偏测试结果；
- 不能仅凭 PROMISE12 单 seed 结果声称 CVPR SOTA；
- 不能把项目基座称为完整原始 BCP。

最窄而准确的 MPD 创新表述是：

> **Training-only moment-constrained robust design of a paired through-plane image-occupancy profile distribution.**

最窄而准确的 PARS 表述是：

> **A frozen patient-balanced axial support law designed from exact acquisition-derived fractional-occupancy opportunity for a paired through-plane image-target operator.**

---

## 15. 当前最需要补的实验

### 15.1 首要：严格无偏评测

- 冻结方法和 checkpoint selector；
- 不再查看 PROMISE12 test 来挑 iteration；
- 在 MM-WHS MRI 或其他未参与开发的 3D MRI 数据上直接迁移；
- 使用同一个选择规则比较 baseline、SliceEq hard、SliceEqOcc、OAAC-Strong、MPD。

### 15.2 公平组件消融

建议最小主表：

| 版本 | 36-view 计算匹配 | 成对图像/目标 | Fractional occupancy | OAAC | MPD |
|---|---:|---:|---:|---:|---:|
| B0 原始 EMA | 否，24 views | 否 | 否 | 否 | 否 |
| C0 计算匹配 | 是 | 否 | 否 | 否 | 否 |
| C1 SliceEq hard | 是 | 是 | 否，argmax | 否 | 父分布 |
| C2 SliceEqOcc | 是 | 是 | 是 | 否 | 父分布 |
| C3 OAAC-Strong | 是 | 是 | 是 | 是 | 父分布 |
| C4 MPD | 是 | 是 | 是 | 是 | 是 |

其中 C0 很重要，因为 B0 是 24-view，而后续方法是 Student 36-view。如果没有 C0，提升可能部分来自额外视图和 BatchNorm 统计，而不全是 paired occupancy。

### 15.3 机制分析

应重点报告：

- fractional pixels 占比；
- 相邻层发生标签变化的位置；
- hard SliceEq 与 SliceEqOcc 在边界切片/小前景切片上的差异；
- MPD 相对父 profile 的 RFI、entropy、moment 和 density diagnostics；
- 每病例 Dice，特别是持续较差病例；
- 真实 spacing 下的 HD95/ASD 或 NSD；
- 伪 occupancy calibration：预测概率是否更接近 exact labeled occupancy；
- checkpoint ranking 在 val 和外部集是否一致。

### 15.4 重复实验

当前 seed1337 单次结果只适合开发结论。若目标是 CVPR，最终至少需要：

- 多次独立运行或多 seed 的 mean ± std；
- patient-level paired statistical test；
- 外部数据集；
- 与同协议强基线比较；
- 参数量、训练成本和推理延迟。

---

## 16. 给老师汇报时可以这样讲

### 16.1 30 秒版本

> 我们发现普通半监督增强默认图像变了但标签不变，这对只改亮度的增强没问题，但对 MRI 层间重采集不成立，因为混合相邻切片后组织组成也会改变。因此我们用同一个 slice profile 同时重采集图像和标签，并保留 fractional occupancy，而不是再变成硬标签。目标正确形成后，只对无标注 Student 图像追加有序外观扰动。最后 MPD 不改网络和 loss，只用有标注训练数据设计不同合法 slice profiles 的采样概率。目前指定 iter29k 的开发 Dice 为 0.854573，但该 checkpoint 不是 validation-best，所以还需要在未参与开发的数据集上确认。

### 16.2 讲 MPD 时

> MPD 不是学一个网络注意力，也不是把权重固定成 0.2/0.6/0.2。我们在原有 sigma/phase 范围内离散出 441 个合法 profile，评价每个 profile 在 7 个病例和三个轴向索引段里能保留多少不改变中心语义的 fractional information，然后设计一个全局概率分布。优化时还锁住父方法的平均邻层质量、方向矩、图像扰动量和熵，所以变化只来自 profile 风险重新分配。

### 16.3 讲 PARS 时

> MPD 决定“对当前样本用什么采集剖面”，但原 sampler 仍会让长病例得到更多更新。PARS 只解决“下一张训练切片从哪个病例和哪个轴向索引段来”：先均衡病例，再按一个由 exact occupancy opportunity 设计的全局三段概率采样。它不是根据 loss 挑难例，也不读取验证集和测试集。目前只是待验证后继方法，不能和已有 MPD 结果混在一起。

---

## 17. 最终判断

### 已经可以确定的

- 这不是单纯改造 U-Net，而是一种训练期的 acquisition-aligned supervision 方法；
- SliceEqOcc 的核心创新是 paired through-plane re-acquisition + fractional occupancy；
- OAAC 的价值在正确的语义顺序，不在三个普通光度算子本身；
- MPD 是当前完整方法中最有研究价值的后续深化，因为它只改变 profile 分布并有明确的训练数据限定数学设计；
- 当前最高观察结果是 MPD iter29000 的 Dice `0.854573`；
- 推理仍是普通单切片 2D U-Net。

### 还不能确定的

- MPD 是否在严格 validation-selected 或外部数据上稳定优于 OAAC-Strong；
- `0.854573` 是否能在独立重复中复现；
- PARS 是否有效；
- 当前方法是否达到可比较的 SOTA；
- 提升是否全部来自 fractional occupancy，而不是额外 36-view/BN 计算，仍需要计算匹配消融。

### 当前最合理的方法定位

> **论文主方法先以 SliceEqOcc-OAAC-Strong-MPD 为核心；PARS 保持为一次冻结验证的候选 successor。**

如果 PARS 不超过 MPD，文章仍然可以围绕一条统一主线展开：

```text
目标会变化的采集增强
        ↓
图像与 occupancy 成对重采集
        ↓
保留 fractional occupancy
        ↓
在目标正确后再做外观一致性
        ↓
用训练集限定的 moment design 优化采集 profile 分布
```

这条逻辑是一种完整的方法论，而不是多个无关技术的堆叠。

---

## 18. 关键代码与证据索引

| 内容 | 文件 |
|---|---|
| SliceEqOcc/OAAC-Strong 父训练 | `code/train_sliceeq_occ_h7_15_base.py`、`code/train_sliceeq_occ_oaac_strong.py` |
| MPD 训练入口 | `code/train_sliceeq_occ_oaac_strong_mpd.py` |
| MPD 数学设计与采样器 | `code/utils/sliceeq_mpd.py` |
| MPD 严格测试入口 | `code/test_sliceeq_occ_oaac_strong_mpd.py` |
| PARS 训练入口 | `code/train_sliceeq_occ_oaac_strong_mpd_pars.py` |
| PARS 设计与 batch sampler | `code/utils/sliceeq_pars.py` |
| PARS 严格测试入口 | `code/test_sliceeq_occ_oaac_strong_mpd_pars.py` |
| 通用逐体积测试与指标 | `code/test_coda.py` |
| MPD 方法说明 | `docs/SLICEEQ_OCC_OAAC_STRONG_MPD_README.md` |
| PARS 方法说明 | `docs/SLICEEQ_OCC_OAAC_STRONG_MPD_PARS_README.md` |
| MPD 冻结协议 | `research/experiments/h7_slice_profile_reacquisition/h7_19_robust_moment_profile_gate_protocol.md` |
| PARS 冻结协议 | `research/experiments/h7_slice_profile_reacquisition/h7_20_patient_axial_acquisition_risk_protocol.md` |
| MPD 结果清单 | `research/experiments/h7_slice_profile_reacquisition/results/sliceeq_occ_oaac_strong_mpd_external_run_2026-08-19/result_manifest.md` |
| 当前研究状态 | `research/research-state.yaml` |
