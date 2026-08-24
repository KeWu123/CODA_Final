# SliceAlign 顶会方法包装与微调方案

> 目标：在不堆叠新技术、不篡改既有结果的前提下，把当前 `SliceEqOcc-OAAC-Strong-MPD` 收束为一个清晰、专业、可答辩的方法。  
> 原则：主张强度不得超过现有代码与证据；所有参数都应被归类为物理支持、数值离散、继承协议或验证选择，而不能写成“经验上效果好”。

---

## 1. 推荐的论文方法名

### 首选标题

**SliceAlign: Acquisition-Aligned Fractional Occupancy Learning for Semi-Supervised MRI Segmentation**

中文：

**SliceAlign：面向半监督 MRI 分割的采集对齐分数占据学习**

### 若希望在标题中体现 MPD

**SliceAlign: Robust Acquisition-Aligned Fractional Occupancy Learning under Through-Plane Profile Shift**

### 不建议继续使用的标题形式

```text
SliceEqOcc-OAAC-Strong-MPD-PARS
```

原因是这种名字会让审稿人先看到五个模块，而不是一个研究问题，容易被判断为技术堆叠。

代码实验名可以继续保留，以维护结果来源；论文中统一称为 `SliceAlign`。

---

## 2. 方法只保留一个主命题

> Conventional semi-supervised augmentation assumes label invariance. We instead model through-plane augmentation as a target-changing acquisition operator and require the image and its fractional occupancy target to be transformed jointly.

中文：

> 常规半监督增强默认标签在变换前后保持不变；我们把层间增强视为会改变组织组成的采集算子，因此要求图像及其分数占据目标由同一算子联合变换。

这句话是全文唯一主命题。其余设计均服务于它：

1. 成对 slice-profile operator 保证 image-target alignment；
2. fractional occupancy 保留部分容积信息；
3. profile design 决定如何在合法算子族中分配训练风险；
4. OAAC 只是目标形成后的外观覆盖策略；
5. PARS 暂不进入主方法。

---

## 3. 推荐的方法层次

### 3.1 核心一：Paired Slice-Profile Operator

定义一个训练期虚拟层间观测算子：

\[
\mathcal A_w(V_z)=\sum_{k\in\{-1,0,+1\}}w_kV_{z+k},
\qquad w_k\ge0,\quad\sum_kw_k=1.
\]

对图像：

\[
\widetilde X_z=\mathcal A_w(X_z).
\]

对有标注 one-hot mask：

\[
q_z^L=\mathcal A_w(Q_z^L).
\]

对无标注 Teacher pseudo mask：

\[
q_z^U=\mathcal A_w(\widehat Q_z^U).
\]

专业表述：

> The same convex through-plane operator is applied to the image stack and its exact-label-derived or teacher-derived occupancy stack.

不要写成“真实模拟扫描仪”，而应写成：

> an acquisition-inspired discrete through-plane observation family

因为当前数据没有真实 DICOM slice profile，三抽头高斯是合理近似，但不是经过物理标定的 scanner PSF。

### 3.2 核心二：Acquisition-Aligned Fractional Occupancy

\(q_z\) 不再 argmax 为硬标签，而作为算子诱导的分数占据目标：

\[
q_{z,c}(v)\in[0,1],\qquad\sum_cq_{z,c}(v)=1.
\]

推荐术语：

- `operator-induced fractional occupancy`；
- `acquisition-aligned soft target`；
- L 分支：`exact-label-derived occupancy`；
- U 分支：`teacher-derived pseudo occupancy`。

不要写：

- `ground-truth physical tissue fraction`；
- `exact physical occupancy`。

当前 occupancy 对离散标签和指定算子是精确的，但不等于真实组织在连续物理空间中的体积分数。

### 3.3 核心三：Moment-Constrained Profile Design

原始 profile family 由 \((\sigma,\phi)\) 参数化。MPD 不学习新网络，而是在该固定支持上设计抽样分布 \(\pi\)：

\[
\pi^*=\arg\max_\pi\min_s
\mathbb E_{w\sim\pi}[S_s(w)],
\]

其中 \(s\) 表示 patient × normalized axial bin，\(S_s(w)\) 表示语义保持条件下的分数占据熵。

第二阶段执行最小偏离投影：

\[
\min_\pi D_{KL}(\pi\Vert\pi_0),
\]

同时保持邻层质量、方向矩和图像扰动预算接近父分布。

论文中建议把 MPD 展开为：

**Moment-Constrained Profile Design**

不要使用“physical-optimal profile”或“scanner profile recovery”。

---

## 4. 把 RFI 改成更准确的术语

当前 `Retained Fractional Information` 容易被审稿人追问：它是否真的是互信息、Fisher information 或具有信息论证明？目前不是。

推荐论文术语：

**Semantic-Preserving Occupancy Entropy（SPOE）**

中文：**语义保持的占据熵**。

定义：

\[
S_s(w)=
\frac{
\sum_v o_s(v)
\mathbf 1[\arg\max \mathcal A_w(Q)(v)=Y_z(v)]
H(\mathcal A_w(Q)(v))
}{
\sum_vo_s(v)+\epsilon
},
\]

其中 \(o_s(v)\) 只表示相邻标签确实发生变化的位置。

这个名字准确表达了三个事实：

1. 计算量本质上是 occupancy entropy；
2. 只在相邻层存在变化时计算；
3. 只有中心 hard semantic identity 保留时才计入。

代码和已有 JSON 中可以继续保留 `RFI` 字段以维护 provenance；论文正文首次出现时写：

> We operationalize retained fractional information using semantic-preserving occupancy entropy (SPOE).

后文统一使用 SPOE，避免伪装成严格信息论量。

---

## 5. OAAC 应该怎样降级包装

OAAC 不应作为与 SliceAlign 并列的主创新，也不应出现在论文标题中。

推荐定位：

### Ordered Appearance Completion

它是一个语义安全的训练策略：

\[
(X,Q)\xrightarrow{\mathcal A_w}(\widetilde X,q)
\xrightarrow{G_\eta}(G_\eta(\widetilde X),q).
\]

先执行 target-changing acquisition，再执行 target-preserving appearance transform。

推荐表述：

> After the acquisition-aligned target has been formed, we complete the unlabeled perturbation space with coordinate-preserving monotonic appearance transforms.

OAAC 的贡献不是 gamma、contrast、brightness，而是顺序约束。`Strong 1.25` 只放在 Implementation Details 和 sensitivity table 中。

---

## 6. 所有“直接来的参数”怎样解释

| 当前参数 | 审稿风险 | 专业定位 | 还需补什么 |
|---|---|---|---|
| `sigma=[0.45,0.85]` | 为什么是这个范围 | 父方法冻结的 bounded virtual profile support，不声称真实 PSF | 报告中心权重范围与 0.35/0.45/0.55 边界敏感性，或在外部集直接冻结迁移 |
| `phase=[-0.25,0.25]` | 是否随意 | 限制虚拟采集中心不偏移超过四分之一 slice，保证中心语义优先 | 报告 hard semantic flip rate 和 phase 对称性 |
| OAAC `1.25×` | 明显像调参 | 在 1.00/1.25/1.50 三点联合敏感性中由 validation 选择的局部设置 | 不写进方法名，不再继续搜索 |
| `21×21` grid | 为什么不是 20 或 30 | profile 积分的数值离散分辨率，不是模型超参数 | 做 11/21/31 零训练 grid convergence，比较 moments、SPOE、JS divergence |
| 3 axial bins | 为什么分三段 | 低样本条件下的 coarse normalized axial support discretization | 做 2/3/4 bins 零训练稳定性；只能叫 index bins |
| moment `±2%` | 人工阈值 | parent-preserving trust region | 报告哪些约束 active；补 1/2/5% 零训练分布稳定性 |
| image RMS `±5%` | 人工阈值 | 防止把 SPOE 增益变成增强强度增益的 nuisance budget | 补 2.5/5/10% 零训练稳定性 |
| entropy ≥70% | 看起来随意 | anti-collapse safeguard | 当前实际保留98.83%，约束很可能不活跃；报告 inactive status |
| density cap 3× | 看起来随意 | 防止少数 grid point 垄断 | 当前最大1.608×；报告 constraint slack，若始终不活跃则从主公式移到安全实现 |
| 保留99% stage-1 optimum | 为什么不是98% | lexicographic optimization 的数值容差 | 做 98/99/99.5% 零训练稳定性，或称近似词典序优化 |
| warm-up 1000 | 是否为本方法调参 | inherited baseline schedule | 明确声明完全继承，不列为贡献 |
| `0.5(Lnative+Lreacquired)` | 是否为调参 | 等权经验风险平均，避免验证集调 loss 权重 | 保持不变即可 |
| EMA 0.99、LR0.01 | 是否专门调过 | inherited baseline optimizer protocol | 保持不变并在表中标“frozen” |
| Teacher hard + 2D LCC | 是否损失信息 | inherited pseudo-label topology prior | 用作固定条件，不包装成贡献 |

关键写法不是“这些值经过大量实验发现最好”，而是：

> We separate support-defining constants, inherited optimization constants, numerical discretization constants, and validation-selected appearance sensitivity. Only the last category uses validation performance, and none uses the test set in the frozen protocol.

当前历史上 test 已参与过开发，因此这句话只能用于未来重新冻结的外部实验协议，不能倒过来洗白 PROMISE12 既有结果。

---

## 7. 建议进行的微小改动

### 7.1 不改变训练结果的改动

1. 论文总名改为 `SliceAlign`。
2. `Acquisition-Equivariant` 全部改为 `Acquisition-Aligned`。
3. `exact occupancy` 改为 `exact-label-derived operator occupancy`。
4. `physical profile` 改为 `acquisition-inspired virtual profile`。
5. RFI 在正文中改称 SPOE，代码字段保留 RFI。
6. OAAC 从主贡献降为 ordered appearance completion。
7. `Strong 1.25` 只进入实现细节和敏感性实验。
8. PARS 暂时只作为 appendix candidate，不进入标题和主方法图。
9. 所有 test-selected 数值统一标记为 development observation。
10. 方法图只画两个主块：`Paired Acquisition Alignment` 与 `Moment-Constrained Profile Design`。

### 7.2 不需要 GPU 重训的审计

1. Grid convergence：11×11、21×21、31×31。
2. Axial discretization：2、3、4 个 normalized index bins。
3. Trust-region sensitivity：moment 1/2/5%，RMS 2.5/5/10%。
4. Lexicographic tolerance：98/99/99.5%。
5. Constraint activity：报告每个约束 slack 和拉格朗日乘子。
6. Seven-patient LOPO：比较七个 \(\pi_{-p}\) 与全量 \(\pi\) 的 JS divergence。
7. 删除不活跃约束后重新求解，检查 \(\pi\) 是否实质不变。

这些实验只重新设计概率分布，不训练分割网络，成本很低，却能显著减少“参数拍脑袋”的印象。

### 7.3 最多允许的一次方法级微调

若零训练审计发现 `±2%/±5%` 对分布高度敏感，可以新开一个干净候选版本，用训练病例 bootstrap 产生 trust-region radius：

\[
|E_\pi[\psi_j]-E_{\pi_0}[\psi_j]|
\le 1.96\,\widehat{SE}_{patient}(\psi_j).
\]

这样容差来自有标注训练病例之间的统计波动，而不是固定百分比。

但只有在审计显示当前阈值不稳定时才值得重训。若当前分布在上述敏感性范围内稳定，就保留现有 MPD，避免为了包装而改坏已经成立的方法。

---

## 8. 顶会版 Method 写法

### 8.1 Problem Formulation

给定少量有标注体积 \(\mathcal D_L\) 和大量无标注体积 \(\mathcal D_U\)，目标是在不改变单切片推理图的条件下，利用训练期相邻层信息。

现有 weak-to-strong consistency 通常要求：

\[
T(g(X))=T(X),
\]

即增强 \(g\) 保持标签不变。但 through-plane aggregation 会改变有限层厚内的组织组成，因此该假设不成立。

### 8.2 Acquisition-Aligned Occupancy Construction

我们定义共享算子 \(\mathcal A_w\)，同时作用于图像与 occupancy：

\[
(\widetilde X,q)=
(\mathcal A_w(X),\mathcal A_w(Q)).
\]

有标注分支使用 one-hot GT occupancy；无标注分支使用 EMA Teacher 产生的邻层 pseudo occupancy。Student 同时学习原始中心硬标签锚点和重新采集后的 fractional target。

### 8.3 Robust Profile-Risk Design

在合法 profile family \(\mathcal W\) 上，SliceAlign 设计训练分布 \(\pi\)，最大化跨病例和归一化轴向区域的最差 SPOE，同时通过矩、图像 residual 与 KL 投影保持对父采集预算的最小偏离。

这一设计不是为每个样本预测 profile，也不读取 validation/test、模型置信度或 loss；它在训练开始前由 labeled-training stacks 一次性求得并冻结。

### 8.4 Training Objective

\[
\mathcal L=
\frac12\left[
\mathcal L_{hard}(f(X_z^L),Y_z^L)
+\mathcal L_{soft}(f(\mathcal A_w(X^L)),\mathcal A_w(Q^L))
\right]
+\lambda(t)
\mathcal L_{soft}(f(G_\eta(\mathcal A_w(X^U))),
\mathcal A_w(\widehat Q^U)).
\]

### 8.5 Inference

\[
\widehat Y_z=f_\theta(X_z).
\]

训练期的相邻层、Teacher、profile operator、SPOE design 和 appearance completion 全部移除，不增加参数量或推理延迟。

---

## 9. 三条贡献建议

1. **Acquisition-aligned supervision.** 识别并修正了 through-plane augmentation 下传统 label-invariance 假设的失效，使用同一非可逆观测算子联合构造图像与目标。
2. **Fractional occupancy learning.** 将层间采集产生的部分容积表示为连续 occupancy，而不是重新硬化为中心标签，并同时用于 exact-label-derived 和 teacher-derived 监督。
3. **Moment-constrained profile design.** 在保持父 profile 的混合强度、方向能量、图像扰动和分布多样性的条件下，稳健设计跨病例/轴向区域的训练算子分布，同时保持单切片推理。

OAAC 不单独列第四条贡献。PARS 未取得冻结结果前也不列贡献。

---

## 10. 可直接使用的摘要核心段

> Semi-supervised segmentation commonly assumes that an augmented image can be supervised by an unchanged hard target. This assumption breaks for through-plane MRI perturbations, where finite slice support changes the observed tissue composition. We introduce SliceAlign, a training-only framework that applies the same discrete slice-profile operator to neighboring images and their exact-label-derived or teacher-derived occupancy maps, producing acquisition-aligned fractional targets. To avoid relying on a heuristic uniform profile sampler, we further design a global profile distribution that maximizes semantic-preserving occupancy entropy across labeled subjects and normalized axial regions while preserving the parent operator's moments, image perturbation budget, and diversity. Coordinate-preserving appearance perturbations are applied only after the acquisition-aligned target is formed. SliceAlign retains the original single-slice 2D network at inference and introduces no additional parameters or latency.

结果句在完成无偏实验后再加入。当前不要直接把 `0.854573` 写进摘要主结论。

---

## 11. 审稿人常见问题与回答

### Q1：这不就是相邻切片 MixUp 吗？

不是。普通 MixUp 主要在样本间做凸组合；SliceAlign 在同一病例的有序相邻层上模拟 bounded through-plane observation，并把同一算子作用于 image 与 exact/pseudo occupancy。核心问题是非标签保持采集增强下的 target construction。

### Q2：为什么不用 2.5D 网络？

2.5D 方法在推理时依赖相邻层。SliceAlign 仅在训练时利用相邻层构造采集对齐监督，测试仍为原单切片 2D U-Net。论文仍必须加入一个计算匹配的 2.5D 对照。

### Q3：高斯 profile 是真实扫描仪 profile 吗？

不是。它是 acquisition-inspired virtual profile family。论文不声称恢复真实 PSF；真实 metadata/PSF 外部实验用于评估对 acquisition shift 的泛化。

### Q4：为什么是 1.25？

它是 1.00/1.25/1.50 三点联合 sensitivity 中由 validation 选择并冻结的局部设置，不是创新，也不声称全局最优。

### Q5：MPD 是否用 test 调分布？

数学设计只读取前 191 个 labeled-training slices；不读取 U labels、validation、test、prediction、confidence 或 loss。但当前 PROMISE12 的 checkpoint 已被 test 检查，因此最终性能仍需在 untouched external evaluation 上确认。

### Q6：MPD 是否只是手工超参数更多？

核心分布由约束优化一次求得，而不是网格搜索 Dice。顶会版需补充离散分辨率、trust region、axial bins 和 LOPO 稳定性，证明结果不是某个常数偶然造成。

---

## 12. 最终建议

论文主方法采用：

```text
SliceAlign
  = Acquisition-Aligned Fractional Occupancy
  + Moment-Constrained Profile Design
```

实现细节采用：

```text
EMA Teacher
+ ordered appearance completion
+ inherited 1000-step warm-up
```

附录候选：

```text
PARS
```

这样包装后，方法不再表现为多个模块累加，而是一条清楚的逻辑：

> **先承认采集增强会改变目标，再联合构造 image-target observation，最后在不改变采集预算的前提下设计训练算子分布。**

