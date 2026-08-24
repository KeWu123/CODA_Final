# SliceEqOcc 三切片融合深化：文献与原创性审计

日期：2026-08-18  
范围：只研究 SliceEqOcc 的三切片 profile/fusion 模块；不改变网络、EMA、损失、OAAC、batch、验证或推理。

## 1. 先纠正当前实现的定义

当前方法不是固定使用 `[0.2, 0.6, 0.2]`。正式训练代码对每个样本独立采样

\[
\sigma\sim U(0.45,0.85),\qquad \phi\sim U(-0.25,0.25),
\]

并在三个离散位置 \(k\in\{-1,0,1\}\) 上计算

\[
w_k=\operatorname{softmax}_k\left[-\frac{1}{2}
\left(\frac{k-\phi}{\sigma}\right)^2\right].
\]

`[0.2,0.6,0.2]` 只是一个很有代表性的对称核：当 \(\phi=0\)、
\(\sigma=\sqrt{0.5/\ln 3}\approx0.6746\) 时，上式恰好约为该比例。它位于当前采样域内部，
不是由 PROMISE12 验证集求出的最优比例，也不是当前训练全程固定的比例。

当前 profile 的实际覆盖示例：

| \(\sigma\) | \(\phi\) | \([w_{-1},w_0,w_{+1}]\) |
|---:|---:|---:|
| 0.45 | 0 | [0.0724, 0.8552, 0.0724] |
| 0.6746 | 0 | [0.2000, 0.6000, 0.2000] |
| 0.85 | 0 | [0.2501, 0.4997, 0.2501] |
| 0.45 | -0.25 | [0.2212, 0.7601, 0.0187] |
| 0.45 | +0.25 | [0.0187, 0.7601, 0.2212] |

因此真正的问题不是“0.2/0.6/0.2 是否最优”，而是：**当前随机 profile 是否应由每个扫描本身的原生
层厚/层间模糊来约束，并且怎样保证再次合成的切片在物理上只能比原始观测更模糊、不能凭空更锐。**

## 2. 为什么线性加权不是方法的弱点

MRI 多层采集通常可抽象为连续解剖沿层面方向与 slice profile/PSF 卷积后采样。在线性观测模型下，
对相邻切片作非负、归一化加权是离散化的 through-plane 积分。SliceEqOcc 再把同一权重作用于图像和
one-hot 组织占比，得到 fractional occupancy target。这一配对关系是方法成立的核心。

如果把融合换成 MLP、最大值、非线性注意力或像素级权重，通常不存在与之严格对应的组织占比算子；
此时图像和监督目标再次发生语义错配。因此本轮不追求“更复杂的融合函数”，而是深化**核从哪里来**。

## 3. 文献边界

### 3.1 直接相关

- Han 等的 ESPRESO 从单个多切片 MRI 的内部 patch 分布估计图像特定的 slice profile，说明无需重复扫描
  也可能从图像本身估计 through-plane PSF；其用途主要是超分辨率，不是半监督分割或 fractional
  occupancy。[PubMed](https://pubmed.ncbi.nlm.nih.gov/36702167/)；
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0730725X23000127)
- 早期 MRI slice-profile estimation 也从图像数据估计 profile，但仍面向重建/超分辨率，而非配对分割监督。
  [arXiv:2104.00100](https://arxiv.org/abs/2104.00100)
- MRI 超分辨率/重建工作把低分辨率切片写成高分辨率体与 slice-selection/PSF 的前向模型，为线性卷积与
  相对退化提供物理基础。[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4644155/)

### 3.2 相邻但不同的路线

- HyperSpace 使用 voxel spacing 条件化超网络生成分割模型；它改变模型条件与推理，而不是训练期的配对
  image--occupancy operator。[MICCAI 2024](https://papers.miccai.org/miccai-2024/394-Paper2109.html)
- Borges 等将 acquisition parameters 与 physics simulation 用于 acquisition-invariant MRI segmentation，
  但不同模拟图像继续使用同一 anatomy label，没有表达 profile-dependent fractional occupancy。
  [Medical Image Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617170/)
- AFTer-UNet 等方法在网络中学习跨切片注意力，属于 2.5D/上下文融合，会改变网络和推理条件。
  [WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Yan_AFTer-UNet_Axial_Fusion_Transformer_UNet_for_Medical_Image_Segmentation_WACV_2022_paper.pdf)
- Learned augmentation、task-driven augmentation 与 adversarial augmentation 会从任务损失优化扰动策略；
  它们不能防止 profile 退化为对分割最容易的 identity 核。
  [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Data_Augmentation_Using_Learned_Transformations_for_One-Shot_Medical_Image_Segmentation_CVPR_2019_paper.html)，
  [MedIA 2021](https://www.sciencedirect.com/science/article/pii/S136184152030298X)

在针对 2026-08-18 的定向检索中，没有找到与下列完整组合同构的方法：

> 扫描特定的原生 profile 估计 → 只允许物理合法的相对退化 → 同一核作用于 MRI 与 exact/pseudo
> fractional occupancy → 半监督训练 → 保持单切片 2-D 推理。

这不是“绝对无人做过”的证明。安全的新颖性表述应为：**本方法把已有的 slice-profile estimation 思想
转化为半监督分割中的 native-aware relative re-acquisition，并与 profile-dependent occupancy 配对。**

## 4. 五类深化方案比较

| 方案 | 科学性 | 新颖性风险 | 与现方法兼容 | 结论 |
|---|---|---|---|---|
| 验证集搜索单一固定比例 | 低 | 高：普通超参搜索 | 高 | 不推荐；会损失 profile 多样性并拟合小验证集 |
| 用注意力/MLP逐样本学习权重 | 中 | 高：接近 2.5D attention/learned augmentation | 中 | 不推荐；容易塌缩为 identity，且改变方法语义 |
| 像素级自适应或非线性融合 | 低 | 高：变成 anatomy attention | 低 | 不推荐；破坏统一物理 profile 与 target pairing |
| metadata-conditioned profile | 高 | 中 | 高 | 有价值，但当前 H5 缺少可靠 thickness/profile 元数据 |
| 自校准原生 profile + 相对退化 | 高 | 中低 | 很高 | **首选**；只替换 profile 生成器 |

## 5. 推荐方法：SCRP-SliceEqOcc

工作名：**Self-Calibrated Relative Profile SliceEqOcc (SCRP-SliceEqOcc)**，中文可称
“自校准相对层面响应 SliceEqOcc”。

### 5.1 原生观测而非理想切片

当前输入已经是经过原生 profile \(h_n\) 观测的切片：

\[
X^{obs}=\mathcal A_{h_n}(X^*).
\]

新的增强不应把它当作零厚度理想切片。选择目标 profile \(h_t\) 时，要求存在额外退化核 \(h_\Delta\)，

\[
h_t=h_\Delta * h_n,
\]

从而只模拟“同等或更厚/更模糊”的合法采集。若用 Gaussian 宽度近似，

\[
s_\Delta=\sqrt{\max(s_t^2-s_n^2,0)}.
\]

### 5.2 从图像自校准 \(s_n\)

优先从原始 NIfTI/DICOM 恢复 spacing、thickness、gap 和方向；它们作为 profile 的初值或约束，而不是直接
等同于完整 PSF。若 thickness 不完整，可借鉴 ESPRESO 的内部 patch 匹配，在**仅训练集、无标签**的原始体上：

1. 采集面内 edge/gradient patch 与层间 patch；
2. 对面内 patch 施加一组候选 1-D profile；
3. 用 sliced-Wasserstein、MMD 或固定频谱距离比较模拟 patch 与真实 through-plane patch 的分布；
4. 每个病例或 protocol 选择稳定的标量宽度 \(s_n\)，不通过分割损失反向学习。

本项目不照搬 ESPRESO 的生成网络；只借鉴“内部 patch 分布可识别 slice profile”的观测，采用简单、可审计的
标量/低维核估计器。

### 5.3 用区间积分而不是中心点采样离散化

对 \(k\in\{-1,0,+1\}\)，建议使用 Gaussian 在切片 bin 上的概率质量：

\[
\tilde w_k=\Phi\!\left(\frac{k+1/2-\phi}{s_\Delta}\right)-
\Phi\!\left(\frac{k-1/2-\phi}{s_\Delta}\right),\qquad
w_k=\frac{\tilde w_k}{\sum_{j=-1}^{1}\tilde w_j}.
\]

这比在三个中心点直接取 Gaussian 值更接近有限厚度切片积分。体积端点只对实际存在的邻层 support 重新归一，
不再用复制同一端点切片的方式重复计数。

### 5.4 保持 SliceEqOcc 的配对合同

\[
\tilde X_i=\sum_{k=-1}^{1}w_kX_{i+k},\qquad
\tilde Q_i=\sum_{k=-1}^{1}w_k\operatorname{onehot}(Y_{i+k}).
\]

L 分支使用 exact mask，U 分支使用原 EMA hard/LCC pseudo stack；OAAC 仍只在完成 re-acquisition 后作用于
U 图像。网络、损失、student batch36、teacher、EMA、验证和 2-D 推理均不改变。

## 6. 风险与反证条件

- **可辨识性风险**：解剖方向性、配准误差和真实 PSF 会共同影响 patch 分布；估计值不等于 scanner ground truth。
- **预处理风险**：若只剩逐切片归一化/resize 后的 H5，through-plane 统计可能已被破坏。没有原始体时应停止，
  不能把估计器包装成“物理校准”。
- **三层截断风险**：较宽的相对 profile 可能需要五层以上 support；若三层有效质量不足，不应强行截断后宣称物理。
- **塌缩风险**：若估计器输出几乎恒定，说明病例自适应没有必要，应保留当前随机 profile。
- **因果混杂**：边界 valid-support renormalization 与 case calibration 必须分开对照。

## 7. 论文贡献的安全表述

1. 用 native-aware relative degradation 替代把已采集 MRI 当成理想零厚度切片的假设。
2. 将扫描内自校准的 relative profile 同时施加到图像和 fractional occupancy，维持非标签保持采集变换下的监督一致性。
3. 不增加分割网络参数或推理开销，仍使用单切片 2-D inference。

不能声称首次 slice-profile estimation、首次 acquisition-conditioned segmentation、首次跨切片融合、首次 learned
augmentation，或已恢复真实 scanner PSF。

