# ARCP 之后的 SliceEq profile 模块检索与收敛

日期：2026-08-18
范围：只允许改变三切片 profile/fusion 模块；冻结 `SliceEqOcc-OAAC-Strong` 的网络、预训练、
SGD/LR、EMA train-mode、pseudo-label/LCC、loss/ramp、batch、OAAC、验证、checkpoint 规则和
单切片推理。当前仅有按病例排序的 H5，无法取得原始 NIfTI/DICOM 或可信采集元数据。

## 1. 现有三切片算子的更清晰参数化

令三抽头权重为 \(w=(w_-,w_0,w_+)\)，定义

\[
b=w_-+w_+=1-w_0,\qquad
r=\frac{w_+-w_-}{b}.
\]

则

\[
w_-=\frac{b(1-r)}2,\quad w_0=1-b,\quad
w_+=\frac{b(1+r)}2,
\]

并且

\[
\mathcal A_w(X)-x_0=b\left[
r\frac{x_+-x_-}{2}+\frac{x_--2x_0+x_+}{2}\right].
\]

这里 \(b\) 是邻层总混合质量，\(r\) 是左右方向偏移。当前点采样 Gaussian 恰好满足

\[
r=\tanh(\phi/\sigma^2),
\]

\[
b=\frac{2e^{-1/(2\sigma^2)}\cosh(\phi/\sigma^2)}
{1+2e^{-1/(2\sigma^2)}\cosh(\phi/\sigma^2)}.
\]

因此 `sigma` 与 `phase` 并不是正交的“模糊强度”和“位移”旋钮。固定
`[0.2,0.6,0.2]` 只是 \(\phi=0\)、
\(\sigma=\sqrt{1/(2\ln3)}\approx0.6746\) 的代表点；生产代码实际在整个父分布中逐样本采样。

## 2. 本轮检索得到的边界

### 2.1 为什么不能把“学习最佳比例”作为主要创新

- ICT 已在 Mean Teacher 语境中约束插值输入与插值预测的一致性，说明“混合输入和目标”本身不是空白。
  [IJCAI 2019](https://www.ijcai.org/proceedings/2019/504)
- AdaMix、AdvChain 与 Learn2Synth 分别覆盖动态 mix 强度、对抗式增强链和用真实标签/超梯度优化合成策略。
  把它们缩成三个权重会成为已有 adaptive/learned augmentation 的应用，而不是新的基本原理。
  [AdaMix, MedIA 2026](https://pubmed.ncbi.nlm.nih.gov/41274085/)，
  [AdvChain, MedIA 2022](https://mediatum.ub.tum.de/doc/1768093/1768093.pdf)，
  [Learn2Synth, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Hu_Learn2Synth_Learning_Optimal_Data_Synthesis_Using_Hypergradients_for_Brain_Image_Segmentation_ICCV_2025_paper.html)
- 当前 PROMISE12 validation 只有 5 个病例，test 又已经长期参与开发。通过 segmentation
  validation 学习或搜索 profile，极易得到 split-specific policy。

### 2.2 为什么不优先做 bin-integrated 或五抽头

连续 Gaussian 在三个单位 bin 上积分的确比在三个中心点取值更接近有限体积离散：

\[
\tilde w_k=\Phi((k+1/2-\phi)/\sigma)-
\Phi((k-1/2-\phi)/\sigma).
\]

本地固定网格审计显示，在当前 \(\sigma,\phi\) 域内：

- 三 bin 平均漏掉连续 Gaussian 约 `3.044%` 的质量，最坏约 `9.046%`；
- 五 bin 平均遗漏约 `0.071%`，最坏约 `0.467%`；
- 约 `24.6%` 的父 profile 在三-bin support 内质量低于 `95%`；
- 三点采样与三-bin积分权重的平均 L1 距离约 `0.106`，中心权重平均从约 `0.625`
  降到 `0.572`，即它同时显著改变增强强度；
- 若反向映射 sigma 使积分版逐 profile 匹配父中心权重，平均权重 L1 仅约 `0.00255`，又近乎恒等。

这并不自动证明积分版更好。H5 切片本身已是未知原生 slice profile 的观测，再做 bin integration
可能构成不可识别的二次积分。离散 Gaussian 文献也表明，在细尺度下，点采样和区间积分 Gaussian
都不天然优于真正的离散 Gaussian。[JMIV 2024](https://arxiv.org/abs/2311.11317)

五抽头外侧 `±2` 总质量平均约 `2.96%`、95分位约 `7.22%`。它还会把 U teacher 输入从
`12x3=36` 改为 `12x5=60`，在本项目固定 train-mode BN/dropout
teacher 下同时改变 batch statistics、随机路径与计算量。因此：三-bin只能作为 severity-matched
数值消融；五抽头不是当前三切片合同下的干净小改。

### 2.3 为什么不做 nonlinear/attention/edge-aware fusion

当前凸组合对类别无关、保持概率质量，并把任意 one-hot occupancy 映射回概率单纯形；因此同一算子
可以无歧义地作用于 MRI 和组织占比。这正是 SliceEqOcc 的理论闭环。

Cross-slice attention、插帧、MLP 或像素级 bilateral 权重已有强邻近工作，而且会从统一、平移不变的
slice profile 变成解剖条件表示学习。[SAINT, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Peng_SAINT_Spatially_Aware_Interpolation_NeTwork_for_Medical_Slice_Synthesis_CVPR_2020_paper.html)，
[AFTer-UNet, WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Yan_AFTer-UNet_Axial_Fusion_Transformer_UNet_for_Medical_Image_Segmentation_WACV_2022_paper.html)，
[CSAM, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Hung_CSAM_A_2.5D_Cross-Slice_Attention_Module_for_Anisotropic_Volumetric_Medical_WACV_2024_paper.html)

更关键的是，edge-aware 权重会抑制层间差异大的位置，和 ARCP 在 apex/base 等快速轴向变化处削弱
有效 fractional signal 的失败机制同向。因此不再继续这一类逐 stack 自适应。

### 2.4 为什么 semigroup/composition 不是三抽头答案

若非平凡三抽头核在 \(\pm1\) 有正质量，则它与自身卷积后在 \(\pm2\) 必有正质量；所以任意固定
有限支持都不可能构成非平凡的连续卷积半群。截断或重归一会破坏半群。两次三抽头与一次五抽头若
只比较解析卷积目标，只是代数恒等式；若再加预测一致性，则已经改变 loss，而不再是只优化融合模块。

## 3. 候选排序

| 排名 | 候选 | 预计 ID 收益 | 原创性 | 结论 |
|---:|---|---:|---:|---|
| 1 | 全局 Robust Moment-Profile Design | 低至中，必须先 gate | 中 | 唯一保留的条件候选 |
| 2 | moment-space max-entropy / D-optimal sampling | 低 | 低至中 | 与 SAQ 负结果相邻，不跑 |
| 3 | profile CVaR / KL-DRO | 中但高风险 | 低至中 | 与 AugMax/AdvChain/GroupDRO 碰撞，且可能放大 U pseudo error |
| 4 | bin-integrated 3-tap | 很低 | 低 | 只作 severity-matched 消融 |
| 5 | 5-tap/full-support profile | 低，外部鲁棒性或较高 | 低 | teacher BN/compute/support 合同混杂，不跑 |
| 6 | learned/attention/nonlinear fusion | 不确定 | 高碰撞 | 破坏 paired occupancy 语义，拒绝 |

## 4. 唯一条件候选：Robust Moment-Profile Design

该候选不寻找一组固定“最佳比例”，也不按每个 stack 的图像响应改变权重。它在父 Gaussian 安全域
的固定 profile 网格上，仅用 labeled-training exact masks 设计一个**全局、病例无关**的采样分布
\(q\)。目标是使每个病人、每个轴向索引三等分都能稳定获得 retained fractional information，同时
保持父 profile 的主要矩和图像扰动预算。

其核心张力是：

> profile 必须产生足够 fractional occupancy，而不是 identity；但也不能靠更强 blur 或 hard semantic
> flip 获得表面上的信息增益。

因此用 exact occupancy entropy 衡量 soft 信息，只在 re-acquired target 的 argmax 仍等于中心 GT 时计入；
再用父分布矩、phase 镜像、密度比、熵和逐层图像 residual 约束防止坍缩或纯强度调参。

这与 ARCP 的差别是：ARCP 逐 stack 反向缩放邻层质量，容易压低真实轴向转换；新候选只设计一个全局
distribution，不读取当前 anatomy、模型、prediction、loss、validation 或 test。

## 5. 原创性与停止规则

定向检索未发现以下完整组合：

> moment-resolved three-slice profile distribution + patient/axial-stratum robust exact-occupancy design +
> identical image/fractional-occupancy operator in semi-supervised segmentation + unchanged 2-D inference.

这不是“首次 DRO/最优增强”的声明。安全表述只能是：**为非标签保持的 through-plane paired operator
设计训练标签内、moment-constrained 的 fractional-information profile distribution。**

H7.19 只授权一次零训练 LOPO gate，不直接授权 30k。若 gate 不通过，profile 模块正式冻结为
`SliceEqOcc-OAAC-Strong`；bin integration、五抽头、DRO、semigroup、edge-aware 和 learned fusion
均不再救援，算力转向 MM-WHS 外部验证及核心因果消融。
