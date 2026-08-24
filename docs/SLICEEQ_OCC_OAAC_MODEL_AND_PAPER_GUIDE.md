# SliceEqOcc-OAAC：模型原理、研究动机与论文构建指南

> 文档状态：基于 2026-08-14 的代码与单次 PROMISE12 开发实验整理。
> 本文档区分“已经由代码/工件支持的事实”和“论文仍需补充的证据”，不把单
> seed、反复查看过的开发测试集结果写成最终 SOTA 结论。

## 1. 方法定位

SliceEqOcc-OAAC 是一个用于半监督 MRI 分割的**训练期增强方法**。网络仍是原来的
单切片 2D U-Net；训练时临时读取中心切片的相邻真实切片，用一个离散的 through-plane
采集算子同时重采集图像和组织占用率，测试时仍只输入一张 2D 切片。

方法由两个有明确顺序的部分组成：

1. **SliceEqOcc**：先执行会改变目标语义的 through-plane 重采集；图像和组织
   occupancy 必须使用同一组权重。
2. **OAAC**：再对未标注学生图像执行不改变坐标和组织占用率的单调光度变换。

简写为：

```text
相邻真实切片 X[z-1:z+1]
        |
        |  EMA teacher 产生未标注伪分割体（训练期）
        v
同一 slice profile A_h
        |-----------------------------|
        v                             v
重采集图像 A_h(X)             fractional occupancy A_h(Y)
        |
        |  OAAC 单调外观变换 G_eta（仅 U 图像）
        v                             |
学生 f_theta(G_eta(A_h(X_U)))  <------|  soft CE + soft Dice

测试：单张 2D 图像 -> 同一个 U-Net；不需要相邻切片或 OAAC
```

OAAC 不增加网络层、可学习参数、teacher/student 前向次数或推理开销。训练入口见
[`code/train_sliceeq_occ_oaac.py`](../code/train_sliceeq_occ_oaac.py)，核心算子见
[`code/utils/sliceeq.py`](../code/utils/sliceeq.py)、
[`code/utils/sliceeq_occ.py`](../code/utils/sliceeq_occ.py) 和
[`code/utils/sliceeq_oaac.py`](../code/utils/sliceeq_oaac.py)。

## 2. 研究动机

### 2.1 常规增强隐含的 target-invariance 假设

常规半监督分割通常用 teacher 在弱视图上产生目标，再让 student 在强视图上拟合该
目标。这隐含了一个条件：增强后的像素仍对应原来的组织语义。对翻转、单调灰度变化等
变换，这一假设通常合理；但对 through-plane 混合并不成立。

MRI 的一张 2D 切片不是无限薄的数学平面。有限层厚、层间距和偏移会使一个观测包含
相邻解剖平面的信号。当虚拟切片混合 `z-1,z,z+1` 时，其边界像素可能同时包含前景和
背景。此时继续使用中心切片的二值 mask，会把“已经改变的观测”与“没有改变的目标”
错误配对。

### 2.2 需要的是 acquisition-derived occupancy，而非标签平滑

SliceEqOcc 不在所有边界上人为加入固定平滑，也不把 teacher 不确定性当作组织混合。
它先确定一次具体的采集 profile，再用相同权重计算每个像素由各组织贡献的比例。因此
soft target 是 forward operator 的确定性结果：

- annotation uncertainty：不同标注者可能给出不同边界；
- epistemic uncertainty：模型不知道正确答案；
- **acquisition occupancy**：给定相邻组织和采集权重后，当前观测确实混合了多种组织。

本文方法处理第三种现象。三者不能在论文中混写。

### 2.3 为什么在 SliceEqOcc 后加入 OAAC

SliceEqOcc 解决了 target-changing acquisition 的配对问题，但其 U student 视图只包含
一次中等强度的重采集。SCPO、ADU 等后继实验只改变极少数伪标签像素，实际作用量接近
零。OAAC 因而选择一个覆盖全部 U 样本、但不会再改变 occupancy 的扰动：先正确形成
采集目标，再扩大 student 的外观扰动范围。

这一顺序非常重要：

```text
target-changing A_h：必须同时作用于 image 和 occupancy
target-invariant G_eta：只能在 A_h 之后作用于 image
```

## 3. SliceEqOcc 的数学原理

### 3.1 三切片训练样本

对中心位置 `z`，训练数据返回：

```text
X_z = {x[z-1], x[z], x[z+1]}
Y_z = {y[z-1], y[z], y[z+1]}
```

同一个随机空间变换同步应用到三张图像及其 mask，避免邻层在坐标上错位。标注分支使用
真实 mask；未标注分支由 EMA teacher 对三张切片预测，随后执行 hard argmax 和逐切片
2D largest connected component。训练代码当前不会使用 U 的真实标签。

### 3.2 离散 slice profile

令离散轴向偏移 `k in {-1,0,+1}`，profile 参数为宽度 `sigma` 与虚拟中心偏移 `phi`：

```math
\tilde w_k = \exp\left[-\frac{1}{2}\left(\frac{k-\phi}{\sigma}\right)^2\right],
\qquad
w_k = \frac{\tilde w_k}{\sum_j \tilde w_j}.
```

当前固定采样范围为：

```text
sigma ~ Uniform(0.45, 0.85)       # slice units
phi   ~ Uniform(-0.25, 0.25)      # slice units
```

所有 `w_k >= 0` 且和为 1。该实现是 **acquisition-inspired three-tap
profile**，不是由每台扫描仪标定出的真实 PSF。当前 H5 数据没有可靠的 slice thickness、
gap、vendor 或 profile 元数据，因此论文不能声称物理校准。

### 3.3 图像与 occupancy 的成对重采集

图像观测为：

```math
\tilde x_z = A_h(X_z) = \sum_{k=-1}^{1} w_k x_{z+k}.
```

对类别 `c`，组织 occupancy 为：

```math
q_{z,c}(u) = A_h(Y_z)_c(u)
= \sum_{k=-1}^{1} w_k\,\mathbf{1}[y_{z+k}(u)=c].
```

因此每个像素满足 `q_c in [0,1]`、`sum_c q_c=1`。若邻层解剖完全一致，目标仍是
one-hot；若 profile 覆盖了组织转变，目标自然变成 fractional occupancy。

早期 SliceEq 在生成 occupancy 后重新 `argmax`，测试约为 `0.832603`。配置完整的
SliceEqOcc 保留连续 occupancy，开发结果约为 `0.844566`。这一关联支持“有效信息可能在
fractional magnitude 中，而不只是 hard class change”的动机；但 SliceEqOcc 同时加入了
re-acquired-L view 并改变 student-view 组成，正式论文仍需 matched batch/BN、component
和 multi-seed 消融后才能作纯 occupancy 的因果归因。

### 3.4 Soft segmentation loss

对 student logits `s`、概率 `p=softmax(s)` 和 occupancy `q`，soft CE 为：

```math
L_{CE}^{soft} = -\frac{1}{N}\sum_{i,c}q_{i,c}\log p_{i,c}.
```

当前 soft Dice 使用平方分母：

```math
L_{Dice}^{soft}
= 1-\frac{1}{C}\sum_c
\frac{2\sum_i p_{i,c}q_{i,c}+\epsilon}
{\sum_i p_{i,c}^2+\sum_i q_{i,c}^2+\epsilon}.
```

重采集分支损失为二者均值。它不是把 soft target 再转回 hard mask。

### 3.5 完整训练目标与 batch36

loader batch 为 24：12 个 labeled center 和 12 个 unlabeled center。1k warmup 后，student
一次前向看到 36 个视图：

```text
12 original-L       -> center hard GT
12 reacquired-L     -> exact fractional occupancy
12 reacquired-U     -> teacher-derived fractional occupancy
```

因此额外 12 个视图是 SliceEqOcc 的 exact-GT acquisition anchor，不是随意把 batch 从
24 调成 36。监督损失为：

```math
L_{sup}=\frac{1}{2}\left[
L_{hard}(f(x_L),y_L)+L_{soft}(f(A_h(X_L)),q_L)\right].
```

未标注损失为：

```math
L_U=L_{soft}(f(A_h(X_U)),q_U),
\qquad
L=L_{sup}+\lambda(t)L_U.
```

`lambda(t)` 沿用原 scaffold 的 sigmoid ramp，最大约为 0.5。student 每步更新一次；EMA
teacher 由 student state 做指数滑动更新，`ema_decay=0.99`。为保持既有 baseline 变量
控制，teacher 在现有实验中保持 train mode。

## 4. OAAC 的原理

OAAC 只把上式 U 输入替换为 `G_eta(A_h(X_U))`，target `q_U` 完全不变：

```math
L_U^{OAAC}=L_{soft}\left(
f(G_\eta(A_h(X_U))),\ q_U\right).
```

对每个非恒定样本，先以其最小值 `m` 和跨度 `r=max-min` 归一化：

```math
v=(x-m)/r.
```

依次采样并应用：

```math
g\sim U(-0.20,0.20),\quad v_g=v^{\exp(g)},
```

```math
c\sim U(-0.15,0.15),\quad
v_c=\mu_g+\exp(c)(v_g-\mu_g),
```

```math
b\sim U(-0.10,0.10),\quad
G_\eta(x)=m+r(v_c+b).
```

gamma 和 contrast 系数始终为正，所以该组合保持像素强度顺序；没有 clipping、noise、
blur、resize、mask、mixing 或坐标变化。它可能略微超出原图强度范围，训练日志会记录
越界比例。常量图像保持原样。

OAAC 使用独立 CUDA generator `seed=1339`，不推进 profile、teacher/student dropout 或
global RNG。训练入口在正式 30k 前自动做 CUDA smoke，检查：

- L 输入/target 未被 OAAC 修改；
- U hard target 与 occupancy 逐位不变；
- U 图像变化非零、有限且保持强度顺序；
- 同 seed 可复现、不同 seed 有活动差异；
- parent-visible CPU/CUDA RNG 未变化。

需要准确披露：L 的输入、GT 和损失定义不变，但全部 36 个 student 视图共享一次
BatchNorm forward，因此强 U 视图可能改变 L activation 的批统计。这是 OAAC 输入干预的
自然结果，不能写成“L activation bit-identical”。

## 5. 与已有工作的关系

| 工作 | 借鉴的思想 | 与本方法的关键区别 |
|---|---|---|
| [Mean Teacher, NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) | EMA teacher 提供稳定一致性目标 | 本仓库保留该 scaffold，但目标经过 acquisition operator 形成 occupancy |
| [BCP, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html) | 医学分割中的 teacher/student 与 hard pseudo-label 基线 | 本地 baseline 已去掉 Copy-Paste，故只能称“BCP-derived EMA scaffold”，不能把 SliceEq 结果写成完整 BCP 的结果 |
| [mixup, ICLR 2018](https://openreview.net/forum?id=r1Ddp1-Rb) | 输入和标签的凸组合 | SliceEq 不混合任意样本，而是用同一病例的真实相邻 MRI、受限 slice profile 和空间 tissue occupancy 表达一次轴向采集 |
| [ICT, IJCAI 2019](https://www.ijcai.org/proceedings/2019/504) | 对输入插值，并让 prediction/target 插值保持一致 | ICT 混合任意 U 样本及 teacher prediction；SliceEq 用同一病人的真实相邻 MRI、采集 profile 和像素级组织 occupancy，服务于 through-plane acquisition 语义 |
| [Inter-Slice Augmentation, ECAI 2020](https://doi.org/10.3233/FAIA200314) | 在连续医学切片间同时插值图像和分割标签 | 该工作是监督式 frame interpolation；SliceEq 使用受限 three-tap profile，在 EMA SSL 的 exact/pseudo 两支形成并保留 fractional occupancy |
| [TCSM, TNNLS 2021](https://doi.org/10.1109/TNNLS.2020.2995319) | 医学分割中的 transformation-consistent self-ensembling | TCSM 处理旋转/翻转等可逆坐标变换；SliceEq 处理非可逆的轴向混合，target 本身发生改变 |
| [SoftSeg, MedIA 2021](https://doi.org/10.1016/j.media.2021.102038) | 医学边界和 partial volume 不应总被二值化 | SoftSeg 讨论一般 soft GT/回归式训练；SliceEq occupancy 由每次采集 profile 从 binary exact/pseudo masks 动态计算，并继续使用 soft CE+Dice |
| [PV-SynthSeg](https://arxiv.org/abs/2004.10221) 与 [SynthSeg](https://arxiv.org/abs/2107.09559) | 用成像/分辨率模拟提高 MRI 分割鲁棒性 | 它们从生成模型模拟多对比度/分辨率；本方法不合成整幅 MRI，而是在半监督 teacher/student 中对真实相邻信号和 mask 做成对局部重采集 |
| [MR Slice Profile Estimation, IPMI 2021](https://link.springer.com/chapter/10.1007/978-3-030-78191-0_9) | 真实 MR slice-selection profile 未知，可从图像内部统计估计 | 当前 SliceEq 只采用固定范围的 index-space Gaussian 近似，未执行每病例 profile 估计或 scanner calibration，因此只称 acquisition-inspired |
| [FixMatch, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html) / [UniMatch, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html) | weak target 监督 strong student view | OAAC 采用这一训练范式，但强视图必须在 target-changing SliceEq 配对完成后构造 |
| [AugSeg, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html) | 连续强度空间中的半监督增强 | gamma/contrast/brightness 不是本文创新；OAAC 只是扩大已经正确配对的 SliceEq U student 视图覆盖 |
| [SWA, UAI 2018](https://auai.org/uai2018/proceedings/papers/313.pdf) / [Model Soups, ICML 2022](https://proceedings.mlr.press/v162/wortsman22a.html) | 同架构权重平均可稳定解并保持单模型推理 | 仅可作为不改架构的训练/选模技巧，不能写成 SliceEq 或 OAAC contribution |

因此不能声称：首次 Mean Teacher、首次输入/目标同步插值、首次 equivariant
consistency、首次 partial-volume soft label、首次 weak-to-strong 或首次 MRI 强度增强。

可守住的差异是这些条件的交集：

> 在 2D 半监督 MRI 分割中，用真实同病例相邻切片和同一随机 through-plane profile，
> 将 exact/teacher hard masks 映射为 acquisition-dependent fractional occupancy，并且只在
> 训练期使用邻层；OAAC 再按“先 target-changing acquisition、后 target-invariant
> appearance”的顺序扩大学生扰动覆盖。

## 6. Motivation、Contribution 与创新点的推荐写法

### Motivation

1. 半监督分割增强通常假设 target invariant。
2. through-plane MRI acquisition 会积分相邻解剖，因此该假设在 partial-volume 边界失效。
3. 中心 hard mask 或重采集后再 argmax 会丢掉采集形成的组织比例。
4. 需要一个与图像 forward operator 同步的 target operator，同时保持部署时的 2D 简洁性。

### 核心 contributions

1. **问题定义**：指出非可逆 through-plane acquisition 下 label-invariant consistency 的
   target-semantics mismatch，并把半监督一致性写成 acquisition-aligned target
   construction。这里的 \(A_h\) 是 stack-to-slice 的非可逆前向算子，并非群作用意义下的
   equivariance。
2. **Paired SliceEq operator**：同一离散 slice profile 作用于真实相邻 MRI 信号与
   exact/teacher-derived one-hot masks，生成空间对齐的 fractional occupancy。
3. **训练/部署解耦**：用 exact re-acquired L anchor 和 pseudo re-acquired U risk 训练
   2D student；相邻层和所有 acquisition/OAAC 操作均为 training-only，推理图不变。

### OAAC 的安全定位

如果 OAAC 在新 untouched/hidden evaluation 和 matched multi-seed controls 中保持提升，
可写成一个次级设计原则：

> target-changing acquisition 必须先与 occupancy 配对；之后才能组合 target-invariant
> monotonic appearance augmentation。

OAAC 不应单独列为“新的 photometric augmentation”或与 SliceEq 并列的主要创新。

## 7. 当前实验事实

| 方法 | 选择/证据类型 | Dice | 说明 |
|---|---:|---:|---|
| 去 Copy-Paste 的 BCP-derived EMA scaffold | 用户确认开发范围 | 0.78--0.80 | 不是原始完整 BCP |
| SliceEq hard predecessor | 用户确认开发结果 | 0.832603 | occupancy 最后 argmax |
| SliceEqOcc | 已接受的开发 checkpoint | 0.844566 | fractional occupancy；正式无偏主表仍待确认 |
| OAAC iter27000 | test-selected development oracle（已检查多个 checkpoint 后取 test 最大值） | 0.849538 | Jaccard 0.740985，HD95 3.554760，ASD 1.868299 |

OAAC 完整训练的 best validation 为 `0.834863@23800`，final 为
`0.831964@30000`。`iter27000` 的 validation 是 `0.828406`。用户已澄清 27k 是从多个
被查询的 test checkpoint 中按最高 Dice 选出，因此它不是 validation-selected primary，
而是 post-hoc test-selected development maximum。相对 SliceEqOcc oracle 的表观 Dice 差
为 `+0.004972`，并有 7/10 病例提高；这些统计以两个事后选定的 checkpoint 为条件，不能
支持无偏增益、显著性或 SOTA 声明。

训练机制确实活跃：146 条日志的 `active_samples` 全为 1，平均归一化强度变化为
`0.055178`。这与 SCPO/ADU 的近恒等负结果不同。

## 8. 不改变模型架构的进一步优化

以下步骤不改变架构，可用于内部开发，但不能在已参与 checkpoint selection 的同一 test
上产生新的论文 primary evidence：

1. **补齐 validation-selected 结果**：可测试一次 `unet_best_model.pth` /
   `iter_23800_dice_0.8349.pth`，但结果仍是 development-only，不能恢复 test 独立性。
2. **一次固定的同轨迹权重平均**：可在内部固定等权平均
   `iter_24000.pth + iter_27000.pth + iter_30000.pth`，并使用相同 OAAC 训练分布重新估计
   BatchNorm statistics；由于方案是在看到 27k test 表现后提出，它在当前 test 上仍是探索。
3. **若允许完整重训**：统一对 SliceEqOcc 和 OAAC 使用 poly/cosine LR 与 late SWA，
   并在新的 untouched outer fold 或 external test 上一次性确认。当前 self-training 全程
   LR=0.01，高位 val 振荡说明存在稳定化空间；只给 OAAC 更好的 schedule 会混淆归因。

不建议：从当前 `.pth` 低 LR 续训（没有 optimizer/EMA state）、继续按 test 搜
checkpoint/soup、TTA、NMS、阈值或 seed。`0.849538` 可按统一三位小数格式显示为
`0.850`，但必须同时标明它是 test-selected development oracle，不能借舍入暗示达到无偏
论文目标。

## 9. CVPR 文章构建路线

### 推荐标题

```text
SliceEq: Acquisition-Aligned Fractional Occupancy for
Semi-Supervised MRI Segmentation
```

若最终只有前列腺数据，应在标题或摘要中收窄到 prostate MRI。

### 论文论证顺序

1. 现有 SSL 强增强依赖 target invariance。
2. through-plane acquisition 是反例：它改变观测的组织组成。
3. SliceEq 用同一 operator 重新定义 image 和 target。
4. fractional occupancy 比 hard argmax 保留更多 acquisition semantics。
5. exact L anchor + pseudo U occupancy 将该机制引入 Mean Teacher。
6. OAAC（若确认）展示 target-changing 与 target-invariant 变换的正确组合顺序。
7. 所有 volumetric 信息均为 training-only，部署仍为原 2D U-Net。

### 必需实验矩阵

至少使用相同 pretrain、teacher policy、batch/BN、更新次数和 checkpoint rule：

```text
B0-24          原 no-Copy-Paste EMA scaffold
B0-36          compute/BN matched ordinary-view control
ImgOnly-36     只重采集图像，target仍为center
SliceHard-36   图像/mask同profile，但occupancy argmax
SliceEqOcc-36  完整fractional occupancy
OAAC-36        SliceEqOcc + ordered U appearance（若作为最终版）
```

核心方法至少 3 seeds，主方法和最强控制最好 5 seeds；另需 L-only/U-only 因子消融、第二
label budget、一个未用于开发的 anisotropic MRI 数据集或 challenge hidden test。OAAC 若
进入最终方法，公平矩阵中的关键 controls 应使用同一 photometric recipe，避免把通用强
增强收益归给 occupancy。

### 统计与指标

- 保存每 seed、每患者 Dice/Jaccard/physical HD95/ASD/NSD；
- primary contrast 使用 patient-level paired permutation 或 Wilcoxon；
- 报告均值/中位数差、win rate 和 patient/seed hierarchical bootstrap CI；
- 恢复真实 spacing 后再把距离写成 mm；现有距离是 voxel-index distance；
- test 只在规则冻结后调用一次。

### 图表建议

1. **主图**：中心 hard target 与 through-plane 混合图像失配；SliceEq 同权得到 image 和
   occupancy。
2. **方法图**：L exact、U teacher、paired operator、OAAC 顺序，以及 training-only
   neighbors / 2D inference。
3. **机制图**：fractional support 位置、hard SliceEq 与 SliceEqOcc 的梯度/误差差异。
4. **实验图**：不同 profile severity、axial position、protocol/spacing 分层结果。

## 10. 局限性与公开声明

1. 当前 Gaussian profile 以 slice units 定义，没有 scanner thickness/spacing 校准。
2. 三 tap 只能近似有限 support；端点通过复制邻层处理。
3. U teacher 采用 hard argmax + 逐切片 2D LCC，丢弃 posterior confidence 和完整 3D
   一致性。
4. teacher 保持 train mode；BN/dropout 与 batch composition 是现有 scaffold 的一部分。
5. 36-view batch 包含额外 exact-GT-derived L view，必须用 B0-36 控制计算与 BN 混杂。
6. 当前 PROMISE12 test 已参与开发，不能继续作为无偏主结果。
7. 当前 OAAC 入口锁定了本项目服务器路径与 pretrain hash，属于 artifact-reproduction
   entry；公开复现者需要准备相同数据合同/权重，或另建不改变算法的可移植配置入口。
8. 不上传数据、患者图像、H5/NIfTI、checkpoint、训练日志或服务器凭据。

## 11. 运行入口

训练服务器先执行：

```bash
cd /home/aiteam/zhengtaoma/CoDA
python -m unittest tests.test_sliceeq_oaac tests.test_sliceeq_oaac_contract -v

cd code
python -u train_sliceeq_occ_oaac.py
```

严格测试：

```bash
python -u test_sliceeq_occ_oaac.py \
  --checkpoint_path ../model/SliceEqOccOAAC_PROMISE12_7_labeled/self_train/unet/unet_best_model.pth \
  --auto_find_checkpoint False \
  --save_result False
```

更短的运行说明见 [`SLICEEQ_OCC_OAAC_README.md`](SLICEEQ_OCC_OAAC_README.md)，论文框架见
[`../research/paper/sliceeq_occ_cvpr_outline_2026-08-13.md`](../research/paper/sliceeq_occ_cvpr_outline_2026-08-13.md)。
