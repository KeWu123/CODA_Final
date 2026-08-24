# SliceEqOcc-OAAC-Strong 最终方法分析与原理说明

> 最终方法：`SliceEqOcc-OAAC-Strong`，即 SliceEqOcc 加 1.25 倍 Ordered Acquisition-Appearance Consistency（OAAC）。
>
> 当前协议：PROMISE12 固定 `seed=1337`，以 7 labeled cases 为主实验，并追加 11 labeled cases；预设使用 MM-WHS MRI 进行冻结参数的跨器官、多类别验证。本项目不进行多随机种子实验，因此论文应明确披露固定单随机种子设置。

## 1. 方法的核心思想

常规半监督分割通常假设，增强后的图像仍可使用原来的标签监督。但 MRI 的 through-plane 采集会整合相邻解剖平面的信号，使重新采集图像中的组织组成发生变化。此时继续使用中心切片硬标签，会造成图像与目标语义不匹配。

本方法分两步解决这一问题：

1. **SliceEqOcc**：用同一个随机轴向采集算子，同时重新采集真实相邻 MRI 切片和对应组织掩码，得到成对的重新采集图像与分数占据目标；
2. **OAAC-Strong**：在上述目标正确形成后，只对无标注学生图像执行保持空间坐标和强度次序的外观增强，occupancy 目标不再改变。

```text
真实相邻切片与组织状态
        │
        │  同一个采集算子 A_h
        ├──────────────────────────┐
        ▼                          ▼
重新采集图像 A_h(X)        分数占据目标 A_h(Y)
        │                          │
        │  OAAC G_eta（仅 U 图像） │ target 保持不变
        ▼                          ▼
学生 f(G_eta(A_h(X_U)))  ── soft CE + soft Dice

测试：单张 2D MRI → 原 U-Net → 分割结果
```

相邻切片、EMA teacher、重新采集、fractional occupancy 和 OAAC 全部只在训练时使用，不增加网络层、可学习参数或推理输入。

## 2. 研究问题与 Motivation

### 2.1 常规一致性学习的隐含条件

teacher–student 半监督学习常使用：

```math
f_\theta(T_s(x_U))\approx\hat y_U,
```

其中 teacher 在弱视图上形成伪标签，student 在强视图上拟合同一目标。这要求增强 `T_s` 不改变每个像素对应的组织语义。

翻转、同步空间变换或单调灰度变化通常可以满足这一条件，但轴向重新采集并不是标签保持变换。有限层厚、切片间隔和轴向偏移会使一个观测包含 `z-1,z,z+1` 多个解剖平面的贡献。器官边界像素可能同时包含前景与背景，因此重新采集图像不再严格对应中心硬掩码。

### 2.2 Fractional occupancy 不是不确定性

本文必须区分：

- **annotation uncertainty**：标注者对真实边界存在分歧；
- **model uncertainty**：teacher 不确定类别；
- **acquisition occupancy**：给定相邻组织和采集权重后，一个观测像素确实由多个组织共同组成。

SliceEqOcc 建模第三项。其 soft target 是采集算子的确定性结果，不是按预测熵做标签平滑。

### 2.3 为什么还需要 OAAC

SliceEqOcc 修复了 target-changing acquisition 下的目标错误，但它提供的新语义主要集中在稀疏的轴向变化和部分容积区域。SAQ、CAP、DA、APTNA、ADU、SCPO 等后继实验表明，继续调整 profile、叠加同源目标或修补极少量伪标签不能稳定提升。

因此优化方向转向：在不破坏 occupancy 的前提下，让全部无标注重新采集视图获得更广的外观覆盖。OAAC 的关键不在 gamma、contrast 或 brightness 本身，而在固定语义顺序：

```text
先执行 target-changing A_h：同时作用于 image 与 target
再执行 target-invariant G_eta：只作用于 U student image
```

## 3. 从 baseline 到最终方法的演化

### 3.1 本地 baseline

项目从 BCP 代码框架出发，但删除了 Copy-Paste。因此论文中的准确名称应为：

> BCP-derived EMA hard-pseudo-label self-training scaffold without Copy-Paste。

它保留 2D U-Net、EMA teacher、hard argmax 伪标签、逐切片 2D largest connected component 和 CE+Dice。其开发 Dice 约为 `0.78--0.80`。U-Net、EMA teacher 和普通伪标签训练均不是本文创新。

### 3.2 SliceEq

SliceEq 开始使用同一病例的真实 `z-1,z,z+1` 切片，并用同一随机 profile 混合图像与掩码。但 v1 最后执行：

```math
\tilde y_z=\operatorname{argmax}_c A_h(Y_z)_c,
```

使目标重新退化为近似中心硬标签，丢失 fractional magnitude。SliceEq 的开发 Dice 约为 `0.832603`。

### 3.3 SliceEqOcc

SliceEqOcc 保留完整 occupancy 分布，并新增 exact-GT 重新采集 L 分支，使网络先从真实标签学习 acquisition–occupancy 对应关系。其开发 Dice 约为 `0.844566`。

这一结果支持 fractional occupancy 的动机，但 SliceEqOcc 同时增加了 re-acquired-L view，并改变 student batch/BN 组成，因此仍需 B0-36、SliceHard-36、L-only/U-only 等匹配消融完成因果归因。

### 3.4 OAAC-Strong

OAAC 在 SliceEqOcc 之后仅增强 U student image。1.00、1.25、1.50 三个联合强度使用相同验证规则比较，1.25 的验证指标最高，因此冻结为最终 Strong 配置。1.25 是当前验证集局部最优，不是数学意义或无限参数空间中的全局最优。

## 4. SliceEqOcc 的数学原理

### 4.1 三切片训练单元

对中心位置 `z`：

```math
X_z=\{x_{z-1},x_z,x_{z+1}\},\qquad
Y_z=\{y_{z-1},y_z,y_{z+1}\}.
```

同一个随机空间变换同步应用于三张图像及掩码。L 分支使用真实 GT；U 分支由 EMA teacher 分别预测三张切片，再执行 hard argmax 和逐切片 2D LCC。训练不读取 U 的真实标签。

### 4.2 随机三抽头切片剖面

对轴向偏移 `k∈{-1,0,+1}`：

```math
\sigma\sim\mathcal U(0.45,0.85),\qquad
\phi\sim\mathcal U(-0.25,0.25),
```

```math
\tilde w_k=\exp\left[-\frac12
\left(\frac{k-\phi}{\sigma}\right)^2\right],
\qquad
w_k=\frac{\tilde w_k}{\sum_j\tilde w_j}.
```

所有 `w_k≥0` 且和为 1。当前 profile 使用 slice units，是 acquisition-inspired Gaussian approximation，不是由每台扫描仪的 slice thickness、gap 或真实 PSF 标定得到。

### 4.3 成对重新采集

图像：

```math
\tilde x_z=A_h(X_z)=\sum_{k=-1}^{1}w_kx_{z+k}.
```

对类别 `c` 和像素 `u`，occupancy 为：

```math
q_{z,c}(u)=A_h(Y_z)_c(u)
=\sum_{k=-1}^{1}w_k\mathbf 1[y_{z+k}(u)=c].
```

所以：

```math
0\le q_{z,c}(u)\le1,
\qquad\sum_cq_{z,c}(u)=1.
```

若三层组织一致，`q` 仍为 one-hot；若相邻层发生组织变化，`q` 自然成为分数占据。图像与目标必须复用完全相同的 `w`，这是方法的核心配对合同。

## 5. Fractional occupancy loss

设 student logits 为 `s`，`p=softmax(s)`，目标为 `q`。

```math
\mathcal L_{CE}^{soft}
=-\frac1N\sum_{i,c}q_{i,c}\log p_{i,c}.
```

```math
\mathcal L_{Dice}^{soft}
=1-\frac1C\sum_c
\frac{2\sum_i p_{i,c}q_{i,c}+\epsilon}
{\sum_i p_{i,c}^2+\sum_i q_{i,c}^2+\epsilon}.
```

```math
\mathcal L_{soft}
=\frac12\left(\mathcal L_{CE}^{soft}
+\mathcal L_{Dice}^{soft}\right).
```

实现不会把 `q` 再次转换成 hard mask。

## 6. 为什么 student batch 是 36

loader batch 为 24：12 labeled centers 与 12 unlabeled centers。1000 iter warmup 后，一次 student forward 包含：

| 分支 | 数量 | 输入 | 目标 |
|---|---:|---|---|
| 原始 L | 12 | `x_L` | 中心硬 GT `y_L` |
| 重新采集 L | 12 | `A_h(X_L)` | exact occupancy `q_L` |
| 重新采集 U | 12 | `G_eta(A_h(X_U))` | teacher occupancy `q_U` |

```text
student batch = 12 original-L + 12 reacquired-L + 12 reacquired-U = 36
```

新增的 12 张图像是 exact-GT acquisition anchor，不是任意增大 batch。原始 L 保持原生解剖能力，重新采集 L 提供无伪标签噪声的 occupancy 教学，重新采集 U 扩展到无标注数据。

同时 batch36 会改变计算量与 BN 组成，所以论文必须设置 compute/BN-matched control。

## 7. OAAC-Strong 的精确定义

OAAC 只替换 U student 输入：

```math
\mathcal L_U^{OAAC}
=\mathcal L_{soft}\left(
f_\theta(G_\eta(A_h(X_U))),A_h(\hat Y_U)
\right).
```

对每个非恒定图像：

```math
m=\min(x),\quad r=\max(x)-\min(x),\quad v=(x-m)/r.
```

Strong 1.25 倍范围为：

```math
g\sim\mathcal U(-0.25,0.25),\qquad
x_g=m+r\,v^{\exp(g)},
```

```math
c\sim\mathcal U(-0.1875,0.1875),\qquad
x_c=\mu_g+\exp(c)(x_g-\mu_g),
```

```math
b\sim\mathcal U(-0.125,0.125),\qquad
G_\eta(x)=x_c+b\,r.
```

gamma 和 contrast 系数始终为正，因此保持样本内像素强度次序。实现不进行坐标变换、noise、blur、mixing 或 clipping，也不接收 target 作为输入。OAAC 使用独立 CUDA generator `seed=1339`，不推进 profile、teacher/student dropout 或 global RNG。

OAAC 只用于 U，是因为 L 分支承担原始硬解剖锚点和 exact occupancy 教学，而 weak-to-strong 外观鲁棒性主要服务于 U consistency。

## 8. 完整训练目标

```math
\mathcal L_L^{native}
=\mathcal L_{hard}(f_\theta(x_L),y_L),
```

```math
\mathcal L_L^{occ}
=\mathcal L_{soft}(f_\theta(A_h(X_L)),q_L),
```

```math
\mathcal L_L
=\frac12(\mathcal L_L^{native}+\mathcal L_L^{occ}),
```

```math
\mathcal L_U
=\mathcal L_{soft}(f_\theta(G_\eta(A_h(X_U))),q_U),
```

```math
\mathcal L=\mathcal L_L+\lambda(t)\mathcal L_U.
```

`λ(t)` 沿用 baseline 的 sigmoid consistency ramp，最大约为 0.5。学习率、优化器、EMA 模式和 `ema_decay=0.99` 均保持 baseline 设置。

## 9. 训练阶段与伪代码

前 1000 iter 走 baseline identity path：只用原始中心切片计算 L hard loss，不启用 SliceEq、occupancy、OAAC 或 U loss。此后：

```text
for each post-warmup iteration:
    Yhat_U = LCC2D(argmax(EMA_teacher(X_U[z-1:z+1])))

    h_L, h_U = independently sample SliceEq profiles
    Xtilde_L, q_L = A_hL(X_L), A_hL(Y_L)
    Xtilde_U, q_U = A_hU(X_U), A_hU(Yhat_U)

    Xstrong_U = G_eta(Xtilde_U)          # OAAC-Strong 1.25x
    logits = student([x_L, Xtilde_L, Xstrong_U])  # 36 views

    L_L = 0.5 * (HardLoss(x_L,y_L) + SoftLoss(Xtilde_L,q_L))
    L_U = SoftLoss(Xstrong_U,q_U)
    L = L_L + lambda(t) * L_U

    update student once
    update EMA teacher once
```

teacher 保持 train mode，以与现有 baseline 控制一致。验证仍每 200 iter 进行，`unet_best_model.pth` 的规则不变；最终 Strong 独立版本只把普通周期权重由每 3000 iter 保存改为每 1000 iter 保存，不改变 best 保存机制。

## 10. 为什么方法可能有效

1. **修复系统性目标错误**：不是只制造更多图像，而是让重新采集图像拥有与 forward operator 一致的目标；
2. **保留 argmax 删除的信息**：hard class change 很少，但 fractional magnitude 可持续表达相邻组织贡献；
3. **exact L anchor 抑制漂移**：`q_L` 来自 GT，先建立可靠 acquisition–occupancy 映射；
4. **原始 L 保留测试域能力**：部署仍是原生单切片，因此不能只训练重新采集状态；
5. **OAAC 提供高覆盖扰动**：SliceEqOcc 负责 target correctness，OAAC 负责 representation robustness；
6. **语义顺序避免二次失配**：只有先形成正确 occupancy，后续 appearance 扰动才能安全复用该目标；
7. **推理零增量**：所有三切片和增强路径均为 training-only。

## 11. 与参考工作的关系

| 参考工作 | 借鉴内容 | 本文区别 |
|---|---|---|
| [U-Net, MICCAI 2015](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/) | 2D 编码器—解码器 | 网络结构不变，创新在训练目标和增强语义 |
| [Mean Teacher, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) | EMA teacher | teacher masks 还要经过采集算子形成 occupancy |
| [BCP, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html) | 医学半监督代码骨架 | 本地 baseline 删除 Copy-Paste，不继承其核心 Copy-Paste 贡献 |
| [mixup, ICLR 2018](https://openreview.net/forum?id=r1Ddp1-Rb) | 输入和标签同步凸组合 | 本文限于同病例真实相邻切片和轴向 profile |
| [ICT, IJCAI 2019](https://www.ijcai.org/proceedings/2019/504) | 插值输入应匹配插值 teacher target | 本文在 MRI acquisition occupancy 空间执行该原则 |
| [Inter-Slice Augmentation, ECAI 2020](https://doi.org/10.3233/FAIA200314) | 相邻医学切片可生成中间观测 | 本文用于 EMA SSL exact/pseudo 双分支并保留 occupancy |
| [TCSM, TNNLS 2021](https://doi.org/10.1109/TNNLS.2020.2995319) | 医学分割变换一致性 | 本文处理会改变目标组成的非可逆轴向混合，而非只做可逆坐标变换 |
| [SoftSeg, MedIA 2021](https://doi.org/10.1016/j.media.2021.102038) | partial volume 不应总被二值化 | 本文 soft target 由每次随机采集 profile 动态形成 |
| [PV-SynthSeg](https://arxiv.org/abs/2004.10221) / [SynthSeg](https://arxiv.org/abs/2107.09559) | 成像与分辨率模拟 | 本文不生成完整 MRI，而是重采集真实相邻图像与 masks |
| [FixMatch, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html) / [UniMatch, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html) | weak target 监督 strong student | OAAC 强视图必须在 target-changing acquisition 配对后形成 |
| [AugSeg, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html) | 连续外观增强设计 | gamma/contrast/brightness 非本文原创；本文强调固定顺序与 target 合同 |

因此不能声称首次 Mean Teacher、首次输入/标签插值、首次 soft label、首次 weak-to-strong 或首次强度增强。可辩护的新意是：

> 在训练期使用真实同病例相邻 MRI，通过同一受约束 through-plane operator 成对构造图像与 exact/teacher-derived fractional occupancy；随后仅在目标正确形成后执行 target-invariant U appearance augmentation，同时保持单切片 2D 推理。

## 12. 推荐的论文 Motivation 与 Contributions

### Motivation

1. 半监督强增强依赖 target invariance；
2. MRI through-plane acquisition 会整合相邻组织，是该假设的反例；
3. 中心硬标签或重采集后 argmax 会删除组织占据信息；
4. 正确增强应先同步构造观测和目标，再扩展目标不变的外观变化；
5. 希望在不引入 3D 推理的情况下利用体数据中的采集信息。

### Contributions

1. **采集语义问题定义**：指出非可逆 through-plane mixing 对常规 label-invariant consistency 的破坏；
2. **SliceEqOcc**：用同一离散 profile 成对重采集真实相邻 MRI 与 exact/teacher masks，显式保留 fractional occupancy，并联合原始硬解剖锚点训练；
3. **OAAC 有序组合**：提出“先 target-changing acquisition、后 target-invariant appearance”的组合原则，最终 Strong 设置不改变网络和推理成本。

OAAC 不能写成“新 gamma/contrast/brightness 算法”，其贡献是变换类别的语义解耦与顺序约束。

## 13. 当前结果的正确理解

| 方法 | 开发结果 | 证据口径 |
|---|---:|---|
| BCP-derived EMA，无 Copy-Paste | 约 0.78--0.80 | 本地基础框架，不是完整 BCP |
| SliceEq hard | 0.832603 | 丢弃 fractional magnitude |
| SliceEqOcc | 0.844566 | fractional occupancy 开发结果 |
| OAAC 1.00 | 0.849538 | test-selected development oracle |
| **OAAC-Strong 1.25** | **0.851960** | 当前最终方法，validation-selected 单 seed 开发结果 |
| OAAC 1.50 | 0.852059 | 验证更低且次要指标更差，不替换 1.25 |

`0.851960` 应表述为 PROMISE12 固定 `seed=1337`、validation-selected 的开发结果，不是已经确认的无偏 SOTA。1.25 是三点局部敏感性实验的验证集选择结果，不称为全局最优。

本项目不进行多随机种子验证。论文应报告逐病例/逐结构结果、病例标准差和病例级置信区间，并明确“固定单随机种子”限制，不能把单次结果解释为对初始化普遍稳定。

## 14. 7-label、11-label 与 MM-WHS

### PROMISE12 7-label

- `train.list` 前 7 例、`train_slices.list` 前 191 slices；
- 使用匹配的 7-label supervised pretrain；
- 是当前方法和 Strong 参数选择的来源。

### PROMISE12 11-label

- `train.list` 前 11 例、`train_slices.list` 前 306 slices；
- 必须重新执行相同 10k supervised pretrain，不能复用 7-label 权重；
- SliceEq profile、OAAC 1.25、seed、LR、EMA、ramp、batch36、30k 和验证规则全部冻结；
- 不根据 11-label 结果重新调参。

11-label 用于分析监督预算增加后方法是否仍有价值，而不是第二轮方法搜索。

### MM-WHS MRI

- 作为跨器官、多类别外部验证；
- 预先固定病例划分、低标注预算和 `seed=1337`；
- 从 PROMISE12 冻结迁移 SliceEq profile 与 OAAC-Strong 参数；
- occupancy 公式可自然扩展到任意 `C≥2`，但当前 PROMISE12 入口锁定 `num_classes=2`，需独立实现多类别数据入口；
- 报告 Mean Dice、逐结构 Dice、NSD 与 physical HD95；
- 结果产生前只写“计划验证”，不能预写提升。

## 15. 必需消融与公平性控制

即使固定单 seed，以下匹配实验仍是解释方法所需：

| 对照 | 目的 |
|---|---|
| B0-24 | 原本地 EMA baseline |
| B0-36 | 排除额外视图、计算量和 BN 影响 |
| ImgOnly-36 | 证明只改图像会产生 target mismatch |
| SliceHard-36 | 比较 hard argmax 与 fractional occupancy |
| Occ-L-only / Occ-U-only | 区分 exact anchor 与 pseudo occupancy |
| SliceEqOcc | OAAC 的直接父方法 |
| GenericStrong | 区分普通光度增强与有序组合 |
| OAAC 1.00/1.25/1.50 | 报告已完成的局部参数敏感性 |

所有对照使用相同 pretrain、teacher policy、训练步数、验证规则和 `seed=1337`。

## 16. 局限性

1. profile 使用 slice units，未按真实 thickness、spacing、gap 或 scanner PSF 标定；
2. 三抽头只能近似有限 support，体积端点采用邻层复制；
3. U teacher 使用 hard argmax + 逐切片 2D LCC，丢弃 posterior confidence 和完整 3D 一致性；
4. teacher 保持 train mode，BN/dropout 行为继承 baseline；
5. batch36 同时改变 exact-GT-derived views 与 BN 组成，需要匹配对照；
6. OAAC 强 U 与 L 共享 student BN 前向，只能说 L 输入、GT 和损失定义不变，不能说 L activation 逐位不变；
7. PROMISE12 已参与开发，现有数值不能作为完全无偏测试；
8. 固定单 seed 无法估计初始化方差；
9. MM-WHS 多类别外部验证尚未完成。

## 17. 实现文件映射

| 功能 | 文件 |
|---|---|
| 最终 Strong 训练入口 | [`code/train_sliceeq_occ_oaac_strong.py`](../../code/train_sliceeq_occ_oaac_strong.py) |
| 隔离 SliceEqOcc 父版本 | [`code/train_sliceeq_occ_h7_15_base.py`](../../code/train_sliceeq_occ_h7_15_base.py) |
| SliceEq profile 与成对重新采集 | [`code/utils/sliceeq.py`](../../code/utils/sliceeq.py) |
| Fractional occupancy loss | [`code/utils/sliceeq_occ.py`](../../code/utils/sliceeq_occ.py) |
| OAAC-Strong 变换 | [`code/utils/sliceeq_oaac_strong.py`](../../code/utils/sliceeq_oaac_strong.py) |
| 最终测试入口 | [`code/test_sliceeq_occ_oaac_strong.py`](../../code/test_sliceeq_occ_oaac_strong.py) |
| 11-label 预训练 | [`code/pretrain_promise12_label11.py`](../../code/pretrain_promise12_label11.py) |
| 11-label 最终训练 | [`code/train_sliceeq_occ_oaac_strong_label11.py`](../../code/train_sliceeq_occ_oaac_strong_label11.py) |
| 11-label 运行说明 | [`docs/SLICEEQ_OCC_OAAC_STRONG_LABEL11_README.md`](../../docs/SLICEEQ_OCC_OAAC_STRONG_LABEL11_README.md) |
| 中文论文提纲 | [`sliceeq_occ_cvpr_outline_2026-08-13_zh.md`](sliceeq_occ_cvpr_outline_2026-08-13_zh.md) |
| 最终流程图 | [`figures/fig_sliceeq_occ_oaac_pipeline_v3.svg`](figures/fig_sliceeq_occ_oaac_pipeline_v3.svg) |

## 18. 最容易误解的十点

1. 本地 baseline 不是完整 BCP；
2. SliceEq 不是任意样本 mixup；
3. fractional occupancy 不是 teacher uncertainty；
4. OAAC 创新不是三个普通灰度算子本身；
5. 正确顺序是先 `A_h(image,target)`，再 `G_eta(image only)`；
6. batch36 新增的 12 张是 exact re-acquired-L 方法分支；
7. 相邻切片只在训练使用，测试仍为单张 2D；
8. 1.25 是验证集局部最优，不是全局最优；
9. 11-label 必须重新 pretrain；
10. MM-WHS 当前是冻结参数的计划验证，尚无结果。

## 19. 论文方法章节建议结构

1. Problem formulation：普通 consistency 的 target-invariance 假设；
2. Paired slice-profile re-acquisition：定义 `A_h`；
3. Acquisition-derived fractional occupancy：定义 `q=A_h(Y)` 与 soft loss；
4. Exact and pseudo occupancy learning：解释 L/U 分支与 batch36；
5. Ordered acquisition–appearance consistency：定义 OAAC 顺序与 Strong 参数；
6. Overall objective and algorithm：总损失与伪代码；
7. Training-only volumetric context：强调单切片 2D inference；
8. Scope and limitations：说明 acquisition-inspired、单 seed 和外部验证状态。

整篇文章的逻辑不应写成“相邻层增强加光度增强”，而应写成：

> 不同增强对目标语义的影响不同。through-plane acquisition 会改变组织组成，必须成对重新定义 image 与 target；只有在该目标形成正确后，才能安全加入 target-invariant appearance perturbation。SliceEqOcc-OAAC-Strong 将二者按语义顺序组合，并把全部体积信息限制在训练期。
