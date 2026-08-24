# SliceEqOcc-OAAC-Strong CVPR 论文提纲（中文版）

> 最终方法的完整原理、演化逻辑、数学定义、参考工作边界以及 7/11-label 与 MM-WHS 实验解释，见 [`sliceeq_occ_oaac_strong_method_analysis_zh.md`](sliceeq_occ_oaac_strong_method_analysis_zh.md)。

状态：本提纲将当前验证集选择的最优配置 `SliceEqOccOAACStrong` 设为最终方法。论文中简称 **SliceEqOcc-OAAC**，其中 **Strong** 表示冻结的 OAAC 1.25 倍外观扰动配置，而不是新的网络结构。PROMISE12 是当前开发数据集；MM-WHS 2017 的 MRI 子集是预设的跨器官、多类别外部验证数据集。MM-WHS 结果尚未产生，所有相关数值与结论均保留为待填项。

## 1. 暂定标题

首选标题：

**SliceEqOcc-OAAC：面向半监督 MRI 分割的采集对齐分数占据与有序外观一致性**

英文标题建议：

**SliceEqOcc-OAAC: Acquisition-Aligned Fractional Occupancy with Ordered Appearance Consistency for Semi-Supervised MRI Segmentation**

如果 MM-WHS 的外部验证未完成或未得到正向结果，则收窄为：

**SliceEqOcc-OAAC：面向半监督前列腺 MRI 分割的采集对齐分数占据学习**

## 2. 一句话核心论点

对于会改变目标语义的不可逆层间采集扰动，应先用同一个前向算子成对重新采集图像与组织占据目标，再对学生图像施加不改变空间坐标和占据语义的外观扰动；这种固定顺序同时保证目标正确性和半监督强视图覆盖。

## 3. 论文核心叙事

论文不应写成“我们叠加了两个数据增强”，而应围绕一个统一问题展开：

1. 现有半监督分割增强通常假设图像受扰动后标签语义不变。
2. 层间 MRI 重新采集会整合相邻组织，因此该假设失效。
3. SliceEqOcc 用同一个采集算子生成重新采集图像和分数占据目标，修复目标语义错配。
4. 在目标语义被正确建立后，OAAC 才对无标注学生图像施加保持坐标不变的单调外观扰动，扩大弱到强学习的有效覆盖。
5. 所有相邻切片、采集算子和 OAAC 都只在训练期出现；测试仍是原始单切片 2D U-Net。
6. PROMISE12 用于建立方法与完成开发；MM-WHS MRI 用于验证该原则能否跨器官、跨类别数和跨采集中心泛化。

## 4. 摘要草稿

半监督医学图像分割通常假设，增强后的图像仍可由原始硬分割目标监督。然而，层间 MRI 采集会在有限切片支持范围内整合相邻解剖组织，使重新采集图像的目标语义随采集过程发生变化。为此，我们提出 **SliceEqOcc-OAAC**，一种仅在训练阶段使用的采集对齐增强框架。首先，SliceEqOcc 将同一个随机切片剖面算子同时作用于真实相邻 MRI 切片及其精确或教师生成的组织掩码，得到成对的重新采集图像与算子诱导的分数占据目标；随后，OAAC 仅对重新采集后的无标注学生图像施加保持坐标不变的单调外观变换，而不再修改已经形成的占据目标。该顺序把会改变目标语义的采集变换与目标不变的外观变换明确分开。最终的 Strong 配置联合使用冻结的1.25倍 gamma、contrast 和 brightness 范围，不增加网络层、可学习参数或推理开销。在当前 PROMISE12 七例标注、固定 seed=1337 的开发协议上，验证集选择的 SliceEqOcc-OAAC-Strong checkpoint 达到 `0.851960` Dice；该数值属于单次开发结果，仍需独立数据评测确认。**[待完成 MM-WHS 后补充：在固定病例划分和固定 seed=1337 的 MM-WHS MRI 低标注协议上的 Dice/NSD/HD95，以及相对 SliceEqOcc 和匹配光度增强对照的结果。]** 该研究表明，半监督 MRI 增强的关键不仅是增加扰动强度，还应按照“先形成正确采集目标、后扩展目标不变外观”的语义顺序组合变换。

在完成 MM-WHS 外部独立评测和协议匹配对照之前，不使用“state-of-the-art”或“显著优于现有方法”等表述。本文固定使用单一随机种子，不把 seed 方差作为证据；这一限制必须在 Limitations 中明确披露。

## 5. 引言论证结构

### 第1段：半监督医学分割与增强范式

- 医学分割标注昂贵，教师—学生框架通过一致性学习利用无标注数据。
- Copy-Paste、混合增强、弱到强视图、遮挡、位移和频域变换不断扩大无标注训练信号。
- 这些方法大多默认：增强改变观测，但硬目标保持不变，或者只需要进行离散标签组合。

### 第2段：层间采集导致目标不再保持不变

- 对单调灰度变化或正确对齐的平面内几何变换，标签不变通常成立。
- 对有限层厚、层间偏移和相邻切片混合，观测本身包含多个解剖平面的组织贡献。
- 重新采集图像不再等价于中心切片，继续使用中心硬掩码会在部分容积边界、器官出现/消失区域形成结构性错配。
- 因此，MRI 层间增强需要同时重新定义图像与监督目标，而不是只增强输入。

### 第3段：SliceEqOcc 的解决方案

- 训练时读取真实 `z-1,z,z+1` 相邻切片。
- 采样同一个离散切片剖面，并以相同归一化权重作用于图像强度和 one-hot 组织掩码。
- 有标注分支使用精确 GT，无标注分支使用 EMA 教师硬伪掩码。
- 保留混合后的分数占据，不再通过 argmax 抹去部分容积语义。
- 原始有标注中心切片继续提供硬解剖锚点。

### 第4段：为什么还需要 OAAC，以及为什么必须有顺序

- SliceEqOcc 修复了目标语义，但未标注学生分支的外观覆盖仍主要由一次重新采集决定。
- 直接在采集之前或同时加入任意强增强，容易再次破坏图像—目标对应关系。
- OAAC 遵循固定组合顺序：先执行目标会变化的 `A_h`，再执行目标不变的 `G_eta`。
- `G_eta` 只改变重新采集后的无标注学生图像，不改变坐标、教师目标或组织占据率。
- Strong 不是更大的网络，而是通过验证集选择后冻结的1.25倍外观强度。

### 第5段：开发证据与外部验证目标

- 描述性开发轨迹：BCP 衍生且去除 Copy-Paste 的 EMA 基线约为 `0.78--0.80`，硬目标 SliceEq 为 `0.832603`，SliceEqOcc 为 `0.844566`，最终 SliceEqOcc-OAAC-Strong 的验证集选择结果为 `0.851960`。
- 上述数值来自开发过程且选择规则不完全相同，不能直接作为严格的组件因果增益。
- OAAC 强度响应为：1.00倍验证 `0.834863`、1.25倍 `0.836475`、1.50倍 `0.835796`。因此选择1.25倍作为验证集局部最优，停止继续调参。
- **待验证：**在 MM-WHS MRI 上使用冻结方法参数，检验该方法能否从二分类前列腺分割迁移到七结构心脏分割。

### 第6段：贡献

1. **采集对齐的目标语义。** 我们指出不可逆层间采集破坏半监督增强的标签不变假设，并将一致性重新定义为图像与精确/伪监督通过同一采集算子的成对映射。该关系是算子对齐，不是群等变性。
2. **分数占据监督。** 我们提出 SliceEqOcc，以离散切片剖面联合重新采集相邻真实图像和组织掩码，显式保留由采集产生的空间分数占据，并结合原始硬解剖锚点进行半监督学习。
3. **有序采集—外观组合。** 我们提出 OAAC 的组合原则：先执行会改变目标的采集算子，再仅对无标注学生图像执行坐标保持的单调外观变换。最终 Strong 配置冻结为1.25倍联合强度，模型结构和推理图保持不变。
4. **待外部证据支持的泛化主张。** 在 PROMISE12 前列腺 MRI 与 MM-WHS 多中心全心 MRI 上，以低标注协议验证跨器官、多类别和采集差异下的泛化能力。

## 6. 模型构建、思想来源与方法演化

### 6.1 网络与半监督骨架从哪里来

最终方法没有设计新的分割主干。学生网络和教师网络均采用2D U-Net：编码器提取多尺度上下文，解码器通过跳跃连接恢复空间细节。教师参数由学生参数的指数滑动平均更新。模型构建分为预训练和自训练两阶段：

1. 使用少量有标注病例完成 U-Net 预训练，并保存网络与优化器状态；
2. 自训练阶段严格加载同一个预训练状态；
3. 学生网络接收有标注与无标注视图并反向更新；
4. EMA 教师只生成无标注伪目标，不接收梯度；
5. 无标注一致性权重按既定 ramp 逐渐增加。

这一骨架参考 [U-Net（MICCAI 2015）](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/) 和 [Mean Teacher（NeurIPS 2017）](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)。本文不把 U-Net、EMA 教师或普通伪标签学习列为创新。

### 6.2 从 BCP 到本地基线

项目最初采用 [BCP（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html) 的教师—学生训练框架。BCP 原方法通过有标注与无标注区域的双向 Copy-Paste 缩小经验分布差异。本项目删除了 Copy-Paste，因此本地基线不能继续简称“BCP”，其准确名称是：

> BCP 衍生的 EMA 硬伪标签自训练框架（Copy-Paste removed）。

本地基线使用中心切片，EMA 教师生成 hard argmax 伪标签并执行逐切片2D最大连通域处理，学生以硬交叉熵和 Dice 学习。该基线在当前 PROMISE12 开发协议上约为 `0.78--0.80` Dice。

### 6.3 SliceEq：为什么开始使用真实相邻切片

对 MRI 体数据的观察表明，中心切片的组织形状会沿轴向连续变化，而真实采集的有限层厚会混合邻层信号。因此，我们从“只扰动单张图像”转向“利用真实相邻切片模拟重新采集”。SliceEq v1：

- 读取 `z-1,z,z+1` 三张真实切片；
- 对三张切片执行同一个空间变换；
- 采样三抽头 Gaussian slice profile；
- 用同一组权重混合图像和 one-hot 掩码；
- 最后对混合掩码 argmax，仍使用硬目标。

这一阶段受到两类工作的启发：[Inter-Slice Augmentation（ECAI 2020）](https://ecai2020.eu/papers/547_paper.pdf) 说明可以利用相邻医学切片生成中间样本；[ICT（IJCAI 2019）](https://www.ijcai.org/proceedings/2019/504) 已提出让插值输入与插值教师预测保持一致。SliceEq 的区别不是“首次插值”，而是把插值限制为同一病例的轴向邻层及受约束切片剖面。硬目标 SliceEq 的开发 Dice 约为 `0.832603`。

### 6.4 SliceEqOcc：为什么必须保留 fractional occupancy

对 SliceEq v1 的进一步分析发现，`argmax(A_h(Y))` 几乎总会退化回中心硬标签。图像已经发生邻层混合，但目标中的部分容积信息被 argmax 删除，这正是新的图像—目标失配。SliceEqOcc 因此做出两个互相关联的改变：

1. 对有标注与无标注重新采集分支都保留完整分数占据 `q=A_h(onehot(Y))`；
2. 新增有标注重新采集视图，以精确 GT occupancy 教会网络如何响应该采集算子，同时保留原始硬中心切片作为解剖锚点。

这解释了 batch36 的来源：它不是随意增大 batch，而是由 `12 原始L + 12重新采集L + 12重新采集U` 三个方法分支构成。相对 SliceEq 的24个学生视图，新增的12个视图是 SliceEqOcc 精确 occupancy 分支所必需的训练输入。其额外监督/BN效应仍需 B0-36 和 L-only/U-only 消融加以区分。SliceEqOcc 的开发 Dice 为 `0.844566`。

这一思想与 [SynthSeg](https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424/) 的分辨率/部分容积模拟相关，但边界不同：SynthSeg 通过生成模型和域随机化训练跨对比度、跨分辨率模型；SliceEqOcc 使用真实相邻 MRI 切片，在半监督教师—学生框架中构造与具体随机采集算子一致的 exact/pseudo occupancy。

### 6.5 OAAC：为什么在采集之后加入外观增强

SliceEqOcc 主要在轴向变化和分数边界处提供新语义，覆盖范围稀疏。SAQ、CAP、DA、APTNA、ADU、SCPO 等实验表明，继续修补剖面采样、增加同源目标或清理极少量伪标签不能稳定提升。因此优化方向转为：在不改变 occupancy 语义的前提下，让所有无标注重新采集视图获得更广的外观覆盖。

[UniMatch（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html) 表明弱视图伪目标监督强学生视图是有效的半监督分割范式；[AugSeg（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html) 强调强度增强的设计本身会显著影响半监督性能。OAAC 参考这一思想，但针对 SliceEqOcc 增加了语义顺序约束：

```text
先：A_h(image, occupancy)        # 会改变目标，必须成对
后：G_eta(A_h(image))            # 坐标保持，只改变 U 图像外观
目标：A_h(occupancy) 保持不变
```

因此，OAAC 的创新点不是 gamma、contrast 或 brightness，而是对“目标会变化的采集算子”和“目标不变的外观算子”进行有序组合。

### 6.6 Strong 配置如何确定

原 OAAC 使用 scale1.00，之后只进行联合强度而非组件网格搜索：

- scale1.00：验证 `0.834863`；
- scale1.25：验证 `0.836475`；
- scale1.50：验证 `0.835796`。

1.25在相同验证规则下最高，因此冻结为 Strong。1.50虽然开发 Dice 为 `0.852059`，比1.25的 `0.851960` 仅高 `0.000099`，但验证更低、逐病例仅2/10获胜且次要指标变差，所以不替换1.25。最终方法由此确定为 `SliceEqOccOAACStrong`。

### 6.7 参考工作与继承边界总表

| 参考工作 | 借鉴内容 | 本文没有声称的内容 | SliceEqOcc-OAAC 的具体区别 |
|---|---|---|---|
| U-Net, MICCAI 2015 | 2D编码器—解码器分割主干 | 新网络结构 | 推理主干不变，贡献全部位于训练目标和增强语义 |
| Mean Teacher, NeurIPS 2017 | EMA教师与一致性学习 | 新教师更新算法 | 教师伪掩码被用于构造轴向 fractional occupancy |
| BCP, CVPR 2023 | 医学分割教师—学生训练入口 | 原始BCP或Copy-Paste贡献 | 删除Copy-Paste，仅保留BCP衍生EMA骨架 |
| ICT, IJCAI 2019 | 输入与教师预测同步插值 | 首次插值一致性 | 使用同一病例真实邻层和受约束MRI切片剖面 |
| Inter-Slice Augmentation, ECAI 2020 | 相邻医学切片可生成中间样本 | 首次邻层插值 | 面向半监督 exact/pseudo occupancy，而非只扩充有标注样本 |
| SynthSeg, MedIA | 分辨率、对比度和部分容积模拟 | 首次采集/分辨率模拟 | 不生成整幅合成图像；使用真实邻层和随机成对算子 |
| UniMatch, CVPR 2023 | 弱目标监督强学生视图 | 首次弱到强一致性 | 强视图只能在 target-changing 采集配对之后形成 |
| AugSeg, CVPR 2023 | 连续外观扰动空间 | 新gamma/contrast/brightness算子 | 强调与采集算子的固定顺序及 occupancy 不变合同 |
| [MM-WHS，MedIA 2019](https://doi.org/10.1016/j.media.2019.101537) | 多中心全心MRI外部基准 | 未完成实验前不得声称外部泛化 | 当前仅为预设外部验证，结果待完成 |

## 7. 相关工作结构

### 7.1 半监督医学图像分割

讨论 Mean Teacher、BCP、ICT、FixMatch/UniMatch、AugSeg、ABD、beta-FFT 以及不确定性方法。必须明确承认：ICT 已经同步插值无标注输入和教师预测；FixMatch/UniMatch 已经建立弱目标监督强学生视图；gamma、contrast 和 brightness 也不是本文原创。本文区别应限定为：

- 同一受试者的真实解剖相邻切片；
- 受约束的不可逆层间采集算子；
- 精确/伪掩码经同一算子产生的空间分数占据；
- 先采集配对、后外观增强的语义顺序；
- 不变的单切片 2D 推理路径。

本地基线删除了 Copy-Paste，因此正文中称为“BCP 衍生的 EMA 硬伪标签框架”，原始 BCP 作为独立公开对比方法。

### 7.2 采集感知增强与部分容积效应

讨论 PV-SynthSeg、SynthSeg、MR 切片剖面估计、分辨率/层厚模拟和 Inter-Slice Augmentation。应承认既有研究已经插值相邻医学图像及其标签。本文研究空缺限定为：在 2D 半监督框架中，用同一个受约束采集算子构造真实相邻切片图像及精确/教师伪分数占据监督。

### 7.3 弱到强一致性与光度增强

讨论 FixMatch、UniMatch、AugSeg 以及医学分割中的外观一致性。OAAC 的 gamma、contrast、brightness 本身不构成创新；可辩护的内容是它们在非标签保持采集算子之后的固定位置，以及只作用于未标注学生图像的目标安全组合。

### 7.4 软标签、歧义与分数占据

严格区分：

- 启发式标签平滑；
- 标注歧义；
- 模型或教师不确定性；
- 由给定采集算子确定性生成的组织分数占据。

SliceEqOcc 处理最后一种现象。不得把 occupancy softness 称为教师不确定性。

## 8. 方法部分

### 8.1 问题定义

定义有标注体数据 `D_L`、无标注体数据 `D_U`、学生网络 `f_theta` 和 EMA 教师 `f_xi`。训练样本以2D中心切片为单位，但可访问三切片堆栈；测试时只输入单张中心切片。

### 8.2 三切片训练样本与教师伪掩码

对中心位置 `z`，构造：

```text
X_z = {x[z-1], x[z], x[z+1]}
Y_z = {y[z-1], y[z], y[z+1]}
```

同一个随机空间变换同步作用于三张图像及其掩码。有标注分支使用真实掩码；无标注分支由 EMA 教师对三张切片预测，执行 hard argmax 和逐切片2D最大连通域处理。教师目标停止梯度，训练代码不得访问无标注样本的真实标签。

### 8.3 离散切片剖面重新采集

令轴向偏移 `k in {-1,0,+1}`，剖面参数为宽度 `sigma` 与虚拟中心偏移 `phi`：

```math
\tilde w_k = \exp\left[-\frac{1}{2}\left(\frac{k-\phi}{\sigma}\right)^2\right],
\qquad
w_k = \frac{\tilde w_k}{\sum_j \tilde w_j}.
```

最终方法冻结为：

```text
sigma ~ Uniform(0.45, 0.85)     # slice units
phi   ~ Uniform(-0.25, 0.25)    # slice units
```

图像重新采集为：

```math
\tilde x_z=A_h(X_z)=\sum_{k=-1}^{1}w_kx_{z+k}.
```

组织分数占据为：

```math
q_z=A_h(\operatorname{onehot}(Y_z))
=\sum_{k=-1}^{1}w_k\operatorname{onehot}(y_{z+k}).
```

当前算子是**受采集启发的三抽头离散剖面**，不是由扫描仪元数据标定的真实 PSF；端点采用现有复制策略。PROMISE12 与 MM-WHS 均不得在缺少可靠层厚/spacing 来源时使用“物理校准”表述。

### 8.4 分数占据监督

`argmax(q_z)` 会把混合目标重新压缩为近乎不变的硬标签，丢失部分容积信息。SliceEqOcc 保留完整 `q_z`，使用软交叉熵与平方分母软 Dice：

```math
L_{soft}(p,q)=\frac{1}{2}\left[L_{CE}^{soft}(p,q)+L_{Dice}^{soft}(p,q)\right].
```

目标的 softness 由同一采集算子和邻层组织共同决定，具有空间结构，不是全边界统一平滑。

### 8.5 半监督目标函数

有标注损失由原始硬锚点和精确重新采集占据组成：

```math
L_L=\frac{1}{2}\left[
L_{hard}(f_\theta(x_z^L),y_z^L)
+L_{soft}(f_\theta(A_h(X_z^L)),q_z^L)
\right].
```

无标注 SliceEqOcc 损失为：

```math
L_U=L_{soft}(f_\theta(A_h(X_z^U)),q_z^U),
```

其中 `q_z^U` 来自 EMA 教师硬伪掩码堆栈经过同一个 `A_h` 后的分数占据。

### 8.6 OAAC：先采集、后外观

OAAC 只替换无标注学生输入：

```math
L_U^{OAAC}=L_{soft}\left(
f_\theta(G_\eta(A_h(X_z^U))),q_z^U
\right).
```

其中 `G_eta` 是逐样本、保持坐标不变且对非恒定输入单调的组合变换：

1. gamma 变换；
2. 围绕变换后均值的 contrast 缩放；
3. 相对原图强度跨度的 brightness 平移。

最终 Strong 配置固定为：

```text
log-gamma       ~ Uniform(-0.25,   0.25)
log-contrast    ~ Uniform(-0.1875, 0.1875)
brightness/span ~ Uniform(-0.125,  0.125)
application probability = 1.0
```

OAAC 使用独立随机数流。它不接收目标作为函数输入，不改变空间坐标，也不修改 `q_z^U`。有标注输入、目标、损失定义及权重保持不变；但由于36个学生视图共享一次含 BatchNorm 的前向，强无标注视图仍会影响联合批统计，论文不得声称有标注分支激活完全逐位不变。

### 8.7 完整目标、训练批次与 EMA

完整目标为：

```math
L=L_L+\lambda(t)L_U^{OAAC},
```

其中 `lambda(t)`、EMA decay、warmup、优化器与学习率均沿用锁定基线。预热结束后学生网络一次前向包含36个视图：

- 12个原始有标注中心切片；
- 12个有标注重新采集切片；
- 12个经重新采集后再执行 OAAC 的无标注强视图。

### 8.8 推理与复杂度

- 测试时仅输入一张2D切片；
- 不读取相邻切片；
- 不执行 `A_h` 或 `G_eta`；
- 不新增网络层或可学习参数；
- 使用与基线相同的 U-Net 推理图。

论文需补充实测训练时间、峰值显存、训练 FLOPs 和单切片推理延迟。

## 9. 实验设计

### 9.1 数据集角色

#### PROMISE12：方法开发与主要机制分析

- 二分类前列腺 MRI 分割。
- 当前协议：35例训练、5例验证、10例已参与开发的测试病例；7例有标注训练体。
- 追加的标注预算实验固定使用 `train.list` 前11例（`train_slices.list` 前306 slices）作为有标注集。该预算必须重新执行相同10k监督预训练，不能复用7例预训练权重；随后直接使用已经冻结的 SliceEq profile、OAAC-Strong 1.25倍参数、seed=1337及其余训练规则，不在11例结果上重新调参。
- 该测试集已经用于多次 checkpoint 和方法开发，因此现有数值统一称为开发结果。
- 用于完成主因果消融、参数敏感性、机制可视化及固定 seed=1337 的方法比较。

#### MM-WHS 2017：预设外部泛化验证

正式名称使用 **MM-WHS 2017（Multi-Modality Whole Heart Segmentation）**，数据组成与任务定义以[官方挑战页面](https://zmiclab.github.io/zxh/0/mmwhs/)及其[基准论文](https://doi.org/10.1016/j.media.2019.101537)为准。主实验只使用其 MRI 子集，不把 CT 与 MRI 混入同一主结果。官方数据包含60例3D心脏 MRI，其中20例训练数据具有七个心脏亚结构标注，40例属于官方测试数据。

MM-WHS 在论文中承担三项验证：

1. 从前列腺到全心脏的跨器官泛化；
2. 从二分类到七结构多类别分割的扩展；
3. 在多中心、不同图像质量和不同轴向采集条件下验证采集对齐原则。

### 9.2 MM-WHS 预注册协议

建议在数据准备完成前锁定以下协议：

- 仅使用 MRI 子集作为主要外部验证；CT 仅可作为附录扩展，不参与方法选择。
- 从20例带标注 MRI 训练数据中固定4例作为验证集；剩余16例中使用4例有标注训练病例和12例标签隐藏的无标注病例。
- 所有方法固定使用 `seed=1337`，不进行多随机种子验证，也不根据结果更换 seed。
- 固定病例划分，不根据单次结果更换有标注、无标注或验证病例。
- 所有方法使用相同预处理、U-Net、训练步数、batch、EMA、学习率、验证频率和 checkpoint 规则。
- SliceEq 剖面范围及 OAAC Strong 参数直接从 PROMISE12 冻结迁移，不在 MM-WHS 上重新搜索。
- 主对比至少包含：监督基线、BCP 衍生 EMA、SliceEqOcc、通用光度强增强对照、最终 SliceEqOcc-OAAC-Strong。
- 如果可以使用官方40例 MRI 隐藏测试评测，则在验证规则和最终方法锁定后只提交最终模型；若无法获得官方评测，必须预先建立独立外层测试划分，不能把验证病例同时作为最终测试。
- MM-WHS 是七前景类别任务，模型输出类别数、标签映射和多类别 Dice 实现属于数据集适配，不属于方法改动。

### 9.3 评价指标与统计

两个数据集统一报告：

- Dice（PROMISE12 二分类；MM-WHS 宏平均及逐结构）；
- Jaccard；
- NSD；
- 使用真实 spacing 的物理 HD95/ASD；
- 逐病例结果、均值 ± 标准差和95%置信区间。

主比较采用病例级配对统计，并报告病例 bootstrap 置信区间、逐病例胜率和中位差。本研究固定单一 seed，不报告 seed 方差，也不把病例 bootstrap 解释为优化随机性的替代。MM-WHS 额外报告 LV、RV、LA、RA、MYO、AA 和 PA 的逐结构指标，避免宏平均掩盖薄壁或小结构退化。

### 9.4 主要对比方法

- Supervised-only；
- BCP 衍生 EMA 框架；
- 原始 BCP；
- UniMatch 或协议可复现的近期强基线；
- SliceEq hard；
- SliceEqOcc；
- SliceEqOcc + 普通光度强增强对照；
- **SliceEqOcc-OAAC-Strong（最终方法）**。

### 9.5 核心因果消融

| 方法 | 相邻图像 | 成对目标 | 分数占据 | OAAC | 作用位置 |
|---|---:|---:|---:|---:|---|
| B0 | 否 | 否 | 否 | 否 | 原始输入 |
| B0-36 | 否 | 否 | 否 | 否 | 计算量/BN匹配 |
| ImgOnly | 是 | 否 | 否 | 否 | 仅重新采集图像 |
| SliceHard | 是 | 是 | 否 | 否 | 图像与硬目标 |
| SliceEqOcc | 是 | 是 | 是 | 否 | 图像与分数目标 |
| GenericStrong | 是 | 是 | 是 | 是 | 不区分顺序或普通强增强 |
| SliceEqOcc-OAAC-Strong | 是 | 是 | 是 | 是 | 先采集配对，后仅增强 U 图像 |

另外报告 Occ-L-only、Occ-U-only，区分精确有标注占据与伪占据的贡献。

### 9.6 参数敏感性

OAAC 强度只报告已经完成的三点：

| OAAC 联合强度 | gamma | contrast | brightness/span | 最优验证 Dice | 开发 Dice |
|---:|---:|---:|---:|---:|---:|
| 1.00 | ±0.20 | ±0.15 | ±0.10 | 0.834863 | 0.849538（test-selected oracle） |
| **1.25 Strong** | **±0.25** | **±0.1875** | **±0.125** | **0.836475** | **0.851960（validation-selected）** |
| 1.50 | ±0.30 | ±0.225 | ±0.15 | 0.835796 | 0.852059（validation-selected） |

选择1.25的依据是最高验证 Dice，而不是最高开发测试 Dice。1.50虽然 Dice 多 `0.000099`，但验证更低、仅2/10病例获胜，且 Jaccard、HD95、ASD 变差。因此将1.25称为“验证集选择的局部设置”，不能称为全局最优。

## 10. 图示规划

### 图1：方法核心图

CVPR 正文推荐使用重新排版的清晰版 [`figures/fig_sliceeq_occ_oaac_pipeline_v3.svg`](figures/fig_sliceeq_occ_oaac_pipeline_v3.svg) / [`PDF`](figures/fig_sliceeq_occ_oaac_pipeline_v3.pdf)。该版本将三类训练样本分别在局部完成“图像—目标”配对，再用三条互不交叉的水平箭头送入共享 batch；损失与 EMA 更新均限制在 learner 面板内部。v2 与第一版保留为历史版本和补充说明图。中英文图注与导出命令见 [`figures/SLICEEQ_OCC_OAAC_FIGURE_CAPTION.md`](figures/SLICEEQ_OCC_OAAC_FIGURE_CAPTION.md)。视觉设计参考审计见 [`../literature/figure_style_audit_2026-08-17.md`](../literature/figure_style_audit_2026-08-17.md)。

四个连续面板：

1. 传统增强：扰动图像仍配中心硬掩码，产生边界错配；
2. 层间采集：`z-1,z,z+1` 经剖面权重形成重新采集图像；
3. SliceEqOcc：同一权重形成分数占据目标；
4. OAAC：在成对采集完成后，仅增强 U 学生图像，测试路径仍为单张2D图像。

图中突出两类算子：`A_h` 会改变目标，必须成对；`G_eta` 不改变目标，只能后置。

### 图2：完整训练流程

- L 中心硬锚点；
- L 精确掩码成对重新采集；
- U EMA 硬伪掩码堆栈；
- U 图像/占据成对重新采集；
- U 图像后置 OAAC；
- 软 CE + 软 Dice、一致性 ramp 和 EMA 更新；
- 虚线标出所有 training-only 路径。

### 图3：目标语义与外观解耦可视化

- 中心硬掩码、硬重新采集目标和分数占据热图；
- 采集前图像、重新采集图像和 OAAC 强图像；
- 显示 OAAC 改变强度但不改变 occupancy；
- 展示 acquisition-active 边界上的稀疏高梯度区域。

### 图4：跨数据集泛化

- PROMISE12 前列腺与 MM-WHS 心脏的代表性切片；
- 二分类到七结构的预测示例；
- 按数据集或心脏结构绘制 SliceEqOcc 与最终方法的 Dice/NSD 改变量；
- 如果 spacing 可用，补充按层间 spacing 分层的鲁棒性曲线。

## 11. 表格规划

### 表1：PROMISE12 主结果

行：Supervised-only、BCP 衍生框架、原始 BCP、UniMatch、近期方法、SliceEq hard、SliceEqOcc、SliceEqOcc-OAAC-Strong。

列：Dice、Jaccard、NSD、物理 HD95、物理 ASD、训练 FLOPs、推理延迟。固定 `seed=1337`，报告病例均值、病例标准差、病例级95%置信区间和逐病例结果。

### 表2：MM-WHS MRI 外部结果

行：监督基线、BCP 衍生 EMA、SliceEqOcc、普通强光度增强、SliceEqOcc-OAAC-Strong。

列：Mean Dice、LV、RV、LA、RA、MYO、AA、PA、NSD、HD95。所有结果标注训练标签预算、固定 `seed=1337` 和病例划分。

### 表3：因果消融

使用第9.5节矩阵，增加有效学生 batch、相邻切片、成对目标、fractional L/U、OAAC、训练成本等列。

### 表4：参数与设计选择

- hard 与 fractional occupancy；
- L-only、U-only 与 Full；
- OAAC 1.00/1.25/1.50；
- OAAC 正确顺序与普通光度增强对照；
- SAQ、SCPO、ADU 等负向设计放在附录。

## 12. 主张—证据账本

| 拟提出的主张 | 所需证据 | 当前状态 |
|---|---|---|
| 最终方法是 SliceEqOcc-OAAC-Strong | 1.00/1.25/1.50 使用相同验证规则比较 | 已支持；1.25验证最高 |
| 分数占据优于硬目标 SliceEq | 固定 seed 下的 SliceHard 与 SliceEqOcc 匹配对照、逐病例分析和外部复现 | 开发结果正向；匹配对照待完成 |
| 增益不是额外 batch/GT 视图造成 | B0-36、L-only/U-only | 待完成 |
| 成对采集优于只增强图像 | Full 对比 ImgOnly/TargetOnly | 待完成 |
| OAAC 收益来自正确组合顺序 | 固定 seed 下 OAAC 对比 SliceEqOcc、普通强光度和顺序对照 | 单次开发正向；匹配对照待完成 |
| OAAC Strong 不改变推理成本 | 相同推理图、参数量及实测延迟 | 图结构已确认；实测待完成 |
| 方法可跨器官和多类别泛化 | MM-WHS MRI 冻结参数评测 | 已预设；实验待完成 |
| 方法对采集差异更稳健 | MM-WHS 多中心结果及 spacing/协议分层 | 待完成 |
| 方法达到 SOTA | 协议匹配的公开方法与独立测试；单 seed 限制需透明披露 | 当前不支持 |

## 13. 实验结果表述规范

- `0.851960`：称为“PROMISE12 单随机种子、验证集选择的开发结果”，不是最终无偏测试结果。
- `0.849538`：称为“事后测试选择的开发 oracle”，不能作为 OAAC 主结果。
- `0.852059`：是1.50倍配置的验证集选择开发结果；不能因为数值略高就替代验证更好的1.25倍配置。
- `0.844566`：称为 SliceEqOcc 的开发结果；其 checkpoint 选择曾使用开发测试反馈。
- 不同 checkpoint 选择规则下的数值只作描述性轨迹，不写成严格因果增益。
- 现有旧距离指标称为“体素索引 HD95/ASD”，修复真实 spacing 后才能使用 `mm`。
- 当前剖面称为“受采集启发的离散切片剖面”，不称为扫描仪真实 PSF。
- MM-WHS 在结果产生前统一写“计划验证”或“待完成”，不得预写提升。

## 14. 附录规划

- 完整数学推导、伪代码和权重构成合法占据分布的证明；
- PROMISE12 与 MM-WHS 的数据预处理、病例划分、标签映射和无标注标签防火墙；
- 所有超参数、独立随机数流及1000 iter周期 checkpoint 保存规则；
- 完整逐病例与逐结构统计，以及固定 seed=1337 的复现配置；
- 端点复制与切片剖面敏感性；
- OAAC 1.00/1.25/1.50强度曲线；
- SAQ 负结果及尾部截断解释；
- SCPO 近恒等干预负结果；
- CAP、DA、APTNA、ADU 等探索性负结果只作为设计诊断，不包装为贡献；
- 训练环境、GPU、版本、哈希、checkpoint 规则和复现实验命令。

## 15. 最终写作主线

最终论文应形成以下闭环：

1. 标签不变并不适用于不可逆层间 MRI 采集。
2. SliceEqOcc 通过同一算子成对重新采集图像与组织目标，把错误的中心硬标签改为采集诱导的分数占据。
3. 正确目标形成后，OAAC 才扩大无标注学生图像的外观覆盖；这一先后顺序避免再次破坏目标语义。
4. Strong 是验证选择的最终1.25倍配置，而不是额外模块或更强网络。
5. 当前 PROMISE12 单 seed 开发证据支持该最终配置；正式结论依赖匹配对照与 MM-WHS 外部验证，单 seed 是明确限制。
6. MM-WHS MRI 以冻结参数验证跨器官、七结构和多中心泛化，决定论文能否使用通用“半监督 MRI 分割”标题。
7. 所有体数据与增强都只在训练阶段使用，部署仍是原始单切片 2D U-Net。
