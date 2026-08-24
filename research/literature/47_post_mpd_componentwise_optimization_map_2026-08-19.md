# MPD 之后的逐模块优化地图

日期：2026-08-19  
当前冻结父方法：`SliceEqOcc-OAAC-Strong-MPD`  
当前 PROMISE12 开发结果：Dice `0.854573`（iter 29000，按项目约定的最高已测试 checkpoint 选择）

## 1. 审计目标

MPD 已经证明了一条有效路线：不增加网络模块，不改变 EMA、优化器、损失或推理，而是把原本启发式的三切片 profile 分布改造成一个由训练集精确 occupancy 约束、全局冻结的稳健分布。本审计沿用同一原则，逐个检查最终方法中仍然存在的启发式分布或近似环节，寻找第二个可独立归因的优化点。

本轮不重新打开以下已经冻结的基础设施：学习率、EMA train-mode、teacher/student 架构、batch 中 L/U 数量、consistency ramp、验证规则、单切片推理和 MPD profile 分布。也不采用多随机种子筛选。

## 2. 逐模块结论

| 模块 | 当前实现 | 机会 | 风险/碰撞 | 结论 |
|---|---|---:|---|---|
| 三切片 profile 分布 | MPD 全局 q，成对作用图像与 occupancy | 已成功利用 | 再调 grid、moment、RFI 会变成同一开发集过度搜索 | **冻结** |
| 病人/轴向切片采样 | 在 L/U stream 内按 slice index 抽样 | 高 | 通用 hard mining/分层采样已有大量工作，必须限定为 acquisition opportunity | **下一优先方向** |
| OAAC 参数联合分布 | gamma、contrast、brightness 独立均匀，Strong scale=1.25 | 中 | 自动增强、teacher-guided augmentation 已拥挤 | 第二候选，不能与切片采样首跑叠加 |
| 端点邻层支持 | 越界时复制 endpoint slice | 中低 | 只影响少量端点，且 MPD 已在该实现上设计 q | 可作物理一致性小改，不优先冲 Dice |
| teacher 伪标签形成 | train-mode EMA，hard argmax，逐层 2D LCC | 低 | ADU、SCPO、posterior/拓扑修复已无稳定收益；通用 uncertainty 很拥挤 | **关闭** |
| fractional occupancy loss | soft CE + squared soft Dice；L exact、U pseudo | 低 | acquisition residual 已有强梯度；DA/AP-TNA 等曾稀释主信号 | **冻结** |
| 栈内几何增强 | 三层共享旋转/翻转/resize | 低 | 属通用增强政策，不是 acquisition-specific 核心 | 不作为下一贡献 |
| LR/EMA/ramp/batch/架构 | baseline-derived 固定协议 | 不开放 | 用户明确要求固定 | **禁止改动** |
| 推理/TTA/后处理 | 原生 2D 单切片 | 不属于训练方法深化 | 容易变成测试侧技巧 | 不用于方法涨点 |

## 3. 第一优先：Patient–Axial Acquisition-Risk Sampling

### 3.1 当前隐含问题

当前 `TwoStreamBatchSampler` 直接在 labeled slice indices 和 unlabeled slice indices 中抽样。于是病例 `p` 被抽到的概率近似正比于其切片数 `N_p`；同一病例中，包含切片最多的轴向区域也贡献更多更新。训练风险实际更接近

\[
\mathbb E_{i\sim\text{uniform slices}}\,\mathbb E_{w\sim q_{\mathrm{MPD}}}
  \ell\big(f(A_wX_i),A_wQ_i\big),
\]

而不是 patient-balanced risk。MPD 的设计目标却显式保护 patient×axial-third 中 acquisition opportunity 较弱的 strata。如果 SGD 仍被长病例和中部切片主导，MPD 在算子分布层面得到的平衡会在数据分布层面被部分抵消。

### 3.2 建议方法

暂命名为 **Patient–Axial Acquisition-Risk Sampling (PARS)**。它不预测“难例”，不读取 U 标签，也不根据当前网络误差在线追逐样本。

1. 维持每个 loader batch 为 `12 L + 12 U`，维持 warmup 后 student 36 views。
2. 对每个 stream 先均匀选择 patient，再按一个全局冻结的轴向 third 分布 `q_z` 选择 first/middle/last third，最后在该 patient-third 内均匀选择 slice。
3. `q_z` 只由前 191 个 labeled-training slices 的 exact occupancy opportunity 设计；目标是提高最弱 patient-third 的预期有效 fractional occupancy，同时保持接近父 slice distribution。
4. 使用与 MPD 相同的安全思想：phase/profile 已冻结，只优化 sampling probability；约束 `KL(q_z||p_z)`、density-ratio cap、entropy floor，并约束期望 image residual / foreground opportunity 不偏离父分布过多。
5. 对 U stream 只使用 case id 和相对轴向位置，绝不读取 U label；L 上设计出的同一个 `q_z` 直接冻结迁移到 U。
6. profile、OAAC、teacher target、loss coefficient、EMA、LR、BN batch size、验证和推理完全不变。

### 3.3 为什么它比普通 hard mining 更适合本文

PARS 的选择量不是 loss、entropy 或模型置信度，而是与本文 forward operator 对齐的 **acquisition opportunity**。它解决的是“哪些 anatomical trajectories 被 paired re-acquisition 风险看见”的问题，而不是泛化地多采困难像素。论文表述必须限制为 patient/axial support of a paired acquisition-occupancy operator。

通用分层采样、难例采样和方差缩减已有明显先例，例如 ARCO 的像素级分层/方差缩减与 PH-Net 的 patch hardness；因此不能声称首次 balanced sampling 或 hard sampling。本文可守的区别是：固定、非模型驱动、patient-balanced 的 axial acquisition-opportunity distribution，并且它与同一 `A_w` 下的 image–fractional-occupancy pairing 联合定义。

### 3.4 预期与停止规则

这是当前作用覆盖率最高、又不改模型的候选：它会改变大量 batch 的病人/轴向组成，而不是只修改极少数像素。预期优先改善 apex/base 或短病例在训练中的覆盖，但不能保证 PROMISE12 Dice 必然提高。

首个实验必须只改 sampler。若没有超过冻结 MPD 父方法，就关闭 patient/third probability、density cap、entropy 和 opportunity proxy 的后续搜索，不继续按测试病例反向调采样权重。

## 4. 第二优先：Robust Appearance Moment Design

OAAC-Strong 已确定合适的总体强度，但仍独立均匀采样三类参数：

\[
(\log\gamma,\log c,\beta)\sim U\times U\times U.
\]

可以沿用 MPD 的“分布设计”思想，在固定 Strong support 上构造一个全局联合分布 `q_a`：

- 保持零均值/符号对称和 Strong 的 expected normalized appearance change；
- 保持 foreground/background intensity ordering、CNR 与 boundary-gradient budget；
- 最大化训练病人和轴向 thirds 的最坏 appearance coverage；
- 设置 density-ratio cap 和 entropy floor，避免退化成几个极端组合；
- 设计完成后冻结，仍只作用于 post-acquisition U image，target 不变。

它不改网络和推理，但创新风险高于 PARS。TeachAugment、iMAS、AugSeg 等已经覆盖 teacher-guided、instance-adaptive 或强增强优化；因此它最多是 OAAC 的稳健联合政策组件，不能作为“首次自动增强”。OAAC scale1.0/1.25/1.5 已经完成局部强度 bracket，不能再把本方向做成 range sweep。

## 5. 第三优先：valid-support endpoint projection

当前邻层越界时复制 endpoint，导致同一个 nominal profile 在 volume 两端具有不同的实际矩：被复制的权重会合并到中心/端点切片。一个小而干净的改法是：对实际可用 support 重新投影权重，使有效 `b` 和 `d` 尽可能接近 MPD 目标矩，同时保持非负、和为 1，并对 image/occupancy 使用同一结果。

该方向物理解释清楚，但预计 Dice 空间有限：端点占比有限，且许多 volume endpoint 是背景；MPD 也已经在当前 clamp 语义下完成训练集设计。它更适合补充材料中的 operator-consistency 分析，而不是下一个主实验。

## 6. 明确关闭的方向

- 不再做 confidence threshold、entropy filtering、第二 teacher、MC dropout、3D-LCC 或 posterior smoothing。它们要么已有本项目负证据，要么会把真实 fractional occupancy entropy 错当伪标签不确定性。
- 不再改 soft CE/Dice 比例、L/U ratio、native-U anchor 或 consistency ramp。此前实验表明这些变化容易稀释已经有效的 acquisition measurement signal。
- 不再搜索 sigma/phase/grid/RFI/moment caps。MPD 已是该模块的最终版本。
- 不改 LR、EMA mode、batch、网络或推理协议。

## 7. 执行顺序

1. 冻结 MPD 父方法，独立实现 **PARS**；首跑只改变 sampler。
2. 只有 PARS 结题后，才决定是否单独测试 Robust Appearance Moment Design；两者首跑绝不叠加。
3. endpoint projection 只在需要进一步完善 operator 定义时考虑。
4. 若前两项均无增益，停止 PROMISE12 组件搜索，直接把最终 MPD 迁移到 MM-WHS。

## 8. 文献边界

- ARCO, NeurIPS 2023：像素级对比学习中的分层采样与方差缩减。PARS 不能宣称首次分层采样。
- PH-Net, CVPR 2024：patch-wise hardness 驱动的半监督分割。PARS 必须避免模型难度驱动。
- TeachAugment, CVPR 2022；iMAS/AugSeg, CVPR 2023：增强策略优化和自适应强增强。appearance distribution design 不能宣称首次 augmentation optimization。
- AdaWAC, ICML 2023；DyCON, CVPR 2025：一致性风险/不确定性动态加权。本文不应把 loss weighting 重新包装成 acquisition contribution。

最终判断：**其他模块仍有机会，但只有“数据进入 acquisition risk 的分布”与“OAAC 参数的联合分布”真正复现了 MPD 的成功范式。下一步应一次只做 PARS，不应同时堆叠两个优化。**
