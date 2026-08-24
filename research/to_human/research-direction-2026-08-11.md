# CVPR 研究方向建议（2026-08-11）

## 结论先行

保留用户当前 baseline 的全部训练结构，停止把现有 CoDA 当作论文主线。首选方向是：

> **BMER（Boundary-Manifold Evidence Resynthesis，边界流形成像证据重合成）**：从全部未标注病例估计器官边界法向两侧的真实成像证据分布，再把采样到的证据重合成到标注图像的 GT 边界；解剖几何与硬标签完全不变。

它与 BCP 没有方法结构上的继承关系。BCP 仅作为创新质量标尺：核心操作能够一句话说明、可视化明确、没有推理开销，并能用一组机制实验支撑整篇文章。

目前不能承诺“必发 CVPR”。在没有训练记录和机制数据时，任何保证都不专业。能够保证的是：BMER 是本轮 2020--2026 文献排雷后仍存在合理新颖性空间、且与当前代码兼容的高上限假设；协议同时规定了快速失败条件，避免继续在无效方向上堆模块。

## 一、当前 baseline 到底是什么

严谨名称应为：**BCP-derived EMA hard-pseudo-label self-training**。

- 它不是 BCP。BCP 的核心是 labeled/unlabeled 双向 Copy-Paste 及 GT/伪标签区域混合监督；当前代码没有调用该路径。
- 它也不是标准 Mean Teacher。teacher 与 student 看的是同一份 loader 增强后的未标注张量；teacher softmax 被 argmax 与 2-D 最大连通域变成硬伪标签；student 用 CE+Dice 学该硬标签。
- baseline 的 U-Net、10k pretrain + 30k self-train、EMA、LCC、loss、sampler、ramp、batch 与数据顺序全部锁定。新方法只允许改变输入增强。

必须保留这一 baseline 作为 B0。teacher train-mode、BN、伪标签与 schedule 不在方法组中单独修复，否则无法归因。评估端的 spacing、空预测距离和 checkpoint 误载只可对所有方法统一修正，并同时报告 legacy 指标。

## 二、为什么 CoDA 小提升目前不能说明问题

当前 CoDA 不是“baseline + 一个 coupling”：

1. student 从原/弱视图换成 strong-U；
2. hard LCC one-hot 换成 LCC 内 teacher soft probability；
3. hard CE/Dice 换成 soft CE/Dice；
4. 再把目标按 gamma 推向 uniform。

因此 `gamma=0` 也不等于 baseline，结果不能归因。更关键的是，二分类下

`q' = (1-gamma) q + gamma [0.5, 0.5]`

会给所有硬背景像素注入 `gamma/2` 的伪前景质量；它不是“忽略不可靠像素”。前列腺占比小，背景累计量很容易抵消增强收益。strong-U 又与 labeled 共用含 BatchNorm 的一次 forward，所以它会改变监督分支统计。现有 Sobel/SNR gamma 也从未证明与真实分割错误相关。

本地没有训练 log、event、checkpoint、命令、seed 或曲线。用户报告的无/小提升只能登记为探索性负结果，不能用于论文数值结论。

## 三、搜索空间与方向排序

| 方向 | 新颖性上限 | 机制强度 | 当前可行性 | 主要风险 | 决策 |
|---|---:|---:|---:|---|---|
| BMER：边界流形成像证据重合成 | 高 | 高 | 高 | 被审稿人视为复杂 local contrast jitter | **主线** |
| 随机虚拟薄层重采集 / grid-phase | 中高 | 高 | 中 | inter-slice、SynthSeg、partial-volume 文献密集 | 独立备选 |
| surface-spectral 形态增强 | 中 | 高 | 低 | deformation/shape prior 的新参数化 | 暂停 |
| calibrated maximum-safe augmentation | 中 | 高 | 中 | task-driven/adversarial/selector 已拥挤 | 不作主线 |
| MRI forward/acquisition simulation | 中低 | 高 | 低 | 缺 raw k-space/DICOM，物理真实性不足 | 不作主线 |
| augmentation-orbit 交并集伪标签 | 低 | 中 | 高 | TTA、CPCL、DUEB、conformal sets 已直接覆盖 | 拒绝 |
| 通用 Fourier/style/mixing | 低 | 中 | 高 | β-FFT、MiDSS、FRCNet、ABD 等高度拥挤 | 拒绝 |

虚拟薄层方向很简洁：用相邻切片和真实 slice profile/PSF 模拟有限厚度采集。但已有 inter-slice augmentation、PV-SynthSeg、SynthSeg 与 acquisition-invariant MRI simulation，CVPR 新颖性风险明显高于 BMER；它应作为 BMER 被 kill 后的独立备选，不能叠加来“救结果”。

## 四、BMER 方法定义

### 4.1 未标注证据库

监督预训练结束后，沿用 baseline 模型对全部 U 生成 detached LCC mask。对每个边界建立 object-relative 坐标：

- `rho`：到边界的 signed distance，表示内外法向位置；
- `s`：2-D 轮廓弧长或 3-D 表面位置；
- `z`：体数据中的归一化 apex--base/长轴位置。

提取标量证据场：

`E(s,rho,z) = standardized low-pass intensity on the unwrapped ribbon`。

normal gradient、两侧 contrast 和 transition width 是由该标量场计算的条件/诊断量，不是额外待生成通道。这样渲染定义是闭合的：采样的是法向强度过渡，recipient 的高频 residual texture 原样保留。

由全部未标注病例建立非参数 empirical bank `P_U(E | z, curvature, class)`。第一版不需要 VAE、GAN、diffusion 或 policy network。

### 4.2 只增强 labeled 输入

在标注图像的真实 GT 边界上，从 bank 采样局部双侧 normal profile，并沿轮廓平滑插值。采用 recipient 的 median/MAD 复原强度、保留其 residual texture，在窄带边缘使用归零 taper：

`x'_l(v)=x_l(v)+w(|d_l(v)|)[E_U(s_l,d_l,z_l)-E_l(s_l,d_l,z_l)]`。

必须满足：

- band 外像素不变；
- GT `y_l` 不变；
- recipient anatomy、texture residual 不变；
- 不复制 donor 的 Cartesian patch、shape 或 label；
- 原 U 图像、原 LCC hard pseudo target、原 loss 和 teacher 更新完全不变。

训练代码层面唯一方法接口应接近：

`labeled_images = bmer(labeled_images, labels, frozen_unlabeled_bank)`。

### 4.3 论文题眼

> 现有增强在图像坐标里随机化外观或移动内容；BMER 在固定解剖条件下，直接采样模型决策发生处的边界成像证据分布。

## 五、Motivation 与 contribution

### Motivation

- 小标注集可能覆盖了器官形状，却没有覆盖真实的边界对比度、过渡宽度、partial-volume 与邻近组织纹理。
- PROMISE12 是多中心、多厂商、多协议 T2 MRI；原挑战论文明确指出 scanner/protocol 会影响外观、分辨率与 artifact，并单独评估 apex/base 边界。
- 全局 histogram/Fourier 只改变 `p(x)`；目标决策更直接依赖 `p(evidence | distance to boundary)`。
- CoDA 通过破坏证据再软化 target，监督变弱；BMER 用未标注数据补充 nuisance support，同时仍由准确 GT 提供强监督。

### 结果成立后才能写入论文的 contributions

1. 把医学分割增强系统化地定义在 object-relative boundary manifold，而不是 Cartesian image grid。
2. 提出无额外网络、无新 loss、无推理开销的未标注证据重合成算子，在固定 GT anatomy 上生成真实边界条件变化。
3. 通过 oracle intervention 建立“边界证据覆盖—边界错误”的机制联系，而非只报 Dice。
4. 在至少三个 binary/single-organ benchmark（覆盖 MRI、CT 与真正 3-D 分割）上，用同一坐标/采样规则验证可迁移性。

这些目前是待验证主张，不是已有结论。

## 六、与最近邻工作的明确边界

- [Task-driven DA](https://arxiv.org/abs/2007.05363) 学习全图 additive intensity/deformation field；BMER 是无生成器的边界条件经验分布重合成。
- [ARHNet](https://arxiv.org/abs/2307.01220) 对整个 foreground 做 affine intensity perturbation，再训练 harmonizer；BMER 不扰动整个器官，也不学习 harmonizer。
- [KeepMask/KeepMix](https://www.sciencedirect.com/science/article/pii/S0262885624001604) 保护/组合前景与背景；BMER 主动改变界面两侧的成像证据。
- [BoundaryMix](https://doi.org/10.1016/j.patcog.2021.107924) 删除/替换边界区域并混合标签；BMER 不切、不贴、不混标签。
- [BoCLIS](https://www.isee-ai.cn/~wangruixuan/files/TMI2025Yang.pdf) 在边界 patch 上做对比学习；BMER 是纯输入算子。
- [β-FFT](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_beta-FFT_Nonlinear_Interpolation_and_Differentiated_Training_Strategies_for_Semi-Supervised_Medical_CVPR_2025_paper.pdf) 改变全图低频分布且已测 PROMISE12；BMER 改变条件 interface distribution。

不能泛称“boundary-aware augmentation”。真正 novelty 必须同时包含：object-relative `(s,rho,z)`、unlabeled empirical support、fixed GT geometry、band 外恒等、recipient residual 保留。实现若退化成 `mask*(a*x+b)`、scalar contrast 或边界 patch 替换，应立即放弃。

## 七、第一优先实验：不训练的 kill test

先冻结同一个 baseline，在 held-out labeled volumes 上用 GT 做 oracle intervention：

1. 预先固定 band 宽度和 profile 指标，不按结果挑参数。
2. 把强边界证据 profile 渲染到弱证据 recipient；反向把弱 profile 渲染到强 recipient。
3. 做等作用面积、等强度的 global histogram、boundary scalar contrast、boundary blur 对照。
4. 重新推理冻结模型，测 signed-distance error、NSD、HD95、Dice、band 外变化与 taper seam。
5. 在等形状/等面积但偏离 GT 的 sham contour 上运行相同 renderer，并用简单 edge-only probe/augmentation detector 检查是否出现“增强痕迹直接泄露 GT 边界”。

BMER 必须同时出现：

- profile 强弱与预测变化有序；
- strong→weak 改善、weak→strong 恶化，具有方向性；
- 变化集中在预注册 boundary band；
- full profile 明显强于 scalar contrast/blur。

预注册的最低量化门槛是：full profile 相对最佳简单对照的 patient-bootstrap 95% 区间高于 0；profile 强度与响应的 Spearman `|rho|>=0.30` 且区间不跨 0；band 内按面积归一后的响应富集至少 2 倍。teacher-LCC 与 GT profile 的排序相关至少 `rho>=0.70`，重复 teacher pass 的 ICC 至少 `0.75`。结果出来后不得放宽。

任一点失败，说明核心只是复杂局部抖动，直接 kill，不进入训练、不加模块补救。

尤其要防止 mask-conditioned augmentation 的标签泄漏：若 renderer 在 GT 周围制造现实扫描中不存在的固定光环，模型会因为人工边缘而提升，这不是泛化。BMER 必须采样难/易边界的完整经验分布，而不是一律强化边缘，并证明 edge-to-GT alignment 没有超出真实未标注数据的支持。

第二个 gate：在有 GT 的病例上比较 teacher-LCC 与 GT 提取的 profile。若排序/形状不稳定，未标注 bank 不成立。

## 八、训练实验链

所有实验共用同一预训练 checkpoint、data order、seed、teacher、hard target、loss 与 schedule：

| ID | 唯一输入变化 | 目的 |
|---|---|---|
| B0 | 无 | 当前 baseline |
| B1 | global brightness/contrast/histogram | 全局外观对照 |
| B2 | full-foreground affine jitter | ARHNet-like 对照 |
| B3 | 等面积非边界随机带 | 位置对照 |
| B4 | boundary scalar contrast/blur | 最强简单对照 |
| B5 | labeled-only profile bank | 未标注数据价值 |
| B6 | U profile bank 但不按 `(s,z)` 条件 | 条件化价值 |
| B7 | 完整 BMER | 主方法 |

先 2k--3k steps、一个固定 seed 筛方向；只有 oracle gate 通过且 B7 同时优于 B4 的 region 和 boundary metric，才跑完整实验。正式阶段至少 PROMISE12 + LA + Pancreas-NIH（或有充分理由的等价 binary/single-organ MRI/CT benchmark），3--5 seeds，patient-level 配对统计，Dice/Jaccard、物理 HD95/ASD、NSD/boundary F-score、按 apex/mid/base 与边界对比度分层。第一篇先明确限制为 coherent single-organ boundary；不要为了加入 ACDC 临时发明 multi-class junction 规则。

若只在 PROMISE12 有小提升，应转 MICCAI/MedIA，而不是继续叠 selector、boundary loss 或 teacher module。

## 九、两周执行顺序

- 第 1--2 天：记录 hashes/args/checkpoint，统一评估口径。
- 第 3--4 天：实现 profile bank/renderer 与 band 外恒等、seam、determinism 测试。
- 第 5 天：完成 oracle kill test，决定继续或停止。
- 第 6--8 天：共享 pretrain 的 B0--B7 短跑。
- 第 9--14 天：仅在 gate 通过后跑 PROMISE12 多 seed 和第二个 binary organ transfer。

完整锁定协议见 `research/experiments/h5_boundary_evidence_resynthesis/protocol.md`。
