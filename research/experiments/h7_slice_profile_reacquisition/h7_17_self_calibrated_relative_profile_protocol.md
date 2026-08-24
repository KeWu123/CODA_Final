# H7.17 — Self-Calibrated Relative Profile (SCRP) Protocol

状态：**关闭；用户确认无法取得原始 NIfTI/DICOM，物理自校准的数据前提不成立**  
日期：2026-08-18

## 1. 假设

当前 SliceEqOcc 使用覆盖合理 severity 的随机三 tap Gaussian profile，但把每个已采集切片隐式当作理想切片。
若先从训练扫描估计原生 through-plane blur，再只采样可由原生 profile 追加得到的相对退化核，则配对的
image--occupancy augmentation 会更符合真实采集，并可能提升跨 protocol 泛化。

## 2. 唯一允许改变的模块

只替换 `sample_slice_profile()` 的 profile 产生与端点 support 处理：

- 当前：index-space 点采样 Gaussian，\(\sigma\in[0.45,0.85]\)、\(\phi\in[-0.25,0.25]\)；
- 候选：训练扫描的 native-profile proxy → physically legal relative blur → bin-integrated 3-tap weights；
- 端点：从 clamp-and-duplicate 改为 valid-support renormalization。

冻结项：最终 `SliceEqOccOAACStrong` 网络、7-label/11-label预算定义、shared pretrain、seed1337、SGD/lr、
EMA train-mode、pseudo-label/LCC、loss/ramp、batch24/student36、30k、OAAC scale1.25、验证、best保存、
1000-iteration periodic archive、测试和单切片推理。

## 3. 数据隔离

- profile estimator 只允许读取 training patients 的原始 image volume/header；不得读取 val/test image 或任何 label。
- 若使用 labeled-train 进行 fidelity gate，只能使用现有训练集标签，不得反复用其结果调 estimator 超参。
- MM-WHS 使用 PROMISE12 冻结的估计规则和 target-severity rule；不得在 MM-WHS validation 上重新拟合规则。

## 4. Gate A：数据与坐标 provenance

必须全部满足：

1. 能恢复训练病例的原始 NIfTI/DICOM，而非只有逐切片处理后的 H5；
2. slice axis/orientation 可验证，病例排序与 `train_slices.list` 一致；
3. 至少 90% 训练病例具有可信 z-spacing；thickness/gap 若不存在，明确标记为 unknown；
4. 记录归一化、resize 与 resampling 的顺序；profile estimation 在会破坏轴向频谱的操作之前进行。

否则 H7.17 终止，保持当前随机 profile。

### 2026-08-18 决策

用户确认无法取得原始 NIfTI/DICOM。当前 H5 又不提供可信 thickness、gap、orientation 或 scanner
profile，因此 Gate A 明确失败。H7.17 不实现、不训练，也不把 H5 的内容统计称为真实 PSF。后续转入
H7.18：只基于 H5 的 axial response calibration，并将其表述为 augmentation-effect normalization，
而非物理 profile recovery。

## 5. Gate B：估计器恢复与稳定性

在训练图像上构造已知 Gaussian、boxcar 和近似 sinc 的合成退化，不使用 segmentation validation：

- 中心权重 MAE \(\le 0.03\)，或有效宽度相对误差 \(\le 10\%\)；
- bootstrap/split-patch 的病例内 width CV \(\le 15\%\)；
- 不超过 10% 病例命中 estimator 搜索边界；
- profile 与 axial gradient/spectrum 的匹配优于 identity 和固定 `[0.2,0.6,0.2]`。

任一核心条件失败，停止，不通过 segmentation loss 救 estimator。

## 6. Gate C：三 tap support 与非退化

- relative kernel 的三层区间积分质量在截断前应覆盖 \(\ge95\%\)；否则三层方法不足，H7.17 不运行；
- 至少 90% eligible cases 的 relative kernel 非 identity；
- 中心权重保持在父方法已探索的安全范围约 `[0.485,0.855]`，首轮不得扩大 severity；
- profile-shuffle 前后 marginal weight distribution 完全匹配，保证后续可归因于病例校准而非强度变化。

## 7. Gate D：训练标签内 paired fidelity

仅使用 7 个 labeled training patients，以现有 exact mask 构造 occupancy：

- image 与 occupancy 必须逐样本使用完全相同的权重；
- occupancy simplex error \(<10^{-6}\)；
- 与父方法相比，fractional-support coverage 不下降超过 10%；
- valid-support endpoint 版本不能降低 first/last-third foreground recall 超过 1 percentage point；
- estimator 在至少 5/7 patients 上改善预注册的 through-plane patch-match objective。

该 gate 仅授权一次 full run，不保证 Dice 上升。

## 8. 首次 full-run 对照

主运行：`SliceEqOccOAACStrong + SCRP`，只改 profile 模块。

最小归因集合：

1. 当前随机 Gaussian profile（父方法）；
2. 固定 `[0.2,0.6,0.2]`；
3. SCRP；
4. case-independent marginal-matched profile；
5. case-shuffled SCRP；
6. bin-integrated、但不做 native calibration 的 profile。

首个探索 run 只运行 1 与 3；只有 3 在未改验证规则下阳性，才补 2/4/5/6。

## 9. 决策规则

父方法当前 best validation Dice 为 `0.836475`。SCRP 的 exploratory pass 要求：

- best validation Dice `>=0.839475`（绝对提升至少 0.003）；
- 提升不能由单个 validation patient 主导；
- 选中的 checkpoint 必须仍由原 validation best 规则产生；
- 只测试一次 val-selected checkpoint；现有 PROMISE12 test 已参与开发，只能作为 development evidence；
- MM-WHS 的冻结迁移结果才是关键外部证据。

若 best validation 提升小于 0.003，则不搜索 estimator distance、kernel family、target ratio 或 profile range；关闭
自校准方向并保留最终 `SliceEqOccOAACStrong`。

## 10. 原创性边界

SCRP 不声称首次估计 MRI slice profile，也不声称首次 spacing-conditioned segmentation。可检验的新组合是：

> native-profile self-calibration + physically legal relative re-acquisition + paired exact/pseudo fractional occupancy
> for semi-supervised segmentation, with unchanged 2-D inference.

若 profile 最终由 segmentation validation/test 选择，或使用 unconstrained attention 直接输出三权重，则不再满足该贡献定义。
