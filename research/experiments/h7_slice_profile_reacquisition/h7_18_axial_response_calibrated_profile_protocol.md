# H7.18 — Axial-Response Calibrated Profile（ARCP）

状态：**实现完成；等待服务器张量测试与 H5-only gate，gate 通过前不授权 30k full run**  
日期：2026-08-18

## 1. 假设

当前随机 Gaussian profile 的参数强度相同，但其实际图像扰动取决于 stack 的一阶/二阶轴向响应，造成
大量过弱与少量过强的 re-acquisition。用训练 H5 的 patient-balanced reference Gram response 对每个
sample 的 profile 沿 identity ray 校准，可以在不学习网络、不使用元数据和不破坏 paired occupancy 的
条件下稳定 acquisition augmentation risk，并改善 SliceEqOcc-OAAC-Strong。

## 2. 唯一方法变量

1. 完全保留父 sampler 产生的 \(\sigma,\phi,w\)；
2. 从 image stack 计算 \(g_1,g_2,C_i\)；
3. 用训练图像预计算并冻结 \(C_{ref}\)；
4. 计算解析 \(\alpha_i\)，得到 \(w_i'=e_0+\alpha_i(w-e_0)\)；
5. 同一 \(w_i'\) 进入现有 `paired_slice_reacquisition`。

冻结：`SliceEqOccOAACStrong` 的网络、pretrain、seed1337、optimizer/lr、EMA train-mode、teacher/LCC、
loss/ramp、batch24/student36、OAAC scale1.25、30k、验证/best规则、1000-step周期保存和2-D推理。

## 3. 数据防火墙

- `C_ref` 只读取 `train_slices.list` 对应的 image H5；不读取 label、val、test。
- patient-balanced：先聚合每个 training case，再等权平均 case matrix，避免长 volume 主导。
- U 权重计算函数不得接收 label、pseudo mask、teacher/student logits、iteration 或 loss。
- 保存 `C_ref`、list hash、代码 hash 与数值 precision；PROMISE12 冻结后原样用于 MM-WHS，不重新拟合。

## 4. 数值合同

- \(C_i,C_{ref}\) 必须对称半正定（允许数值容差 `1e-7`）；
- \(w_i'\ge0\)、sum error `<1e-6`；
- \(w_0'\in[0.4850,0.8553]\)，覆盖父 sampler 的完整数值范围；
- 当 \(C_i=C_{ref}\) 时 \(\alpha=1\)，输出严格复现父权重；
- 当 stack 三张相同或数值退化时回退父权重；
- `neighbor_clamped=True` 首轮回退父权重；
- 权重与 target detach，不向 image、teacher 或模型传播 profile-calibration 梯度。

## 5. 零训练 gate

使用所有 training image stacks；exact occupancy 检查只使用现有 labeled-train masks。固定使用
`sigma={0.45,0.65,0.85} × phase={-0.25,0,+0.25}` 的九点诊断网格；该网格只评估机制，正式训练
仍使用父方法连续随机 sampler。

### 5.1 作用量与非退化

- 至少 50% non-clamped stacks 有 `|alpha-1|>=0.05`；
- 命中中心权重上下界的 stacks 均不超过 25%；
- profile-moment 分层后，归一化 image residual 的 patient-balanced CV 相对父方法下降至少 20%；
- 实际 residual 与未校准 axial response 的相关性绝对值下降至少 30%。

### 5.2 核心信号保护

- labeled exact fractional-support coverage 保留至少 90%；
- first/middle/last index thirds 分别报告，first/last 的 occupancy residual 不得下降超过 10%；
- parent 与 ARCP 的 mean center weight 差 `<0.03`，防止结果只是全局 profile severity 调参；
- image residual 与 exact occupancy residual 的 patient-balanced Spearman 不得下降。

任一核心条件失败，关闭 ARCP；不调 epsilon、reference quantile、alpha margin 或 center范围。

## 6. 首次 full run

从最终 `SliceEqOccOAACStrong` 独立 fork，只加入 ARCP profile utility。原始代码不改。

日志增加：alpha mean/std/quantiles、bound-hit、parent/calibrated center weight、image residual before/after、
L/U fractional-support。它们只用于机制诊断，不进入 checkpoint 选择。

父方法 best validation `0.836475`；通过要求 `>=0.839475`，仍使用原 validation-best selector。
只有 val-selected checkpoint 测试一次；PROMISE12 结果属于 development，MM-WHS 使用冻结 `C_ref`/规则验证。

## 7. 阳性后的必要对照

1. Parent random profile；
2. ARCP；
3. `alpha` 在样本间随机 shuffle，但保持完全相同 marginal；
4. 全部 stack 使用训练集 mean alpha；
5. bin-integrated Gaussian with marginal-matched center/phase moments。

若 ARCP 不优于 shuffled-alpha，则收益只能归因于改变 profile distribution，不能归因于 axial response calibration。

## 8. 论文边界

可表述：H5-observed axial-response normalization of a paired acquisition-inspired operator。

不可表述：scanner PSF recovery、thickness calibration、physical millimeter profile、first adaptive mixup、first
cross-slice fusion。ARCP 是 SliceEqOcc 的 profile-risk calibration 组件，主贡献仍是 paired image--fractional
occupancy re-acquisition 与 OAAC 的有序组合。

## 9. 实现

- `code/utils/sliceeq_arcp.py`：image-only patient-balanced reference、Gram response 与解析校准；
- `code/analyze_sliceeq_arcp_gate.py`：固定九 profile 的零训练 gate；
- `code/train_sliceeq_occ_oaac_strong_arcp.py`：从最终 Strong 包装的独立训练入口；
- `code/test_sliceeq_occ_oaac_strong_arcp.py`：严格2-D单checkpoint测试；
- `tests/test_sliceeq_arcp.py` 与 `tests/test_sliceeq_arcp_contract.py`；
- `docs/SLICEEQ_OCC_OAAC_STRONG_ARCP_README.md`。

原 `train_sliceeq_occ.py` 与 `train_sliceeq_occ_oaac_strong.py` 均未修改。ARCP 包装器只截获
`paired_slice_reacquisition` 的三权重，并把实际校准权重传给原 diagnostics；训练循环、validation 和
checkpoint 保存继续由 Strong parent 执行。
