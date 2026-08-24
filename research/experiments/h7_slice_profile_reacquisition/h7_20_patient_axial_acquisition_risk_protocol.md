# H7.20 — Patient–Axial Acquisition-Risk Sampling（PARS）协议

状态：**实现前锁定；直接完整训练；单一 sampler 干预**  
日期：2026-08-19

## 1. 动机与假设

最终父方法 `SliceEqOcc-OAAC-Strong-MPD` 已经把三切片 acquisition profile 从均匀
sigma/phase 分布改造成 exact-L 约束的稳健全局分布，但训练 loader 仍然在 L/U slice indices 上
抽样。病例拥有的切片越多，其进入 SGD 的概率越高；同一病例中切片更密集的 index 区域也得到更多
更新。这使数据支持分布与 MPD 的 patient×index-third 稳健目标不完全一致。

假设：在不改变 MPD、OAAC、模型或训练目标的前提下，将 slice-entry risk 改成 patient-balanced、
exact-L acquisition-opportunity-designed 的全局轴向分布，能够让 paired image–fractional-occupancy
监督更均衡地覆盖不同病例和轴向轨迹。

## 2. 与已有工作的边界

本实现不是通用 hard mining、class balancing 或 uncertainty sampling：

- 不读取 loss、gradient、prediction、confidence、entropy 或训练 iteration；
- 不给某个测试病例、U 病例或在线“困难样本”分配特定权重；
- 只设计一个跨所有病例共享的 three-index-third 概率向量；
- 设计量是 frozen MPD operator 下 exact-L fractional-occupancy opportunity；
- U 运行时只使用 case id 与相对 index third，不读取 U label。

定向检索未发现“patient-uniform + exact fractional-occupancy acquisition opportunity + paired
through-plane image/target operator”的同构实现，但这不是穷尽式唯一性证明。以下先例限定了可声明范围：

- ARCO, NeurIPS 2023：像素级对比学习的 stratified sampling / variance reduction；
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/1f7e6d5c84b0ed286d0e69b7d2c79b47-Abstract-Conference.html
- PH-Net, CVPR 2024：patch-wise model hardness；
  https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_PH-Net_Semi-Supervised_Breast_Lesion_Segmentation_via_Patch-wise_Hardness_CVPR_2024_paper.pdf
- Versatile Medical Image Segmentation, CVPR 2024：multi-source hierarchical dataset/image sampling；
  https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Versatile_Medical_Image_Segmentation_Learned_from_Multi-Source_Datasets_via_Model_CVPR_2024_paper.pdf

因此禁止声称首次 stratified sampling、balanced sampling、hard sampling 或 patient sampling。允许的窄表述是：

> a frozen patient-balanced axial support distribution designed from exact acquisition-derived
> fractional-occupancy opportunity for a paired through-plane image–target operator.

## 3. 冻结父方法

H7.20 必须继承并冻结：

- `SliceEqOcc-OAAC-Strong-MPD` 网络与 shared Pre10000；
- seed1337，SGD、LR0.01；
- EMA train-mode、hard argmax、逐层 2-D LCC；
- MPD 21×21 profile grid、约束与冻结 q；
- exact-L / pseudo-U paired fractional occupancy；
- soft CE + squared soft Dice、consistency ramp；
- loader `12L+12U`，warmup 后 student36；
- OAAC Strong scale1.25；
- 30k、每200 iter validation、原 best rule、每1000 iter 周期权重；
- 原 strict 2-D single-slice test/inference。

唯一变化：iter1000 后父 `TwoStreamBatchSampler` 的抽样分布被 H7.20 sampler 替换。iter0--999
consistency关闭的 warmup 完整保留父 sampler，避免把 pre-acquisition supervised/teacher warmup 改变
混入归因。

## 4. Exact-L acquisition opportunity

启动时重新生成父 MPD artifact。对 labeled patient `p`、index-third `t` 中的每个 slice `i`，在
冻结 MPD 分布下计算 retained fractional information：

\[
u_{pt}=\mathbb E_{i\in(p,t)}\mathbb E_{w\sim q_{MPD}}
\left[H(A_wY_i)\;\mathbf1\{\arg\max A_wY_i=Y_{i,0}\}\right],
\]

其中没有相邻 label 变化的 slice 对 per-sampled-slice opportunity 贡献0。每个病人按其父轴向分布下
的期望 opportunity 归一化，避免大器官/容易病例凭绝对像素量主导设计。

## 5. 全局轴向概率设计

父概率 `p_z` 是当前前191个 labeled slices 在 first/middle/last index-third 中的实际比例。每个病例
的 thirds 仅由 slice index rank 等分，不使用器官真值位置，因此不能称为 anatomical apex/mid/base。

使用两阶段确定性 SLSQP：

1. 最大化所有 active patient-third 的最小 normalized exposure `q_t * u_norm[p,t]`；
2. 在保留第一阶段 `99%` 最优 worst exposure 的集合内，最小化 `KL(q||p_z)`。

锁定约束：

- `q>=0, sum(q)=1`；
- `q_t/p_z,t <= 1.50`，因此任一 third 不会占据过半期望样本；
- `H(q) >= 0.90 H(p_z)`，保留宽覆盖；
- 三个 axial thirds 均必须包含 exact opportunity；
- 优化不可行或未收敛时 fail closed，不放宽约束。

该设计只输出一个三元素全局 q，不输出病人特定权重。

## 6. 运行时 sampler

L 与 U stream 分别执行：

1. 对 stream 内 patient 做循环随机排列，使每个 epoch 内 patient count 最大差不超过1；
2. 按冻结 `q` 选择 index third；
3. 在所选 patient-third 内均匀选择 slice；
4. 拼接为与父方法相同的 `12L+12U` 顺序。

sampler 使用独立 NumPy Generator seed1341，不消耗 profile、OAAC、augmentation、dropout 或全局 NumPy/
Torch RNG。epoch 长度仍为 `191//12=15`，所以 iteration、validation 和 checkpoint cadence 不变。
前1000 iterations 仍调用冻结父 `TwoStreamBatchSampler`，只从第一个 acquisition-active iteration 开始
使用上述分布。

## 7. 数据防火墙

- 设计只读 `train_slices.list` 和前191个 labeled-training H5 exact image/label（经父 MPD统计）；
- U manifest 只读 slice name，不打开其 H5 label；
- 不构建 val/test dataset；
- 不加载模型预测、历史 checkpoint 指标、loss 或 uncertainty；
- 保存 `mpd_profile_design.json` 与 `pars_sampling_design.json` 及数据/协议/distribution hash。

## 8. 执行与判断

用户授权直接实现并完整训练，不设置额外 gate。该 run 固定为 exploratory single-seed1337；用户明确
不要求多随机种子验证。

父开发上界为 MPD iter29000 Dice `0.854573`。按项目现有最高已测试 checkpoint 选择策略，只有超过该值
才作为数值替代；若不超过，PARS 关闭，不调 thirds 数量、density cap、entropy floor、sampler seed 或
opportunity 公式，也不根据具体测试病例改概率。

首跑禁止同时加入 OAAC joint-distribution design、endpoint projection 或任何其他组件。

## 9. 实现入口

- train：`code/train_sliceeq_occ_oaac_strong_mpd_pars.py`
- test：`code/test_sliceeq_occ_oaac_strong_mpd_pars.py`
- utility：`code/utils/sliceeq_pars.py`
- tests：`tests/test_sliceeq_pars.py`、`tests/test_sliceeq_pars_contract.py`
- runtime artifact：`pars_sampling_design.json`
