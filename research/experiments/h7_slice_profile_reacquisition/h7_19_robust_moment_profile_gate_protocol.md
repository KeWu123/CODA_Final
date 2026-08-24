# H7.19 — Robust Moment-Profile Design：零训练授权协议

状态：**原始协议锁定为 gate-first；第10节记录用户随后授权跳过 gate 的探索性 direct full training**
日期：2026-08-18

## 1. 假设

当前连续 Gaussian sampler 覆盖了合理的三抽头 profile，但其 `sigma/phase` 参数耦合了邻层总质量和
方向偏移，也没有保证不同训练病人及轴向区域获得相近的有效 fractional-occupancy 信息。

一个只由 labeled-training exact occupancy 设计、在父 profile 矩和图像扰动预算内的全局稳健分布
\(q\)，可能比均匀 `sigma/phase` 分布更有效，同时避免 ARCP 的逐 stack anatomy-dependent attenuation。

## 2. 冻结项

H7.19 不得改变：

- `SliceEqOcc-OAAC-Strong` 网络、shared pretrain、seed1337；
- SGD/LR、EMA train-mode、teacher hard argmax + 2-D LCC；
- exact L / pseudo U fractional occupancy、soft CE + squared Dice、ramp；
- loader24、student36、三张相邻切片、endpoint clamp 合同；
- OAAC scale1.25、30k、每200步 validation、best-model rule、每1000步周期保存；
- test 与单切片 2-D inference。

H7.19 唯一可能改变的是：通过 gate 后，用一个训练前冻结的全局离散分布 \(q\) 替换父
`sigma ~ U(0.45,0.85), phase ~ U(-0.25,0.25)` sampler。每次得到的三权重仍同时作用于 image 和
one-hot occupancy。

## 3. 数据防火墙

- 只读 `train.list`、`train_slices.list` 与前191个 labeled-training H5 image/label；
- 必须验证前191 slices 恰好属于7个完整病例，不能跨 labeled/unlabeled 边界取邻层；
- 不构建 val/test dataset，不读取任何 U label；
- 不加载 segmentation checkpoint，不读取 logits、confidence、loss 或历史 val/test 结果；
- 本 gate 是 leave-one-patient-out 数据稳定性检验，不是多随机种子实验。

## 4. 父分布网格与 moment 参数

使用锁定的 `21 x 21` midpoint quantile grid：

\[
\sigma_i=0.45+(i+1/2)(0.40/21),\quad
\phi_j=-0.25+(j+1/2)(0.50/21),
\]

其中 \(i,j=0,\ldots,20\)，父离散分布 \(p_0(g)=1/441\)。不得在看到 gate 后改变网格。

对每个 profile：

\[
b_g=1-w_{0,g},\qquad r_g=(w_{+,g}-w_{-,g})/b_g.

\]

## 5. Retained Fractional Information

对 labeled stack \(n\) 与 profile \(g\)：

\[
Q_{ng}=\mathcal A_{w_g}(\operatorname{onehot}(Y_n)),\qquad
C_n=\operatorname{onehot}(Y_{n,0}).

\]

定义 profile-independent axial-transition opportunity

\[
O_n(v)=\mathbf 1[\exists k\in\{-1,+1\}:Y_{n,k}(v)\ne Y_{n,0}(v)].

\]

只在 hard semantic identity 保留时累计 fractional entropy：

\[
U_{ng}=\frac{\sum_v O_n(v)\,\mathbf1[\arg\max Q_{ng}(v)=Y_{n,0}(v)]\,
H(Q_{ng}(v))}{\sum_v O_n(v)+\epsilon}.

\]

空 opportunity slice 只记录，不进入 utility 均值。先按病例和轴向索引三等分聚合，得到
\(u_{s,g}\)；三等分是 index strata，不能在论文中冒称真值定义的 apex/mid/base。

同时记录：hard-target-change rate、fractional-support coverage、foreground mass error，以及
归一化 image residual
`RMS(A_w(X)-X0)/(RMS(X0-mean(X0))+eps)`。RMS 形式可由一阶/二阶轴向差分的
`2x2` Gram matrix 精确计算，不需要为441个 profile 物化441张融合图像；这是运行前锁定的计算实现，
不改变候选定义。

## 6. 全局分布设计

使用两阶段、无可调 `lambda` 的确定性优化：

1. 在以下约束下最大化最差 patient-stratum 期望 utility
   \(t=\min_s\sum_gq_gu_{s,g}\)；
2. 在达到第一阶段最优值 `>=0.99*t_star` 的可行集合中，最小化
   \(D_{KL}(q\|p_0)\)。

锁定约束：

- \(q_g\ge0,\sum_gq_g=1\)；
- phase mirror：\(q(\sigma,\phi)=q(\sigma,-\phi)\)；
- `E[b]`、`E[b^2]`、`E[(b*r)^2]` 相对父分布均在 `±2%`；
- 每个 patient-stratum 的期望 image residual 相对父分布在 `±5%`；
- \(q_g/p_{0,g}\le3\)；
- \(H(q)\ge0.70H(p_0)\)。

若优化器不能证明可行/收敛，结果直接为 no-go，不放宽约束。

## 7. Leave-one-patient-out gate

对7个 labeled 病人逐一留出：用其余6例设计 \(q_{-p}\)，仅在留出病人的三个 index strata
评估。另用7例设计 \(q_{all}\) 供稳定性比较。

全部条件必须满足：

1. 至少 `5/7` 留出病人的**最差 index stratum** RFI 相对 \(p_0\) 提升 `>=10%`，且7例中位提升
   `>=10%`；
2. `JS(q_-p, q_all)` 的7例中位数 `<0.05`、最大值 `<0.10`；
3. 每个 LOPO 设计均满足全部 moment、image residual、density-ratio 与 entropy 约束；
4. 留出病人的 hard-target-change rate 相对父分布增加不超过 `1 percentage point`；
5. 结果不能只由 clamped endpoint stacks 驱动：non-clamped 子集也必须达到同方向且至少 `5%`
   的最差-stratum中位提升；
6. 所有191个 labeled slices、7个病人均被审计；缺失、NaN 或部分运行直接 `no_decision`。

任一条件失败：关闭 H7.19，不调 grid、阈值、utility、moment tolerance、entropy floor 或 density cap。

## 8. Gate 阳性后的唯一 full run

只有 gate 全部通过，才允许从 `train_sliceeq_occ_oaac_strong.py` 创建隔离 successor：

- 冻结 `q_all` 为一个有 hash 的只读 artifact；
- 用独立 profile RNG 从该离散分布采样；
- 其余训练、验证、checkpoint 与 test 代码保持父方法合同；
- 不与 bin integration、五抽头、ARCP、DRO 或其他模块叠加。

父方法 best validation 是 `0.836475`。探索性通过线预先锁为 `>=0.839475`；否则保留
`SliceEqOcc-OAAC-Strong`。只有 validation 通过才测试一次 val-selected checkpoint。PROMISE12 test 已是
development evidence；MM-WHS 必须复用相同网格、约束和设计程序，且只能读取其 labeled-training subset。

## 9. 贡献与表述边界

允许的表述：

> moment-resolved, training-only robust design of a paired through-plane image–occupancy profile
> distribution.

禁止声称首次 DRO、首次 optimal augmentation、首次 MixUp、恢复 scanner PSF，或证明全局最佳融合比例。
H7.19 即使阳性也只是 SliceEqOcc 主方法的 profile-design 组件；核心创新仍是 paired re-acquisition 与
fractional occupancy。

## 10. 2026-08-18 用户授权的执行偏离：跳过 gate，直接完整训练

本协议第7--8节保留为最初的严谨方案和时间线证据。用户随后明确要求“不做测试，直接完整实现训练”。
因此新增一个**探索性 direct-training 执行模式**，不回写或假装已经通过原 LOPO gate：

1. 训练入口启动时只使用全部7个 labeled-training 病人/前191 slices，按第4--6节一次性设计
   `q_all`；
2. 不执行7次 LOPO，不使用原 gate 的10% held-out 增益、JS稳定性或5/7病人授权条件；
3. 数学优化本身仍必须满足 phase mirror、moment、image residual、density ratio 与 entropy
   的全部锁定约束；不可行或未收敛时 fail closed；
4. 将 `q_all`、数据/协议/distribution hash、约束诊断和本次 protocol deviation 原子写入
   `mpd_profile_design.json`；
5. artifact 成功后立即从 OAAC-Strong 隔离入口开始原30k训练，不再等待任何零训练判定；
6. 只替换 `sample_slice_profiles`；父网络、teacher、optimizer/LR、EMA train mode、loss/ramp、
   batch36、OAAC1.25、validation、best-model规则、1000-step周期保存及2-D inference不变；
7. 结果证据等级固定为 `exploratory_user_override_without_lopo_gate`。即使 validation 阳性，也不能
   声称已证明 profile 设计可跨病人泛化；该问题留给 MM-WHS 等外部验证。

实现澄清：若某个 patient/index-third 中所有 exact neighbor labels 均等于 center label，则
opportunity 分母严格为0。该 stratum 对所有 profile 都不提供可定义的 RFI，因而只从 RFI max-min
集合排除；它仍进入 image-residual 约束、完整性统计和 artifact。每个病人至少一个 active RFI
stratum 是硬条件。把结构性空集合当作 `U=0` 会人为令全局 robust optimum 恒为0，因此不是正确
的数学处理。

直接训练入口：`code/train_sliceeq_occ_oaac_strong_mpd.py`。该偏离是用户授权的执行选择，不代表
原 gate 设计有误，也不允许在看到训练结果后调整441点网格、utility或约束。
