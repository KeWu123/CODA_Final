# SliceEqOcc-OAAC-MPD 机制原理与最终方法判断

## 1. 最终判断

结合方法机制和当前最高性能，`SliceEqOcc-OAAC-Strong-MPD` 比
`SliceEqOcc-OAAC-Strong` 更适合作为最终论文方法。它不是增加网络模块，
而是把原本启发式的切片剖面均匀采样，改写为由 labeled-training exact
occupancy 驱动、受物理形态约束的全局稳健分布设计。网络、损失、EMA、
OAAC、batch 和推理均不变，因此性能变化能够归因于 profile risk 的重新分配。

当前开发结果为：

- OAAC-Strong：Dice `0.851960`；
- OAAC-Strong-MPD iter29000：Dice `0.854573`；
- 增益：`+0.002613` Dice、`+0.003983` Jaccard。

## 2. SliceEqOcc 的基本测量算子

对中心切片及其相邻层，令

\[
X^- = X_{z-1},\quad X^0=X_z,\quad X^+=X_{z+1},
\]

三抽头 profile 为

\[
w=(w_-,w_0,w_+),\qquad w_k\ge 0,\quad \sum_k w_k=1.
\]

重新采集图像为

\[
\widetilde X_w=w_-X^-+w_0X^0+w_+X^+.
\]

同一权重同时作用于 exact GT 或 teacher pseudo mask 的 one-hot 表示：

\[
\widetilde Q_w=w_-Q^-+w_0Q^0+w_+Q^+.
\]

`Q_w` 不是类别不确定性，而是该模拟采集算子诱导的组织分数占据。这样，
图像中的层间混合和监督目标中的组织混合严格成对，避免“图像已经改变但
仍使用中心硬标签”的 target mismatch。

## 3. 用两个轴向矩理解三切片融合

定义邻层总质量和方向比：

\[
b=w_-+w_+=1-w_0,\qquad
r=\frac{w_+-w_-}{b}.
\]

再令

\[
g_1=\frac{X^+-X^-}{2},\qquad
g_2=X^- - 2X^0 + X^+.
\]

则融合残差可精确写成

\[
\widetilde X_w-X^0=(br)g_1+\frac{b}{2}g_2.
\]

因此：

- `b` 控制总体邻层混合/轴向平滑强度；
- `br=w_+-w_-` 控制向前或向后的方向偏移；
- `b^2` 控制增强强度的二阶能量；
- `(br)^2` 控制方向位移的二阶能量。

原始 SliceEqOcc 在
`sigma~U(0.45,0.85), phase~U(-0.25,0.25)` 上采样，再转换为三点
Gaussian 权重。它不是固定 `[0.2,0.6,0.2]`。该固定比例只是一个接近
`phase=0, sigma=0.6746` 的代表性剖面。

## 4. MPD 优化的对象不是一组权重，而是一个分布

MPD 在父采样支持内构造 `21x21=441` 个候选 profile。父分布 `p0` 对
`(sigma,phase)` 网格均匀采样；MPD 求出一个全局离散分布 `q`，训练时仍然
随机抽取不同 profile。它没有把所有切片固定成某个单一比例。

对训练标注栈，先计算

\[
Q_{n,g}=A_{w_g}(\operatorname{onehot}(Y_n)).
\]

若某像素的相邻层语义与中心层不同，但融合后的 hard class 仍等于中心层，
则该像素的 occupancy entropy 被定义为 Retained Fractional Information：

\[
U_{n,g}=\frac{\sum_v O_n(v)
\mathbf 1[\arg\max Q_{n,g}(v)=Y_n^0(v)]H(Q_{n,g}(v))}
{\sum_vO_n(v)+\epsilon}.
\]

RFI 奖励的是：

1. profile 确实产生部分容积/分数占据；
2. 同时不把中心层的主要语义翻转；
3. 信号必须出现在真实存在轴向标签变化的位置。

## 5. 两阶段稳健优化

首先按7个训练病人和每例三个轴向 index-third 聚合 RFI。第一阶段求：

\[
\max_q\min_s\sum_g q_g u_{s,g},
\]

即最大化最弱病人—轴向区域的期望RFI，避免 profile risk 主要服务于某个
病人或某个切片密集区域。

第二阶段在保留第一阶段99%最优值的可行域内最小化：

\[
D_{KL}(q\|p_0).
\]

这一步使最终 `q` 尽量靠近原始均匀父分布，只进行解决最弱区域所必需的
最小重分配。

关键约束包括：

- `q(sigma,phase)=q(sigma,-phase)`，不产生方向偏置；
- `E[b]`、`E[b^2]`、`E[(br)^2]` 相对父分布限制在±2%；
- 每个病人—轴向区域的图像RMS扰动限制在父分布±5%；
- 单个候选密度不超过父分布3倍；
- 分布熵不少于父分布70%，避免坍缩成固定比例。

## 6. 实际求得的分布说明了什么

父分布和MPD设计分布的关键矩为：

| 统计量 | 父分布 | MPD | 相对变化 |
|---|---:|---:|---:|
| `E[b]` | 0.375076 | 0.382577 | **+2.00%** |
| `E[b^2]` | 0.149946 | 0.152546 | **+1.73%** |
| `E[(br)^2]` | 0.014946 | 0.014647 | **-2.00%** |

由于分布严格 phase 对称，MPD 的平均权重约为：

\[
E_q[w]\approx[0.1913,\;0.6174,\;0.1913],
\]

父分布平均约为：

\[
E_{p_0}[w]\approx[0.1875,\;0.6249,\;0.1875].
\]

这表明求解器选择了一个清晰但克制的策略：

- 略微增加相邻层的总贡献；
- 略微增强二阶轴向混合；
- 同时减少方向偏移能量；
- 保留很高的 profile 多样性，实际熵为父分布的98.83%；
- 最大密度比只有1.608，远低于约束上限3。

所以MPD不是“更强地模糊”，而是把增强风险向**稍强、更加对称、在最弱
病人/轴向区域仍能产生有效分数占据**的profile重新分配。

## 7. 为什么这可能提高分割结果

### 7.1 提高有效边界监督，而不是增加普通样本数量

SliceEqOcc 的新增信息集中在轴向组织变化处。MPD让更多采样概率落在能够
产生fractional occupancy、但不翻转中心语义的profile上。因此网络看到的
不是更多重复硬标签，而是更稳定的亚像素/部分容积边界监督。

### 7.2 降低无意义的方向噪声

`E[(br)^2]`下降说明MPD减少了“向上一层或下一层偏移”的随机位移能量。
这种方向偏移容易把具体相邻层的伪标签错误带入中心层；对称混合则更接近
局部组织积分。MPD因此可能保留部分容积信息，同时减少伪目标的方向噪声。

### 7.3 防止长体积或中间区域支配采样风险

普通均匀 profile 采样对所有slice使用同一先验，但数据中的不同病人和轴向
区域具有不同的标签变化机会。max-min目标让最弱patient/index-third也获得
足够RFI，使训练风险不只由容易产生大边界的区域决定。

### 7.4 与OAAC形成互补而不是重复

MPD改变的是**采集域和监督占据分布**；OAAC改变的是重新采集后U图像的
gamma、contrast和brightness，不改变空间坐标或occupancy target：

\[
X\xrightarrow{A_w}(\widetilde X_w,\widetilde Q_w)
\xrightarrow{\text{OAAC}}(G(\widetilde X_w),\widetilde Q_w).
\]

MPD解决“应该以什么频率模拟哪些层间测量”，OAAC解决“同一个测量在不同
外观下是否保持预测一致”。两者分别覆盖采集变化和外观变化，因而能叠加。

## 8. 为什么增益不会特别大

MPD被刻意限制为接近父分布：矩变化不超过2%，图像扰动不超过5%，熵实际
保留98.83%。因此它是在已经达到0.851960的强方法上重新分配训练风险，而
不是创造新的网络容量。最终`+0.002613` Dice、`+0.003983` Jaccard与这种
小而结构化的分布调整量相匹配。

## 9. Strong 与 MPD 哪个机制更优

| 维度 | OAAC-Strong | OAAC-Strong-MPD |
|---|---|---|
| profile来源 | 人工均匀采样sigma/phase | exact-training occupancy稳健设计 |
| 是否固定比例 | 否 | 否 |
| 病人/轴向均衡 | 无显式约束 | max-min RFI |
| 方向对称 | 分布上由均匀phase隐式满足 | 优化硬约束显式满足 |
| 强度控制 | 参数区间间接控制 | moment + image residual双重控制 |
| 防坍缩 | 连续均匀先验 | KL、熵下界、密度上限 |
| 网络/推理变化 | 无 | 无 |
| 当前Dice | 0.851960 | **0.854573** |

因此MPD在机制和结果两方面都更优。它补上了原方法最容易被审稿人追问的
问题：profile比例并非任意指定，而是在保持父采集预算的前提下，由训练
exact occupancy的跨病人/轴向稳健信息准则求得。

## 10. 论文贡献边界

安全且准确的贡献表述是：

> We replace heuristic uniform profile sampling with a training-only,
> moment-constrained robust design over paired through-plane image–occupancy
> operators, maximizing retained fractional information across subjects and
> axial strata while preserving acquisition severity and profile diversity.

不能声称：恢复真实scanner PSF、获得物理最优层厚、首次MixUp/DRO、或证明
全局最佳融合比例。MPD使用的是processed H5上的acquisition-inspired profile，
而不是scanner metadata标定的真实采集核。

## 11. 最终方法链

最终方法可以概括为：

1. 用exact labeled-training occupancy离线设计并冻结全局MPD分布；
2. 从MPD抽取三切片profile；
3. 对L图像/GT和U图像/teacher pseudo mask使用同一个profile；
4. 保留连续fractional occupancy进行soft CE + squared Dice监督；
5. 仅对重新采集后的U学生图像施加OAAC-Strong；
6. 使用原EMA、U-Net和一致性ramp完成训练；
7. 推理时丢弃所有三切片、MPD和OAAC路径，仍使用原2-D单切片U-Net。

这使最终贡献形成一条完整逻辑：**设计采集风险 → 成对模拟采集 → 保留组织
占据 → 扩展外观覆盖 → 保持零推理开销。**
