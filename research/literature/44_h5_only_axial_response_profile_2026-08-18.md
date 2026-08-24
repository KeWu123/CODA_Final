# 无原始 MRI 时的三切片融合深化：H5-only 方案与文献边界

日期：2026-08-18  
约束：只有按顺序保存的相邻 H5 切片；无 NIfTI/DICOM、spacing、thickness、gap、vendor 或真实 PSF。

## 1. 还能主张什么，不能主张什么

仍可主张：

- 三切片来自同一病例的真实相邻观测；
- 非负归一化权重定义一个 acquisition-inspired 离散融合算子；
- 同一算子作用于图像和 exact/pseudo one-hot occupancy；
- H5 图像能观测某个 profile 在当前 anatomy 上造成的实际图像变化。

不能主张：

- 权重恢复了 scanner slice profile；
- `sigma` 对应毫米层厚；
- H5 轴向频谱仍保留完整 acquisition provenance；
- 由分割验证集选出的比例是物理最优。

## 2. 当前计算的精确含义

父方法采样 \(h=(\sigma,\phi)\)，在 \(-1,0,+1\) 三个位置上计算离散 Gaussian 并 softmax。
\(\sigma\) 控制邻层总质量，\(\phi\) 控制左右不对称。该核对每个样本独立采样，但与该 stack 的真实
轴向变化幅度无关。因此同一个 \(w\) 可能在几乎相同的三张中段切片上接近 identity，却在快速变化的
apex/base stack 上产生很强扰动。

## 3. 低风险改进：bin-integrated profile

把三个中心点上的 Gaussian 值改成三个 slice bin 的积分：

\[
\tilde w_k=\Phi((k+1/2-\phi)/\sigma)-\Phi((k-1/2-\phi)/\sigma),\qquad
w_k=\tilde w_k/\sum_j\tilde w_j.
\]

优点是离散化更合理且不需元数据。缺点是如果沿用同一 sigma 范围，会同时改变 profile severity
distribution，单独的论文新颖性较弱。正式比较必须用中心权重/一阶矩匹配控制，不能把单纯变强误认为
积分形式有效。

## 4. 推荐深化：Axial-Response Calibrated Profile（ARCP）

### 4.1 轴向响应分解

对 stack \(X=(x_-,x_0,x_+)\)，定义

\[
g_1=(x_+-x_-)/2,\qquad g_2=x_--2x_0+x_+.
\]

对任意三 tap 权重 \(w=(w_-,w_0,w_+)\)，令

\[
m_1=w_+-w_-,\qquad m_2=w_-+w_+.
\]

则存在精确恒等式

\[
\mathcal A_w(X)-x_0=m_1g_1+\frac{m_2}{2}g_2.
\]

其中 \(m_1\) 是 phase/方向位移项，\(m_2\) 是邻层混合/模糊项；\(g_1,g_2\) 则完全来自现有 H5。

### 4.2 训练集参考响应

用无标签 training H5 为每个 stack 计算归一化 Gram matrix：

\[
C_i=\frac{1}{s_i^2+\epsilon}
\begin{bmatrix}
\langle g_1,g_1\rangle & \langle g_1,g_2\rangle\\
\langle g_1,g_2\rangle & \langle g_2,g_2\rangle
\end{bmatrix},
\]

其中 \(s_i\) 是中心切片去均值后的 RMS。先在病例内平均，再对训练病例等权平均，得到固定的
\(C_{ref}\)。它不是 PSF，只是项目训练数据的典型轴向响应。

对采样 profile 的向量 \(v=(m_1,m_2/2)^T\)：

\[
e_i^2=v^TC_iv,\qquad e_{ref}^2=v^TC_{ref}v.
\]

### 4.3 只沿 identity--profile 线调整权重

\[
\alpha_i=\sqrt{\frac{e_{ref}^2+\epsilon}{e_i^2+\epsilon}},\qquad
w_i'=e_0+\alpha_i(w-e_0),\quad e_0=(0,1,0).
\]

即

\[
w_-'=\alpha_iw_-,\quad
w_0'=1-\alpha_i(1-w_0),\quad
w_+'=\alpha_iw_+.
\]

\(\alpha\) 只按非负性和父方法已覆盖的中心权重 `[0.485,0.855]` 裁剪，不新增可搜索强度。由于融合
对权重线性，有

\[
\mathcal A_{w_i'}(X_i)-x_{i,0}
=\alpha_i[\mathcal A_w(X_i)-x_{i,0}],
\]

所以它真正校准的是 profile 在当前 stack 上的实际作用量。快速变化的 stack 会自动减弱，变化很小的
stack 会增强，但输出仍是非负凸组合。端点重复邻层的 stack 首轮令 \(\alpha=1\)，不改变父端点策略。

### 4.4 配对监督保持不变

\[
\tilde X_i=\mathcal A_{w_i'}(X_i),\qquad
\tilde Q_i=\mathcal A_{w_i'}(\operatorname{onehot}(Y_i)).
\]

权重只由 image stack 和 training-only `C_ref` 计算，不读取 U label，不读取模型预测置信度，也不通过
validation/test 优化。L exact mask 与 U EMA pseudo mask 仍只作为被融合的 occupancy source。

## 5. 与已有方法的差异

- Inter-Slice Augmentation 使用 frame interpolation 合成相邻切片之间的新图像/标签；ARCP 不训练插值网络，
  不生成几何中间帧，而是校准同一三 tap measurement operator 的实际响应。
  [ECAI 2020](https://journals.sagepub.com/doi/10.3233/FAIA200314)
- AdaMix 根据模型学习状态形成 self-paced mixup 强度；ARCP 不使用 iteration、loss、confidence 或 model
  state，而由同一 stack 的一阶/二阶轴向响应解析计算，并保持 profile-dependent occupancy。
  [Medical Image Analysis 2026](https://www.sciencedirect.com/science/article/pii/S1361841525004037)
- Learned/adversarial augmentation 通过任务损失搜索困难扰动；ARCP 没有额外网络或 inner maximization，
  不会直接把 profile 推到最大 loss 的边界。
- AFTer-UNet/2.5D methods 学习相邻切片 feature attention并改变推理；ARCP 只在训练期改变数据构造。

定向检索未发现“axial first/second response Gram normalization + paired fractional occupancy SSL”的同构方法。
安全的新颖性表述是 operator-specific effect calibration，而不是首次 adaptive mixup。

## 6. 为什么暂不首选 adversarial profile

另一种可行方案是

\[
\min_\theta\mathbb E_X\max_{h\in\mathcal H}
\ell(f_\theta(\mathcal A_hX),\mathcal A_hQ).
\]

它可能更强，但需要额外 student forward、BN/RNG 隔离，并与 VAT、AdvChain、AdaMix 等 adversarial/
adaptive augmentation 高度相邻；U 分支还可能主动放大 pseudo-label 错误。因此只在 ARCP gate 失败且
exact-L profile-loss landscape 显示最坏 profile 不会统一塌缩到同一边界时再考虑，不能与 ARCP 首跑叠加。

## 7. 风险

- ARCP 是数据响应校准，不是真实采集校准；论文必须使用 `acquisition-inspired`。
- 高 axial response 往往位于 apex/base，减弱它可能损失 SliceEqOcc 的有效 fractional signal。
- nearest-neighbor resize 会影响响应统计，但候选与父方法使用同一 H5/resize 合同，仍可作内部比较。
- 如果大量 \(\alpha\) 命中上下界，说明三 tap 安全范围无法实现响应匹配，应停止而非扩大范围。

