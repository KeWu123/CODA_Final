# H7.20 PARS 定向原创性边界

日期：2026-08-19

## 检索问题

是否已有工作将以下四项同时用于半监督医学图像分割：

1. patient-uniform 的训练 slice-entry distribution；
2. 相对 axial index strata；
3. 用 exact labeled fractional occupancy under a through-plane acquisition operator 设计全局概率；
4. 同一冻结分布迁移到 U，且不依赖模型难度/不确定性。

本轮在 CVF、NeurIPS、MICCAI 与 PubMed 的定向组合检索中未找到完全同构方法。这只是检索结论，不是
数学上的“无人做过”证明。

## 最近邻与不能声明的内容

### ARCO — NeurIPS 2023

链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/1f7e6d5c84b0ed286d0e69b7d2c79b47-Abstract-Conference.html

ARCO 使用分层采样与方差缩减改善半监督医学分割中的像素级对比学习。因此 PARS 不能声称首次 stratified
sampling、variance-reduced sampling 或 medical segmentation balancing。区别是 PARS 不抽像素/类别、
不依赖对比特征，而是为 paired acquisition-occupancy operator 的病人/轴向支持设计固定 sample law。

### PH-Net — CVPR 2024

链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_PH-Net_Semi-Supervised_Breast_Lesion_Segmentation_via_Patch-wise_Hardness_CVPR_2024_paper.pdf

PH-Net 根据 patch-wise hardness 进行专门学习/增强。PARS 必须保持 prediction/loss-free，不能把 exact
acquisition opportunity 写成一般“难度”。

### Versatile Medical Image Segmentation — CVPR 2024

链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Versatile_Medical_Image_Segmentation_Learned_from_Multi-Source_Datasets_via_Model_CVPR_2024_paper.pdf

该工作包含多源数据的分层 dataset/image sampling，并分析稀疏 axial/sagittal/coronal slices。因此 PARS
不能声称首次 hierarchical sampling 或 axial slice selection。区别是单一MRI数据集内、训练风险层面的
patient-balanced acquisition opportunity，而不是多源标签消歧或标注子集构造。

### 数据集分层工作

MIDRC 等工作使用 patient-level stratification 构造平衡数据集/holdout；NeurIPS 2025 的 Stratify or Die
研究 segmentation split stratification。它们限制了“patient stratification”表述，但都不是本方法的
online training batch law或 paired fractional occupancy operator。

## 允许的贡献表述

> We align the support distribution of stochastic training with the paired acquisition risk: patients are
> sampled uniformly, while a single global axial-support law is robustly designed from exact labeled
> fractional-occupancy opportunity and frozen for both labeled and unlabeled streams.

## 禁止表述

- first patient-balanced sampling；
- first stratified/hierarchical sampling；
- first axial slice sampling；
- hard-example mining；
- learned anatomy-aware sampling；
- recovered scanner acquisition distribution；
- patient-specific/adaptive sampling。

## 论文角色

若阳性，PARS 只能作为 SliceEqOcc-OAAC-MPD 的 support-distribution refinement：MPD 设计 operator law，
PARS 设计该 operator risk 看到的数据支持。核心贡献仍是 paired re-acquisition 与 fractional occupancy。
