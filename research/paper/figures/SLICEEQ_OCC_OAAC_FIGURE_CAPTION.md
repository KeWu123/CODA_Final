# SliceEqOcc-OAAC 方法总览图说明

## 中文图注

**图1：SliceEqOcc-OAAC-Strong 的训练与推理流程。** 在有标注分支中，同一随机三抽头切片剖面算子 $A_h$ 同时作用于真实相邻 MRI 切片和 one-hot 真值掩码，生成重新采集图像及精确分数占据目标；原始中心切片保留为硬语义锚点。在无标注分支中，EMA 教师首先对三张邻片产生经逐切片二维最大连通域处理的硬伪掩码，随后同一 $A_h$ 生成配对的重新采集图像与伪分数占据目标。OAAC-Strong 只在采集配对完成后，对无标注学生图像依次施加 gamma、contrast 和 brightness 扰动，不修改占据目标。学生网络以一次36-view前向联合学习原始有标注中心切片、重新采集有标注切片和 OAAC 无标注切片。相邻切片、$A_h$ 与 OAAC 均仅用于训练；推理仍使用原始单切片2D U-Net，因而不增加模型参数或推理开销。

## English caption

**Figure 1: Training and inference pipeline of SliceEqOcc-OAAC-Strong.** In the labeled branch, the same stochastic three-tap slice-profile operator $A_h$ is applied to real neighboring MR slices and one-hot ground-truth masks, producing a re-acquired image and its exact fractional-occupancy target, while the native center slice remains a hard semantic anchor. In the unlabeled branch, the EMA teacher first produces hard pseudo masks for the three neighboring slices with slice-wise 2-D largest-component filtering. The same $A_h$ then forms a paired re-acquired image and pseudo-occupancy target. OAAC-Strong is applied only after this pairing and perturbs only the unlabeled student image through ordered gamma, contrast, and brightness transforms; the occupancy target is unchanged. A shared student network jointly learns from 12 native labeled, 12 re-acquired labeled, and 12 OAAC unlabeled views. Neighboring slices, $A_h$, and OAAC are training-only; inference uses the unchanged single-slice 2-D U-Net with no additional parameters or latency.

## 文件

- `fig_sliceeq_occ_oaac_pipeline_v3.svg`：重新排版的可编辑矢量源文件，三条训练流互不交叉，**推荐作为 CVPR 正文主图**。
- `fig_sliceeq_occ_oaac_pipeline_v3.pdf`：重新排版版论文排版文件。
- `fig_sliceeq_occ_oaac_pipeline_v3.png`：重新排版版高清预览。
- `fig_sliceeq_occ_oaac_pipeline_v2.svg/pdf/png`：保留的紧凑美化历史版本。
- `fig_sliceeq_occ_oaac_pipeline.svg/pdf/png`：保留的详细教学版，适合补充材料或答辩。
- `render_sliceeq_occ_oaac_pipeline.html`：固定画布导出包装页。
- `render_sliceeq_occ_oaac_pipeline.ps1`：使用本机 Edge 重复导出 PNG/PDF。
- `render_sliceeq_occ_oaac_pipeline_v2.ps1`：重复导出紧凑美化历史版本。
- `render_sliceeq_occ_oaac_pipeline_v3.ps1`：重复导出重新排版版。

重新导出：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File research\paper\figures\render_sliceeq_occ_oaac_pipeline.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File research\paper\figures\render_sliceeq_occ_oaac_pipeline_v2.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File research\paper\figures\render_sliceeq_occ_oaac_pipeline_v3.ps1
```

## LaTeX 示例

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_sliceeq_occ_oaac_pipeline_v3.pdf}
  \caption{Training and inference pipeline of SliceEqOcc-OAAC-Strong.}
  \label{fig:method_overview}
\end{figure*}
```
