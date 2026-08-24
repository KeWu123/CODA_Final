# SliceEqOcc-OAAC 流程图视觉风格审计（2026-08-17）

目标：只借鉴顶会方法图的视觉语法，不复制图形资产，不改变最终方法的数据流。

## 参考图

1. [BCP, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html)，Fig. 3。
   - 可借鉴：L/U 颜色编码、真实图像缩略图、Teacher/Student 居中、监督/数据流线型区分、紧凑图例。
   - 不照搬：双向 Copy-Paste、输入混合拓扑和 BCP 配色语义。
2. [UniMatch, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)，Fig. 2。
   - 可借鉴：去除大面积装饰、以最少网络方块表达共享权重、虚线只表示监督关系、弱/强视图关系直接可读。
   - 不照搬：双强流与 feature perturbation。
3. [AugSeg, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)，Fig. 2。
   - 可借鉴：横向紧凑 ribbon、图像缩略图置于流程节点、不同分支使用固定箭头色、Teacher/Student 与 EMA 关系直接展示。
   - 不照搬：其先后增强算子、adaptive labeled augmentation 和损失结构。

## v2 设计决策

- 删除第一版内部大标题和说明书式长文本，正文图由 caption 承担解释。
- 将流程压缩为 `(a) paired acquisition → (b) ordered appearance → (c) shared learner` 三个连续面板。
- 只保留一个 Teacher、一个 Student，防止读者误以为存在多模型或额外推理分支。
- 实线表示图像/前向流；绿色虚线表示 occupancy target supervision；灰色虚线表示 EMA 更新。
- 蓝色表示原生语义锚点，绿色表示成对采集与 fractional occupancy，珊瑚色表示 OAAC 强视图。
- 通过硬掩码与渐变 occupancy 缩略图直观表达：`argmax hard label` 与 `fractional occupancy` 不是同一目标。
- 推理单独放在底部窄条，强调相邻切片、$A_h$、OAAC 均为 training-only。
- 保留第一版作为详细教学图、v2 作为紧凑历史版本；v3 进一步取消跨面板监督长线，作为 CVPR 正文主图。

## v3 线路整理

- 将方法重新组织为三条自上而下排列、完全左到右的训练样本流：Native-L、Re-acquired-L、OAAC-U。
- 每条流在离开构造面板前先封装为明确的“image | target”样本对，目标不再以虚线跨越多个面板。
- 三个样本对通过三条平行、互不交叉的短箭头进入共享 batch，并在 batch 内再次标出各自目标。
- Student、EMA 更新和训练目标全部放在 learner 面板内；EMA 是侧支，Student 到 objective 是独立主路径。
- 无标签 MRI 到 $A_h$ 的图像旁路从 teacher 与 pseudo-mask 流下方经过，避免把 teacher 误读成图像生成器。
