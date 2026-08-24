# SliceEq optimization literature boundary

## Primary sources checked

- Kervadec et al., *Boundary loss for highly unbalanced segmentation*, MIDL 2019.
  https://proceedings.mlr.press/v102/kervadec19a.html
  It integrates over contours to address regional class imbalance. A SliceEq extension must
  distinguish acquisition-residual measure from generic contour weighting.
- Liu et al., *Soft Augmentation for Image Classification*, CVPR 2023.
  https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Soft_Augmentation_for_Image_Classification_CVPR_2023_paper.html
  It softens image-level classification targets according to transform severity. SliceEqOcc's
  tissue occupancy is spatially and physically derived rather than global confidence decay.
- Zhao et al., *Augmentation Matters*, CVPR 2023.
  https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
  Confidence-adaptive augmentation is already established; adaptive severity/selection should
  not be the next SliceEq headline.
- Wang et al., *Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels*, CVPR
  2022. https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Semi-Supervised_Semantic_Segmentation_Using_Unreliable_Pseudo-Labels_CVPR_2022_paper.html
  Confidence/entropy partitioning of reliable and unreliable pixels is occupied prior art.
- Sun et al., *Two Losses, One Goal*, ICCV 2025.
  https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.html
  Generic supervised/unsupervised gradient balancing is occupied and should be only a control.
- Kumari and Singh, *Annotation Ambiguity Aware Semi-Supervised Medical Image Segmentation*,
  CVPR 2025.
  https://openaccess.thecvf.com/content/CVPR2025/html/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.html
  It models multiple plausible annotations using pruned decoders and a latent distribution;
  this differs from acquisition-derived fractional tissue occupancy.
- Billot et al., *Partial Volume Segmentation of Brain MRI Scans of Any Resolution and
  Contrast*, MICCAI 2020. https://arxiv.org/abs/2004.10221
  It establishes partial-volume simulation from high-resolution label maps. SliceEq cannot
  claim partial-volume simulation itself as novel.

## Resulting novelty boundary

Do not optimize SliceEq by adding a generic boundary loss, confidence mask, uncertainty head,
adaptive augmentation selector, or gradient-balancing module. The clean remaining extension is
to treat the acquisition-induced occupancy residual as a second normalized integration measure
while retaining the same physical target. This is still adjacent to boundary weighting and
requires the matched binary-boundary control described in H7.3.
