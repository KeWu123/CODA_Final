# Adaptive Bidirectional Displacement (ABD)

- Venue/year: CVPR 2024.
- Core idea: use confidence ranks to replace low-confidence patches across weak/strong views for unlabeled data; inversely inject unreliable content into labeled views to force learning.
- Relevance: strongest collision for confidence-guided augmentation and handling multiple perturbations.
- Gap relative to ViSA-MT: patch displacement repairs inputs through cross-region exchange; ViSA-MT would select among un-mixed transformations using teacher stability and student utility.
- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Chi_Adaptive_Bidirectional_Displacement_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2024_paper.html
- Code: https://github.com/chy-upc/ABD
