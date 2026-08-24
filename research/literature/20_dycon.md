# DyCON

- Paper: *Dynamic Uncertainty-aware Consistency and Contrastive Learning for Semi-supervised Medical Image Segmentation*.
- Venue/year: CVPR 2025.
- Core idea: an uncertainty-aware consistency loss dynamically changes voxel contributions during training, complemented by focal entropy-aware contrastive learning.
- Relevance: confirms that high-uncertainty voxels should not simply be discarded and that uncertainty dynamics matter in medical SSL.
- Difference from CoDA-MT: DyCON derives weights from model predictive uncertainty. CoDA-MT proposes to transform the target distribution using the information loss of the augmentation channel.
- Collision caution: avoid describing CoDA-MT as the first uncertainty-aware consistency method or the first method to retain uncertain voxels.
- Paper: https://openaccess.thecvf.com/content/CVPR2025/html/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.html

