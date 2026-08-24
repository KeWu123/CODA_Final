# Enhanced Soft Label

- Paper: *Enhanced Soft Label for Semi-Supervised Semantic Segmentation*.
- Venue/year: ICCV 2023.
- Core idea: Dynamic Soft Labels retain a subset of dominant classes for low-confidence pixels, combined with pixel-to-part contrastive learning to preserve discrimination.
- Relevance: establishes soft pseudo-label use in semi-supervised semantic segmentation.
- Collision risk: CoDA cannot claim the first soft pseudo-label method in semi-supervised dense prediction.
- Required distinction: ESL derives softness from prediction/class ambiguity and training curriculum. CoDA derives it from augmentation-induced evidence loss at the corresponding spatial location.
- Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Enhanced_Soft_Label_for_Semi-Supervised_Semantic_Segmentation_ICCV_2023_paper.html

