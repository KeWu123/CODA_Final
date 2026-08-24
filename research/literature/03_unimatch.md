# UniMatch

- Paper: *Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation*.
- Venue/year: CVPR 2023.
- Core idea: a strong FixMatch-style segmentation baseline, two strong image perturbation streams, and a feature perturbation stream guided by a shared weak prediction.
- Relevance: proves that baseline engineering and perturbation diversity can dominate complex SSL designs; must be a primary comparator.
- Gap relative to ViSA-MT: views are generated according to a predefined scheme, not selected by separate validity and student utility signals.
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/LiheYoung/UniMatch
