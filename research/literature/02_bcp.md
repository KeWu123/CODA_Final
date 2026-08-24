# Bidirectional Copy-Paste (BCP)

- Venue/year: CVPR 2023.
- Core idea: paste labeled into unlabeled and unlabeled into labeled samples in both directions, mixing ground truth and EMA-teacher pseudo-labels to reduce empirical distribution mismatch.
- Reported scope: ACDC, LA, and later NIH Pancreas code.
- Relevance: the user's code lineage and strongest nearby collision.
- Key warning: Copy-Paste is not an optional peripheral module; it is the paper's defining contribution and is also used in released pretraining.
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/DeepMed-Lab-ECNU/BCP
