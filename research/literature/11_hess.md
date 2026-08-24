# HESS Evidential Semi-Supervised Segmentation

- Paper: *From Softmax to Dirichlet: Evidential Learning for Semi-Supervised Semantic Segmentation*.
- Venue/year: CVPR 2026.
- Core idea: model class probabilities with Dirichlet distributions and separate exclusive/collective evidence to improve pseudo-label uncertainty estimates.
- Relevance: any reliability claim based only on maximum softmax probability will look outdated.
- Gap relative to ViSA-MT: HESS selects pseudo-labels; it does not characterize augmentation validity or learning utility. HESS uncertainty could later replace the base confidence term, but should not be part of the first minimal method.
- Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Mai_From_Softmax_to_Dirichlet_Evidential_Learning_for_Semi-supervised_Semantic_Segmentation_CVPR_2026_paper.html
