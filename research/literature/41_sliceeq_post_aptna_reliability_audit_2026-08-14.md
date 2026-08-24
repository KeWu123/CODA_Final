# SliceEq post-APTNA reliability novelty audit

## Literature boundary

Generic stochastic uncertainty, pseudo-label reliability, and dynamic uncertainty weighting are
already occupied. UA-MT uses uncertainty-aware self-ensembling for semi-supervised medical
segmentation; ST++ uses prediction stability to select reliable pseudo labels; DyCON dynamically
weights consistency under uncertainty. These works prohibit claims such as "first uncertainty
weighting", "first MC-dropout teacher", or "first stable pseudo-label selection".

Primary sources:

- UA-MT (MICCAI 2019): https://link.springer.com/chapter/10.1007/978-3-030-32245-8_67
- ST++ (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/html/Yang_ST_Make_Self-Training_Work_Better_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.html
- DyCON (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.html

## Defensible SliceEq-specific statement

The narrow unresolved distinction is between two sources of soft occupancy under the same
non-invertible measurement operator:

- **measurement ambiguity**, which is the intended fractional tissue occupancy and must remain
  supervised even when its entropy is high; and
- **model instability**, which arises when stochastic teacher states disagree and is potentially
  reducible.

A safe contribution, only if the locked gate and full run are positive, is: *operator-space
reliability that preserves acquisition-induced occupancy while suppressing stochastic teacher
instability*. SCT is the lower-complexity realization; ADU is the more explicit uncertainty
decomposition. Neither should be marketed independently from the main SliceEqOcc operator.

Poly learning-rate decay, checkpoint averaging, SWA, 3-D LCC, and ordinary confidence thresholds
remain training/post-processing controls rather than paper contributions.

