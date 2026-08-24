# H7.3 gate-driven literature boundary and pivot

Date: 2026-08-12

## Result-driven question

The H7.3 gate shows that acquisition-residual pixels already dominate the
SliceEqOcc gradient. The remaining empirical weakness is local pseudo-occupancy
magnitude, not spatial weighting. The literature was revisited to determine
whether generic soft/set-valued targets or uncertainty weighting are viable
headline directions.

## Primary top-venue boundaries

- Zhang and Boykov, *Soft Self-labeling and Potts Relaxations for
  Weakly-supervised Segmentation*, CVPR 2025.
  https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Soft_Self-labeling_and_Potts_Relaxations_for_Weakly-supervised_Segmentation_CVPR_2025_paper.html
  General soft pseudo-labeling is not novel by itself.
- Kumari and Singh, *Annotation Ambiguity Aware Semi-Supervised Medical Image
  Segmentation*, CVPR 2025.
  https://openaccess.thecvf.com/content/CVPR2025/html/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.html
  Diverse pseudo-label sets and latent ambiguity modeling are occupied.
- Chen et al., *ConformalSAM*, ICCV 2025.
  https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_ConformalSAM_Unlocking_the_Potential_of_Foundational_Segmentation_Models_in_Semi-Supervised_ICCV_2025_paper.pdf
  Calibrated prediction sets for semi-supervised segmentation are occupied.
- Fan et al., *UCC: Uncertainty Guided Cross-Head Co-Training*, CVPR 2022.
  https://openaccess.thecvf.com/content/CVPR2022/html/Fan_UCC_Uncertainty_Guided_Cross-Head_Co-Training_for_Semi-Supervised_Semantic_CVPR_2022_paper.html
  Generic uncertainty reweighting of pseudo-labels is occupied.
- Wang et al., *Self-Supervised Equivariant Attention Mechanism*, CVPR 2020.
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.html
  Generic transformation equivariance is established.
- Li et al., *Delving Aleatoric Uncertainty in Medical Image Segmentation via
  Vision Foundation Models*, CVPR 2026.
  https://openaccess.thecvf.com/content/CVPR2026/html/Li_Delving_Aleatoric_Uncertainty_in_Medical_Image_Segmentation_via_Vision_Foundation_CVPR_2026_paper.html
  Data-uncertainty filtering and dynamic loss weighting are now especially
  crowded as medical-segmentation contributions.

## Remaining narrow claim

The defensible next claim is not soft pseudo-labeling, uncertainty filtering,
or equivariance in general. It is commutation with a non-invertible physical
image-formation operator:

`f_student(A_h X) ~= A_h f_teacher(X)`.

Here `A_h` integrates real neighboring slices under a sampled MRI slice
profile and acts in the probability simplex on the corresponding teacher
posterior. The novelty boundary is the same acquisition operator acting on
observation and posterior before supervision. This must beat the current
hard/LCC occupancy and a generic raw-soft target under a no-training fidelity
gate before it is promoted.
