# Post-OBA Top-Conference Pivot Map

Date: 2026-08-11

## Decision context

- Locked legacy baseline: 2D U-Net EMA hard-pseudo-label self-training.
- User-reported baseline test Dice: 0.78--0.80.
- Archived CoDA combination: 0.819876, but four factors change together.
- Archived OBA: 0.818872, best validation at 13.8k followed by persistent deterioration to 30k.
- User-reported UniMatch comparator: approximately 0.83.

The next idea must therefore do more than beat the weak end of the legacy baseline range. It
must have a mechanism not already explained by generic dual strong views and must eventually
beat or complement UniMatch under one shared pretrained checkpoint.

## Occupied directions

| Direction | Closest primary work | Consequence |
|---|---|---|
| Two independent strong views / common weak target | UniMatch, CVPR 2023 | OBA cannot claim novelty from using two views; its antithetic pairing would need a direct IID win. |
| Adaptive augmentation strength or sample selection | AugSeg, CVPR 2023; SAA, ICCV 2023 | Do not rescue CoDA/OBA with confidence-driven severity or curriculum as the headline. |
| Confidence/hardness-guided medical patch perturbation | ABD and PH-Net, CVPR 2024 | Patch reliability/displacement is already crowded. |
| Supervised/unsupervised gradient conflict control | POS, ICCV 2025 | OBA collapse cannot be repackaged as a new loss-balancing paper. |
| Inter-slice continuity and volumetric pseudo-label refinement | SDC-UDA, CVPR 2023; DeSCO, CVPR 2023 | A generic adjacent-slice consistency loss or 2.5D model is insufficiently novel. |
| Generating intermediate slices and labels | Inter-slice image augmentation, ECAI 2020 | The pivot must not claim that adjacent-slice interpolation itself is new. |

## Surviving gap

MRI physics treats a reconstructed 2D slice as a slab observation: the scanner integrates
signal through a slice-selection profile before sampling. Slice-wise SSL, in contrast, treats
the central image and its target as a zero-thickness plane and applies generic 2D transforms
that either assume label invariance or move image geometry exactly.

The surviving hypothesis is **acquisition equivariance**: a stochastic acquisition operator
must act jointly on the image volume and the tissue-occupancy/teacher-mask volume. This is
different from:

- inter-slice interpolation, which synthesizes missing intermediate planes;
- 2.5D context models, which change the inference network;
- generic blur/downsampling, which leaves the target fixed;
- MRI contrast simulation, which primarily randomizes marginal appearance;
- confidence filtering, which decides whether to trust an existing pseudo-label.

The MRI basis is independently supported by slice-profile estimation work such as ESPRESO,
which models 2D multi-slice acquisition as through-plane convolution followed by sampling.
That physics is established; its use as a paired image/target semi-supervised augmentation is
the candidate contribution and remains unvalidated.

## Ranked candidates

| Candidate | Novelty boundary | Fit to locked code | Main rejection risk | Decision |
|---|---|---:|---|---|
| SliceEq: paired slice-profile re-acquisition | Joint acquisition of image and hard occupancy target | High after adding neighbor access | “Only slab averaging / mislabeled blur” | Primary, gated |
| Shape-trajectory pseudo-label projection | Project per-slice masks onto a smooth 3D path | Medium | SDC-UDA/registration/shape-prior collision | Backup |
| Checkpoint-orbit pseudo-label consensus | Multiple supervised snapshots estimate epistemic stability | High | Temporal ensemble/snapshot uncertainty is crowded | Reject as headline |
| CoDA repair | Replace uniform smoothing and disentangle factors | High | Becomes a different method; original mechanism unsupported | Comparator only |
| OBA repair | Ramp/BN/averaging/IID changes | High | Post-hoc and directly covered by prior work | Close |

## Primary sources

- UniMatch, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- AugSeg, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- SAA, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Gui_Enhancing_Sample_Utilization_through_Sample_Adaptive_Augmentation_in_Semi-Supervised_Learning_ICCV_2023_paper.html
- ABD, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Chi_Adaptive_Bidirectional_Displacement_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2024_paper.html
- POS, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.html
- SDC-UDA, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Shin_SDC-UDA_Volumetric_Unsupervised_Domain_Adaptation_Framework_for_Slice-Direction_Continuous_Cross-Modality_CVPR_2023_paper.html
- DeSCO, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Cai_Orthogonal_Annotation_Benefits_Barely-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- ESPRESO, Magnetic Resonance Imaging 2023: https://doi.org/10.1016/j.mri.2023.01.012
- E(3)-Pose, arXiv 2025: https://arxiv.org/abs/2512.04890
- C3, MedEurIPS workshop 2025: https://openreview.net/forum?id=Tle57HRCdB

## Latest adjacent-work qualification

E(3)-Pose includes acquisition/artifact augmentation for fetal MRI pose estimation, and C3
includes label-preserving physics counterfactuals such as slice-profile changes for robustness.
These works mean that neither “using a slice-profile perturbation” nor “clinically grounded
counterfactual augmentation” is a novelty claim. The still-unverified distinction is the joint
semi-supervised operator: SliceEq passes both the reconstructed volume and the GT/teacher-mask
occupancy through the same forward model, allowing the hard target to change at partial-volume
interfaces rather than assuming that the original mask remains invariant.
