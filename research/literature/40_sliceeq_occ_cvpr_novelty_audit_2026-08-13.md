# SliceEqOcc CVPR novelty audit

Date: 2026-08-13.  
Scope: primary/top-venue sources most relevant to the claimed augmentation and target semantics. This is a targeted collision audit, not proof of absolute firstness.

## Defensible novelty intersection

No single component is new: adjacent-slice augmentation, slice-profile modeling, partial-volume simulation, synchronized image/label transforms, soft labels, and EMA pseudo-label consistency all have precedents.

The defensible intersection is:

> training-only through-plane slice-profile re-acquisition that applies one non-invertible acquisition operator jointly to real neighboring MR signal and exact/teacher-derived tissue occupancy, producing spatial fractional supervision for a 2D semi-supervised model with unchanged single-slice inference.

## Closest SSL augmentation work

- Bai et al., **Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation**, CVPR 2023. BCP's defining contribution is labeled/unlabeled bidirectional Copy-Paste in a Mean Teacher framework. A version with Copy-Paste removed is a BCP-derived EMA pseudo-label scaffold, not BCP.  
  https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html

- Yang et al., **Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation**, CVPR 2023. UniMatch establishes dual strong views and feature perturbation under a common weak-view target. SliceEq must compare against it and distinguish acquisition-dependent target semantics from broader perturbation coverage.  
  https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html

- Zhao et al., **Augmentation Matters**, CVPR 2023. Occupies continuous strong augmentation and confidence-adaptive labeled injection; adaptive severity alone is not a SliceEq contribution.  
  https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html

- Chi et al., **Adaptive Bidirectional Displacement**, CVPR 2024. Confidence-guided reliable/unreliable patch displacement is a close BCP-line successor. Generic confidence/masking additions would weaken SliceEq's novelty.  
  https://openaccess.thecvf.com/content/CVPR2024/html/Chi_Adaptive_Bidirectional_Displacement_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2024_paper.html

- Hu et al., **beta-FFT**, CVPR 2025. Nonlinear Fourier interpolation and differentiated co-training occupy another data-diversity route and provide a direct PROMISE12 comparator.  
  https://openaccess.thecvf.com/content/CVPR2025/html/Hu_beta-FFT_Nonlinear_Interpolation_and_Differentiated_Training_Strategies_for_Semi-Supervised_Medical_CVPR_2025_paper.html

- Kumari and Singh, **Annotation Ambiguity Aware Semi-Supervised Medical Image Segmentation**, CVPR 2025. This is annotator/plausible-mask ambiguity; SliceEq should explicitly distinguish acquisition-induced measurement ambiguity.  
  https://openaccess.thecvf.com/content/CVPR2025/html/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.html

## Closest acquisition and partial-volume work

- Billot et al., **Partial Volume Segmentation of Brain MRI Scans of Any Resolution and Contrast**, MICCAI 2020. PV-SynthSeg simulates low-resolution images from high-resolution label maps using a partial-volume generative model. SliceEq cannot claim the first partial-volume simulation. Its distinction is paired re-acquisition of real neighboring images and teacher pseudo occupancy inside SSL.  
  https://arxiv.org/abs/2004.10221

- Billot et al., **SynthSeg**, Medical Image Analysis 2023. Domain randomization over contrast and resolution establishes broad acquisition-robust supervised generative training.  
  https://arxiv.org/abs/2107.09559

- Han et al., **MR Slice Profile Estimation by Learning to Match Internal Patch Distributions**, IPMI 2021. Establishes that MR slice-profile estimation itself is prior art and motivates careful calibration.  
  https://doi.org/10.1007/978-3-030-78191-0_9

- Wu et al., **Inter-Slice Image Augmentation Based on Frame Interpolation for Boosting Medical Image Segmentation Accuracy**, ECAI 2020. Jointly synthesizes intermediate images and labels from adjacent slices. SliceEq differs by integrating finite slice support, retaining fractional occupancy, and extending the operator to teacher pseudo masks.  
  https://doi.org/10.3233/FAIA200314

## Claim map

| Claim | Decision |
|---|---|
| first neighboring-slice augmentation | prohibited |
| first partial-volume/slice-thickness simulation | prohibited |
| first synchronized image-label augmentation | prohibited |
| first augmentation-dependent soft label | prohibited |
| first slice-profile model | prohibited |
| acquisition-aligned fractional pseudo occupancy in 2D SSL | defensible with matched controls |
| training uses volumetric acquisition physics, inference remains 2D | defensible with efficiency and 2.5D/3D controls |
| protocol-conditioned acquisition equivariance | prospective; requires metadata-shuffle and cross-protocol evidence |

## Related-work organization

1. Semi-supervised medical segmentation and perturbation design: BCP, UniMatch, AugSeg, ABD, beta-FFT.
2. Ambiguity and soft targets: distinguish annotation/model uncertainty from acquisition-derived occupancy.
3. Acquisition simulation and partial volume: PV-SynthSeg, SynthSeg, slice-profile estimation, inter-slice augmentation.
4. Gap: existing lines do not jointly define the target semantics of a non-invertible acquisition operator for exact and EMA pseudo supervision in a single-slice 2D SSL model.

