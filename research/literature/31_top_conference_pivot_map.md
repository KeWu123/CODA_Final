# Top-conference pivot map after CoDA and BMER

This note prioritizes accepted CVPR/ICCV/ICLR/NeurIPS work and records why several
obvious pivots are not sufficiently open.

## Directly relevant papers

- **UniMatch, CVPR 2023.** Two independently perturbed strong views are supervised by
  one weak view. It establishes the strong two-view baseline that any paired-view idea
  must beat; merely doubling views is not novel.
- **AugSeg, CVPR 2023.** Uses randomized intensity transformations and confidence-
  adaptive labeled injection in a simple teacher-student framework. It occupies broad
  claims about simple adaptive strong augmentation.
- **SAA, ICCV 2023.** Adjusts augmentation to sample learning status/difficulty. It
  narrows novelty for sample-adaptive severity or policy selection.
- **iMAS, CVPR 2023.** Uses model-adaptive supervision and augmentation strength based
  on instance hardness. It further closes the "choose the maximum safe severity" route.
- **TeachAugment, CVPR 2022.** Searches transformations that are difficult for a target
  model but recognizable to a teacher. A teacher-stable/student-hard augmentation is
  therefore not a new core claim.
- **Conflict-Based Cross-View Consistency, CVPR 2023.** Exploits conflict between two
  learned views in co-training. Generic conflicting/complementary-view language is
  occupied.
- **Diverse Co-training, ICCV 2023.** Grounds co-training gains in compatible and
  conditionally independent views across input, augmentation, and architecture axes.
  View diversity alone is not a sufficient contribution.
- **Two Losses, One Goal (POS), ICCV 2025.** Directly formulates supervised versus
  unsupervised gradient conflict as Pareto optimization. Post-hoc gradient projection or
  reweighting should be a comparator, not the new augmentation headline.
- **A Simple Baseline with Strong Data Augmentation, ICCV 2021.** Shows that strong-view
  distribution shift can corrupt BatchNorm and proposes distribution-specific BN. This
  is particularly relevant because the current CoDA/BMER code concatenates branches
  through BatchNorm, but correcting it is a shared-baseline audit rather than novelty.
- **beta-FFT, CVPR 2025.** Uses FFT nonlinear interpolation plus differentiated training
  for semi-supervised medical segmentation. Generic frequency/style mixing is crowded.
- **AmbiSSL, CVPR 2025.** Models annotation ambiguity with diverse pruned decoders and a
  latent mask distribution. Boundary ambiguity is important, but the current data lack
  multi-annotator supervision and this would abandon the fixed baseline.
- **Antithetic Noise in Diffusion Models, ICLR 2026.** Shows that `z/-z` initial noise
  produces negatively correlated diffusion samples and supports variance reduction.
  It supplies a modern precedent for antithetic design but does not study training-time
  augmentation, hard pseudo-labels, or dense prediction. OBA must cite it and cannot
  claim invention of antithetic sampling.

## Synthesis

The literature leaves little room for another confidence selector, learned policy,
dual random view, gradient projection, frequency mixer, or boundary generator. A
narrower open question remains: **does the sampling design of a transformation orbit,
independent of its marginal augmentation distribution, control the bias and variance of
hard-pseudo-label training?** OBA targets this question through paired antithetic
quadrature and explicit IID/shuffled-pair controls.

## Primary-source links

- UniMatch: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- AugSeg: https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- SAA: https://openaccess.thecvf.com/content/ICCV2023/html/Gui_Enhancing_Sample_Utilization_through_Sample_Adaptive_Augmentation_in_Semi-Supervised_Learning_ICCV_2023_paper.html
- TeachAugment: https://openaccess.thecvf.com/content/CVPR2022/html/Suzuki_TeachAugment_Data_Augmentation_Optimization_Using_Teacher_Knowledge_CVPR_2022_paper.html
- CCVC: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Conflict-Based_Cross-View_Consistency_for_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html
- Diverse Co-training: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Diverse_Cotraining_Makes_Strong_Semi-Supervised_Segmentor_ICCV_2023_paper.html
- POS: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.html
- Strong-augmentation baseline: https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_A_Simple_Baseline_for_Semi-Supervised_Semantic_Segmentation_With_Strong_Data_ICCV_2021_paper.html
- beta-FFT: https://openaccess.thecvf.com/content/CVPR2025/html/Hu_beta-FFT_Nonlinear_Interpolation_and_Differentiated_Training_Strategies_for_Semi-Supervised_Medical_CVPR_2025_paper.html
- AmbiSSL: https://openaccess.thecvf.com/content/CVPR2025/html/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.html
- Antithetic Noise: https://openreview.net/pdf/5b53a47d4524e67cec163aed1d224599599800bf.pdf
