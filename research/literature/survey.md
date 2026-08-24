# Literature Survey

## Landscape Map

| Direction | Representative work | What it already covers | Consequence for this project |
|---|---|---|---|
| EMA teacher baseline | Mean Teacher, NeurIPS 2017 | Weight-averaged teacher and consistency under independent noise | The current code is only BCP-derived EMA hard-pseudo-label self-training; it is neither BCP nor canonical Mean Teacher |
| Cross-set mixing | BCP, CVPR 2023 | Bidirectional labeled/unlabeled Copy-Paste and distribution-gap argument | Do not propose another rectangular or mask-based paste as the core idea |
| Strong multi-view perturbation | UniMatch, CVPR 2023 | Two strong image streams plus feature perturbation | Must compare to a tuned weak-to-strong baseline |
| Strong-augmentation BN shift | Simple strong-augmentation baseline, ICCV 2021 | Strong views can corrupt shared BN statistics | Match effective batch/BN exposure in every OBA control |
| Multi-augmentation prediction averaging | MixMatch, NeurIPS 2019 | Average predictions over multiple augmented views | A plain OBA prediction barycenter is a diagnostic, not standalone novelty |
| Unreliable pseudo-label handling | U2PL, CVPR 2022 | Avoid treating uncertain pseudo-labels as ordinary hard positives | Antithetic views cannot cancel shared teacher-label error |
| Augmentation-centric SSL | AugSeg, CVPR 2023 | Continuous-strength intensity transforms and confidence-adaptive labeled injection | Confidence-adaptive strength alone is not novel |
| Reliability-guided mixing | ABD, CVPR 2024 | Replace unreliable regions with reliable regions across views; inverse strategy on labels | Region confidence and patch exchange are crowded |
| Patch rearrangement | PSC, MICCAI 2024 | Labeled-unlabeled and unlabeled-unlabeled patch shuffling | Patch shuffling has artifact/novelty concerns |
| Shape/edge overlay | OMF, MICCAI 2024 | Overlay augmentation and differentiated teacher/student inputs | Edge-aware mixing is already occupied |
| Multi-domain augmentation | MiDSS, CVPR 2024 | Copy-Paste intermediate domains plus progressive Fourier amplitude style transfer | Multi-center style transfer must avoid merely reusing Fourier/mixing ideas |
| PROMISE12 fusion | SF-DA, ICASSP 2025 | Self-symmetric flipping plus cross-sample stitching on ACDC and PROMISE12 | Direct collision with flip+fusion proposals |
| Learning-progress adaptation | DPCL, 2025/2026 journal publication | Difficulty- and progress-aware image perturbation plus feature perturbation; reports PROMISE12 | Progress-based magnitude scheduling is not enough |
| Anatomy-preserving intensity | Anatomy-Preserving Consistency Training, IVC 2026 | Fixed nonlinear grayscale perturbation plus feature/output consistency | "Preserve anatomy with intensity transforms" alone is occupied |
| Frequency consistency | FRCNet, MICCAI 2024 and follow-ups | Frequency and region consistency | Fourier/wavelet novelty is crowded |
| Current generative medical SSL | SemiGDA, CVPR 2026 | Image/mask dual-distribution generative alignment | Current top-venue baseline; not augmentation-focused but important |
| Current uncertainty SSL | HESS, CVPR 2026 | Dirichlet evidential uncertainty for pseudo-label selection | Softmax confidence is no longer a sufficient reliability contribution |
| Dynamic uncertainty consistency | DyCON, CVPR 2025 | Reweights voxel consistency and contrastive learning by predictive uncertainty | Uncertainty weighting is occupied; target transformation must be distinguished from loss weighting |
| Coupled degradation and label entropy | Supervised Mollification, AISTATS 2025 | Couples input noise/blur severity with label smoothing for supervised classification | Direct conceptual support and closest collision; the new work must contribute dense, spatial, pseudo-label-specific coupling |
| Spatial label smoothing | GeoLS, MIDL 2024 | Image-gradient/geodesic soft labels for supervised medical segmentation | Spatial or image-aware smoothing alone is occupied |
| Soft pseudo-labels | Enhanced Soft Label, ICCV 2023 | Dynamic dominant-class soft targets for semi-supervised segmentation | Soft pseudo-labels alone are occupied; augmentation-channel conditioning is the differentiator |
| Pre/post augmentation regional supervision | SGRS-Net, MICCAI 2025 | Synergy-guided region partition after mix augmentation | Transformation-response selection is close to ViSA; avoid making it the headline |
| Generative augmentation | MatchMask, CVPR 2026 | Mask-centric diffusion augmentation and semi-supervised extension | A diffusion generator would require a substantially distinct medical contribution |
| Risk-controlled augmentation | Conformal Data Augmentation, ICLR 2026 desk-rejected submission | Conformal filtering of synthetic data in classification/table tasks | Useful adjacent concept, but not yet semi-supervised dense view selection |
| Learned task-driven augmentation | Semi-supervised Task-driven DA, MedIA 2021 | Learns full-image additive intensity and deformation fields from labeled and unlabeled data | BMER must not claim that unlabeled-driven intensity augmentation itself is new |
| Foreground/background preservation | KeepMask/KeepMix, ISBI 2023 / IVC 2024 | Perturbs background or composes foreground while protecting organ evidence | BMER must actively model the two-sided interface rather than merely preserve foreground |
| Foreground harmonization | ARHNet, MLMI 2023 | Whole-foreground affine intensity perturbation plus a learned boundary harmonizer | A mask-affine or generic local-contrast BMER implementation would be incremental |
| Boundary-region replacement | BoundaryMix, Pattern Recognition 2021 | Removes/replaces risky predicted-boundary pixels and mixes pseudo-labels | Do not cut, paste, or replace a Cartesian boundary patch |
| Boundary contrastive SSL | BoCLIS, TMI 2025 | Samples boundary patches with conservative/radical teachers for contrastive learning | “Focus on the boundary” is not novel; BMER's contribution must be input evidence resynthesis |
| Current Fourier augmentation | beta-FFT, CVPR 2025 | Nonlinear low-frequency exchange and differentiated co-training, including PROMISE12 | Global/frequency-style transfer is not the headline gap |
| Through-plane synthesis | Inter-slice augmentation (2020), PV-SynthSeg (2020/2021), SynthSeg | Interpolates slices or simulates resolution/partial volume | Virtual slab/grid-phase re-acquisition is a backup with a narrower novelty boundary |

## Current Candidate Gap (2026-08-11 Pivot)

Earlier ViSA/CoDA gaps are retained as research history but are no longer the primary
direction. Current code audit and the user's exploratory negative result triggered an
outer-loop pivot.

The targeted 2020--2026 search did not find a method with the complete BMER definition:

- unwrap a segmentation interface into object-relative normal/tangential coordinates;
- estimate its empirical evidence-field distribution from the unlabeled set;
- resynthesize that field on labeled recipient anatomy while preserving residual
  texture, exact hard GT, and all pixels outside the boundary band;
- leave the self-training network, teacher, pseudo-target, loss, and inference graph
  unchanged.

This is a search result, not proof of first publication. The gap collapses if BMER is
implemented as whole-foreground affine jitter, scalar boundary contrast, histogram
matching, learned global intensity fields, or boundary patch replacement. An oracle
intervention must first show that the complete normal profile produces an ordered,
localized model response beyond matched simple controls.

## Most Relevant Primary Sources

- [BCP, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html)
- [UniMatch, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)
- [AugSeg, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)
- [ABD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chi_Adaptive_Bidirectional_Displacement_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2024_paper.html)
- [MiDSS, CVPR 2024](https://arxiv.org/abs/2404.08951)
- [OMF, MICCAI 2024](https://papers.miccai.org/miccai-2024/585-Paper0481.html)
- [PSC, MICCAI 2024](https://papers.miccai.org/miccai-2024/586-Paper0942.html)
- [FRCNet, MICCAI 2024](https://papers.miccai.org/miccai-2024/340-Paper0245.html)
- [SF-DA, ICASSP 2025](https://doi.org/10.1109/ICASSP49660.2025.10889068)
- [SemiGDA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_SemiGDA_Generative_Dual-distribution_Alignment_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2026_paper.html)
- [HESS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_From_Softmax_to_Dirichlet_Evidential_Learning_for_Semi-supervised_Semantic_Segmentation_CVPR_2026_paper.html)
- [DyCON, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.html)
- [Supervised Mollification, AISTATS 2025](https://proceedings.mlr.press/v258/heinonen25a.html)
- [GeoLS, MIDL 2024](https://proceedings.mlr.press/v227/vasudeva24a.html)
- [Enhanced Soft Label, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Enhanced_Soft_Label_for_Semi-Supervised_Semantic_Segmentation_ICCV_2023_paper.html)
- [SGRS-Net, MICCAI 2025](https://papers.miccai.org/miccai-2025/0890-Paper1721.html)
- [MatchMask, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_MatchMask_Mask-Centric_Generative_Data_Augmentation_for_Label-Scarce_Semantic_Segmentation_CVPR_2026_paper.html)
- [PROMISE12 challenge paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4137968/)
- [Semi-supervised Task-driven Data Augmentation, MedIA 2021](https://arxiv.org/abs/2007.05363)
- [KeepMask/KeepMix, IVC 2024](https://www.sciencedirect.com/science/article/pii/S0262885624001604)
- [ARHNet, MLMI 2023](https://arxiv.org/abs/2307.01220)
- [BoundaryMix, Pattern Recognition 2021](https://doi.org/10.1016/j.patcog.2021.107924)
- [BoCLIS, TMI 2025](https://www.isee-ai.cn/~wangruixuan/files/TMI2025Yang.pdf)
- [beta-FFT, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_beta-FFT_Nonlinear_Interpolation_and_Differentiated_Training_Strategies_for_Semi-Supervised_Medical_CVPR_2025_paper.pdf)
- [Inter-slice Image Augmentation](https://arxiv.org/abs/2001.11698)
- [PV-SynthSeg](https://arxiv.org/abs/2004.10221)
- [FDIF, 2026](https://arxiv.org/abs/2603.23199)
- [MR Slice Profile Estimation](https://arxiv.org/abs/2104.00100)
- [TeachAugment, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Suzuki_TeachAugment_Data_Augmentation_Optimization_Using_Teacher_Knowledge_CVPR_2022_paper.html)
- [Diverse Co-training, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Diverse_Cotraining_Makes_Strong_Semi-Supervised_Segmentor_ICCV_2023_paper.html)
- [Conflict-Based Cross-View Consistency, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Conflict-Based_Cross-View_Consistency_for_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)
- [Two Losses, One Goal, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.html)
- [AmbiSSL, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.html)
- [Antithetic Noise in Diffusion Models, ICLR 2026](https://openreview.net/pdf/5b53a47d4524e67cec163aed1d224599599800bf.pdf)

The post-BMER top-conference collision map and OBA novelty boundary are recorded in
`31_top_conference_pivot_map.md`.

## Search Caveat

This is a targeted bootstrap survey, not a completed systematic review. Search emphasized primary conference pages, papers, and official repositories. Journal/preprint collisions were included when they directly threatened the proposed novelty.
