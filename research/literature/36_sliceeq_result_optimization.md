# SliceEq result-driven optimization and collision update

Date: 2026-08-11

## Literature facts used

- [ESPRESO (Magnetic Resonance Imaging, 2023)](https://doi.org/10.1016/j.mri.2023.01.012)
  models the through-plane MR slice profile as a PSF and reports that true profile/thickness
  can differ substantially from nominal metadata. This supports the physical motivation and
  argues against interpreting a universal sigma in slice units as a scanner-calibrated model.
- [PV-SynthSeg (MICCAI 2020)](https://arxiv.org/abs/2004.10221) simulates partial-volume
  low-resolution MRI from high-resolution label maps and learns a mapping back to high-
  resolution segmentation. It establishes strong prior art for partial-volume-aware training.
- [SynthSeg (Medical Image Analysis, 2023)](https://arxiv.org/abs/2107.09559) randomizes
  contrast and resolution and explicitly simulates acquisition direction, slice spacing, and
  slice thickness. SliceEq therefore cannot claim novelty for resolution/slice-thickness
  randomization alone.
- [Acquisition-invariant brain MRI segmentation (Medical Image Analysis, 2024)](https://arxiv.org/abs/2111.04094)
  casts MR acquisition simulation as augmentation and combines it with acquisition-aware
  representation learning. This further narrows generic "physics augmentation" claims.
- [AmbiSSL (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.pdf)
  models multiple-annotator ambiguity with diverse decoders and latent distributions. It is
  adjacent to soft/distributional targets but not to forward-model-derived tissue occupancy.
- [beta-FFT (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Hu_beta-FFT_Nonlinear_Interpolation_and_Differentiated_Training_Strategies_for_Semi-Supervised_Medical_CVPR_2025_paper.html)
  remains a close top-conference augmentation comparator on PROMISE12.

## Novelty boundary after the positive run

The defensible claim is not that MRI slices have thickness, that partial volume exists, or
that acquisition simulation improves generalization. All three are established. The narrower
open claim is:

> In semi-supervised segmentation, a sampled acquisition operator should act on both real
> neighboring image signal and detached GT/pseudo tissue occupancy, producing a physically
> aligned supervision distribution without changing the inference model.

The current hard-argmax v1 does not yet fully instantiate that claim because its target branch
is analytically likely to be inactive. Preserving fractional occupancy strengthens both the
mechanism and novelty boundary, but must be compared directly with image-only slab averaging,
ordinary soft pseudo-labels, and SynthSeg-style resolution augmentation.

## Result-driven next hypothesis

`H7.2`: Direct fractional-occupancy supervision, anchored by exact-GT paired labeled views,
improves the median case and acquisition-stratified boundary performance over hard-argmax
SliceEq, rather than merely amplifying Case05/09/34.

Falsification conditions:

- occupancy entropy is not localized to boundary/apex/base locations;
- the median paired Dice does not improve even if the mean rises;
- matched image-only slab averaging performs equivalently;
- benefit has no interaction with slice spacing/thickness or acquisition site;
- a same-checkpoint UniMatch or locked baseline matches the result.

