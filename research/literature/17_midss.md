# MiDSS

- Paper: *Constructing and Exploring Intermediate Domains in Mixed Domain Semi-supervised Medical Image Segmentation*.
- Venue/year: CVPR 2024.
- Core idea: unified Copy-Paste constructs intermediate semantic domains; symmetric guidance merges supervision; training-aware random amplitude MixUp progressively bridges styles.
- Reported result: large gains on a prostate mixed-domain setting.
- Relevance: the main collision for using PROMISE12 multi-center variation plus Fourier style mixing.
- Gap relative to ViSA-MT: ViSA-MT should not claim intermediate-domain construction and should avoid cross-patient pixel/amplitude transfer.
- Paper: https://arxiv.org/abs/2404.08951
- Code: https://github.com/MQinghe/MiDSS
