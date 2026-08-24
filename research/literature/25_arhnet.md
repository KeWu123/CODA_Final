# ARHNet

- Primary source: [ARHNet, MLMI 2023](https://arxiv.org/abs/2307.01220)
- Core: affine intensity perturbation over the entire lesion foreground followed by a
  learned boundary-aware harmonization generator.
- Collision: the closest threat to any mask-conditioned intensity-jitter proposal.
- BMER distinction: no whole-foreground affine transform, no harmonizer/generator, and
  no objective to erase paste artifacts; it samples empirical two-sided normal profiles
  from unlabeled data and renders them on fixed GT anatomy.

