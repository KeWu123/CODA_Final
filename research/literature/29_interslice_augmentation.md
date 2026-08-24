# Inter-slice Image Augmentation

- Primary source: [arXiv:2001.11698](https://arxiv.org/abs/2001.11698)
- Core: synthesizes intermediate images and labels between adjacent medical slices for
  segmentation augmentation.
- Collision: simple adjacent-slice interpolation is not a new CVPR headline.
- Project consequence: virtual slab/acquisition-grid augmentation is a backup only and
  must explicitly model finite slice profile, sampling phase, and partial-volume
  occupancy to avoid collapsing to this work.

