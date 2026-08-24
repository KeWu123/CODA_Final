# H7.5 SliceEqSAQ full implementation protocol

## Status and evidence class

Implemented on 2026-08-12 at the user's explicit request to skip the previously
planned zero-training quadrature gate. The first full seed-1337 run is therefore
**exploratory and gate-skipped**, not confirmatory evidence for H7.5.

## Frozen parent

`train_sliceeq_occ.py`, `test_sliceeq_occ.py`, `utils/sliceeq.py`,
`utils/sliceeq_occ.py`, and the stack dataset remain unchanged. Preserve the
fixed PROMISE12 root, ordered split, labelnum 7, seed 1337, fixed Pre10000
network+optimizer checkpoint, 2D U-Net, EMA 0.99, SGD, 30k iterations,
loader batch 24, labeled batch 12, first-1k identity warmup, LCC pseudo masks,
fractional occupancy targets, CE+Dice losses, consistency ramp, validation,
checkpoint selection, and inference graph.

## Sole intervention

Replace independent uniform profile draws after warmup with the 2x2
Gauss-Legendre nodes for the unchanged uniform ranges:

- sigma: `0.5345299462`, `0.7654700538`;
- phase: `-0.1443375673`, `+0.1443375673`;
- four Cartesian nodes, each assigned exactly three times in each 12-sample
  labeled or unlabeled branch;
- independently seeded `randperm` assignments for L and U at every step.

Node assignment is independent of images, anatomy, masks, teacher confidence,
loss, and training history. Each sample still has one re-acquired view. The
student forward remains exactly 36 images and there is one optimizer/EMA update.

## Isolation

- train: `code/train_sliceeq_saq.py`
- test: `code/test_sliceeq_saq.py`
- sampler: `code/utils/sliceeq_saq.py`
- experiment id: `SliceEqSAQ_PROMISE12`
- default output: `../model/SliceEqSAQ_PROMISE12_7_labeled/self_train/unet`
- test prediction saving: disabled by default
- checkpoint auto-search: disabled

The parent SliceEqOcc sources are protected by SHA-256 contract tests.

## Required interpretation

A positive result supports lower-discrepancy physical profile coverage at
identical forward cost. It does not retroactively pass the skipped H7.5 gate,
does not establish quadrature as standalone novelty, and does not repair the
known validation/test checkpoint-selection issue. A negative result closes
H7.5 without node tuning.
