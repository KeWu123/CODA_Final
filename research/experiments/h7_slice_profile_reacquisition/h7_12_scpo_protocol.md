# H7.12 Slab-Coherent Pseudo Occupancy protocol

## Status

Locked after the neutral H7.11 result and before implementation or training.
This is the only authorized next full method experiment. It is an exploratory
test, not a claim that generic 3-D post-processing or inter-slice consistency
is novel.

## Problem and hypothesis

SliceEqOcc models a through-plane measurement by integrating three neighboring
source occupancies. Its current U target first applies largest-connected-
component filtering independently to each 2-D teacher slice. The retained
component can therefore jump between disconnected structures across the three
slices, even though the subsequent acquisition operator assumes one latent
anatomical slab.

**Hypothesis:** forming one 26-connected foreground component over each
three-slice pseudo slab before profile integration removes systematic
slice-to-slice pseudo components that two-pass dropout disagreement cannot
detect. This should improve acquisition-target coherence and late stability
without altering the SliceEqOcc image operator or network.

## Single method change

After the unchanged 1k warmup:

1. Run the original single train-mode EMA teacher forward on the same flattened
   `B_U x 3` unlabeled input.
2. Take hard argmax exactly as the parent.
3. Reshape to `B_U x 3 x H x W` and retain the largest foreground component
   in each three-slice slab using fixed 26-connectivity.
4. Apply the unchanged sampled SliceEq profile jointly to the original image
   stack and this slab-coherent hard pseudo occupancy.

There is no 3-D network, probability averaging, morphology radius, confidence
threshold, component-size threshold, extra forward, or inference
post-processing. Empty slabs remain empty. The exact-GT labeled branch is not
filtered or changed.

## Frozen parent contract

- PROMISE12 root and seven-patient/191-slice labeled prefix;
- seed 1337, 30k iterations, deterministic mode, SGD and EMA 0.99;
- shared Pre10000 net+optimizer SHA-256
  `49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9`;
- loader batch24 and student batch36: 12 original-L, 12 reacquired-L,
  12 reacquired-U;
- one teacher forward, one student forward, one optimizer update, and one EMA
  update per post-warmup iteration;
- radius 1, sigma `[0.45,0.85]`, phase `[-0.25,0.25]`, profile RNG streams,
  full Occ coefficient, loss, warmup, and train-mode teacher;
- parent validation calculation/checkpoint selection and strict 2-D inference.

## Diagnostics and decision

Log the fraction of U pseudo pixels changed relative to the parent per-slice
2-D LCC, removed/added foreground mass, number of raw/slab components, and the
existing SliceEqOcc diagnostics. These observables do not tune the run.

The matched parent best validation is `0.817373`; exploratory success requires
best validation `>=0.820373`. Test only the validation-selected SCPO checkpoint
once. The desired test Dice `>0.844566` is a result criterion, never a
checkpoint selector. If SCPO is negative, do not search connectivity,
morphology, component thresholds, or slab widths; close pseudo-topology
optimization and return to the SliceEqOcc paper evidence matrix.

## Publication boundary

SDC-UDA and other volumetric/self-training work already establish that
slice-direction continuity and volumetric pseudo labels can help medical image
segmentation. SCPO cannot be claimed as the first 3-D connected-component or
inter-slice consistency method. The defensible role, if positive, is narrower:
the latent source occupancy of a non-invertible through-plane acquisition must
be formed coherently before applying the paired measurement operator. SCPO is
a secondary target-construction component under SliceEqOcc, not a standalone
CVPR contribution.
