# SliceEq: Paired Slice-Profile Re-Acquisition

Status: final fixed-seed implementation completed; no training result or positive result claimed.
`SliceEq` is a working name and can change before a paper submission.

## Core observation

The current 2D pipeline discards the acquisition process along the slice axis. A clinical MRI
slice is not an infinitesimal plane: it integrates signal over a slab according to a
slice-selection point-spread function and is then sampled. CoDA and OBA modify the observed
image while retaining a target formed on another evidence distribution. SliceEq instead makes
the acquisition transformation explicit and applies it to image and target together.

## Operator

For a volume `X`, a one-hot GT or detached EMA/LCC mask volume `Y`, central index `z`, and a
sampled nonnegative normalized slice profile `h_phi`, define

`x_phi(z) = sum_k h_phi(k) X[z+k]`

`o_phi(z) = sum_k h_phi(k) one_hot(Y[z+k])`

`y_phi(z) = argmax_c o_phi(z,c)`.

Optional in-plane PSF and lattice resampling must be applied consistently in physical
coordinates. The output target remains hard, so the inherited CE+Dice implementation does
not need to change. For unlabeled volumes, `Y` is built from detached teacher predictions and
the baseline's existing LCC cleanup before the shared operator is applied.

This is a paired forward model, not image-only blur. The same sampled acquisition parameters
must be used for `X` and `Y`.

## Baseline contract

- Keep the existing 2D U-Net and all inference code.
- Keep seed 1337, label split/order, the inherited 30k self-training schedule, EMA, hard CE+Dice, consistency ramp,
  batch proportions, optimizer, and validation/test lists.
- Load exactly one explicit pretrained checkpoint containing `net` and `opt`; record its
  SHA-256 and reset self-training RNG.
- Add only volume-neighbor lookup and the SliceEq training transform. The supervised branch
  remains the original central-slice anchor; SliceEq acts on the post-warmup unlabeled pair.
- Do not copy code ideas from Desktop/Baseline experiments; only reproduce the explicit
  single-checkpoint path contract.

## Why it is not the discarded methods

- Unlike CoDA, the target is not softened toward a uniform class prior.
- Unlike OBA, two views are not the contribution and no cancellation claim is needed.
- Unlike BMER, no donor appearance profile is transplanted onto another patient.
- Unlike ordinary 2.5D input, adjacent slices are used only to simulate an acquisition; the
  network still receives one channel/slice at train and test time.
- Unlike inter-slice interpolation, SliceEq does not create an allegedly missing anatomical
  plane. It changes slice thickness/profile and derives its occupancy target through the same
  operator.

## Falsification gates

### H7.1: zero/full-training-free operator validity

On labeled volumes, sweep a preregistered small profile family and compare:

1. joint image+target re-acquisition;
2. image-only slab averaging with the original central target;
3. matched 2D Gaussian blur;
4. identity.

Measure central-target Dice, changed-label fraction, change distance to the GT boundary, and
apex/mid/base strata. Reject SliceEq if the joint target changes are not boundary-localized, if
plausible profiles destroy central anatomy, or if the operator is numerically indistinguishable
from matched 2D blur.

### First training screen

Reuse one fixed checkpoint and identical self-training RNG/data order:

- locked baseline;
- generic matched 2D blur/down-up;
- SliceEq through-plane only;
- UniMatch strong comparator (user reports approximately 0.83).

The full SliceEq run is justified only after H7.1. For the fixed-seed exploratory run, advance
only if SliceEq exceeds CoDA/OBA and is competitive with or improves upon UniMatch, while not
showing the sustained late collapse seen in OBA. A gain confined to one test case or one
validation spike is insufficient.

## CVPR-level claim boundary

The publishable thesis is not “MRI augmentation helps.” It is:

> Semi-supervised segmentation augmentations should be equivariant to the image-formation
> operator: acquisition changes must transform observations and tissue-occupancy targets
> together.

To support a CVPR submission, the final study must demonstrate the operator on more than
PROMISE12, include a non-prostate anisotropic MRI/CT volume dataset, compare with UniMatch and
modern medical SSL methods, and report acquisition-stratified and boundary metrics. A
single-seed PROMISE12 improvement is only a direction-selection result.
