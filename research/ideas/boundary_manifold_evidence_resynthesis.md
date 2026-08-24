# BMER: Boundary-Manifold Evidence Resynthesis

Status: **primary pivot candidate; not yet experimentally validated**  
Date: 2026-08-11

## Two-sentence pitch

Most segmentation augmentation acts in Cartesian image coordinates: it changes global
style, corrupts pixels, or moves spatial content. BMER instead learns the empirical
distribution of image evidence in object-relative boundary coordinates from all
unlabeled cases, then resynthesizes that evidence on labeled anatomy while leaving the
geometry and hard ground truth unchanged.

BCP is only an innovation-level reference for this project. BMER is not bidirectional,
does not Copy-Paste or construct mixed samples, and does not inherit BCP's method logic.

## Core operator

For an image-mask pair `(x, m)`, unwrap a narrow boundary ribbon into normal-tangential
coordinates:

`E_{x,m}(s, rho, z) = standardized low-pass intensity of x(Phi_m(s,rho))`, 

where `rho` is signed distance to the boundary, `s` is normalized arclength/surface
position, and `z` is normalized longitudinal position when a volume is available.

After supervised pretraining, use the baseline model's detached LCC masks on all
unlabeled images to construct an empirical bank

`P_U(E | z, curvature, foreground class)`.

Normal gradient, two-sided contrast, and transition width are derived descriptors used
for conditioning and diagnostics; they are not separately rendered channels. Local
two-sided scalar profiles are sampled from the bank, smoothly interpolated along the
labeled recipient boundary, converted to recipient median/MAD units, and rendered with
a taper `w(R)=0`:

`x'_l(v) = x_l(v) + w(|d_l(v)|) [E_U(s_l(v),d_l(v),z_l) - E_l(s_l(v),d_l(v),z_l)]`.

Here `x_l-E_l` is the recipient high-frequency/residual component within the ribbon.
It is retained; pixels outside the band are identical, and
`y'_l = y_l`. There is no Cartesian donor patch, donor shape, or donor label transfer.
The first implementation should use a non-parametric profile bank and smooth bootstrap;
it should not add a generator, policy network, or learned loss.

## Locked-baseline integration

The user's current baseline remains the experimental substrate:

- the same 2-D U-Net and BatchNorm/Dropout behavior;
- the same supervised pretraining and self-training schedule;
- the same EMA teacher, train/eval mode, LCC hard pseudo-label, CE+Dice losses,
  consistency ramp, sampler, label fraction, optimizer, seed policy, and evaluation
  cases;
- the original unlabeled inputs and original hard pseudo-target path.

BMER changes only the labeled training input after the pretraining checkpoint has
created the frozen unlabeled evidence bank:

`x_l -> A_BMER(x_l, y_l; bank_U)`.

No target, network, loss, teacher update, or inference path changes. Any evaluation bug
fix must be applied to every checkpoint and reported alongside the legacy metric; it is
not part of the method.

## Motivation

1. The segmentation decision is made at an organ interface, but global style
   augmentation models marginal `p(x)` rather than conditional
   `p(evidence | signed distance to boundary)`.
2. A small labeled subset under-samples real boundary contrast, transition width,
   partial-volume pattern, and adjacent-tissue texture. The larger unlabeled subset can
   estimate this nuisance support without using those pixels as new labels.
3. Generic degradation plus target smoothing, as in the current CoDA attempt, removes
   evidence and weakens supervision. BMER instead creates new exact-GT examples of
   boundary appearance variation.
4. PROMISE12 is multi-center, multi-vendor, and multi-protocol; its official analysis
   explicitly reports sensitivity of appearance/resolution to scanners and protocols,
   while apex/base boundaries are known failure locations. This motivates—but does not
   prove—the boundary-evidence hypothesis.

## Conditional paper contributions

Only if the pre-registered tests succeed:

1. Reframe segmentation augmentation from image-coordinate perturbation to sampling on
   the object boundary manifold.
2. Introduce a parameter-free empirical resynthesis operator that uses unlabeled data
   to broaden exact-GT boundary evidence without changing anatomy or targets.
3. Establish a mechanistic link between conditional boundary-evidence coverage and
   boundary error through controlled oracle interventions, not only final Dice.
4. Demonstrate the same operator on at least three binary/single-organ benchmarks
   spanning MRI and CT, including genuinely 3-D segmentation, with no inference
   overhead and an unchanged self-training backbone.

## Novelty boundary

The defensible novelty is the simultaneous combination of:

- object-relative `(s, rho, z)` boundary-manifold coordinates;
- an empirical distribution estimated from unlabeled scans;
- resynthesis on labeled recipient geometry with exact hard GT;
- preservation of recipient texture residual and exact identity outside the band.

It is not generic boundary-aware augmentation. Closest families already include
task-driven learned intensity/deformation fields, whole-foreground intensity
perturbation plus harmonization (ARHNet), KeepMask/KeepMix, boundary-region replacement
(BoundaryMix), histogram/Fourier transfer, and boundary-aware losses/contrastive
learning. If the implementation collapses to `mask * (a*x+b)`, a scalar contrast
jitter, or a boundary patch replacement, the novelty claim fails.

## Strongest reviewer objection

“This is a complicated mask-conditioned local contrast jitter, and inaccurate teacher
boundaries make an artificial halo bank.”

The required response is empirical: a frozen-model oracle intervention must show an
ordered, localized response to full donor profiles that is stronger than area- and
severity-matched scalar contrast, blur, and histogram controls; teacher-derived
profiles must agree with GT-derived profiles on held-out labeled cases; rendered fields
must lie inside empirical support and have no taper-edge discontinuity. A sham-contour
control and a simple edge-only probe must also rule out a renderer artifact that reveals
the GT contour. Failure of any of these core observations kills the direction rather
than triggering extra modules.

## Initial scope

The first paper targets coherent binary/single-organ boundaries. Fragmented lesions and
overlapping multi-class boundary junctions are outside the initial claim; they should
not be handled by an unregistered rule merely to add another benchmark.

## Backup direction, not an add-on

If BMER fails, the clean backup is **Acquisition-Grid Phase Augmentation**: model a
volume as a continuous signal, shift the through-plane sampling lattice by a random
sub-voxel phase, and use one slice-profile/PSF operator to resample both image and label
occupancy. It targets apex/base appearance and partial volume, but its collision with
inter-slice augmentation and resolution simulation makes it a likely MICCAI/TMI rather
than CVPR direction unless validated broadly. It must not be combined with BMER to
rescue a weak result.
