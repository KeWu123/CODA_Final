# H7.6 Protocol-Conditioned, Scan-Coherent SliceEqOcc

Status: physical/metadata-conditioned stage remains planned. The user authorized a synthetic scan-coherent H7.6a optimization before causal controls; its separately locked implementation protocol is `h7_6_scan_coherent_implementation_protocol.md`.  
Date: 2026-08-13.

## Hypothesis

The current fixed slice-unit, per-sample IID profile distribution is an acquisition-inspired heuristic. Conditioning the operator on trustworthy acquisition geometry in physical coordinates and sharing a virtual protocol coherently across each scan will improve cross-spacing/cross-center robustness over SliceEqOcc while preserving its paired fractional-occupancy mechanism and zero inference overhead.

## Motivation

- A fixed `sigma` in slice units represents different physical support when slice spacing changes.
- Slice spacing is not necessarily slice thickness, and neither is the true RF slice-selection profile; claims must follow available metadata.
- A scanner protocol is volume-level. Independently sampling a different profile for every adjacent center slice creates a set of virtual observations that does not correspond to one coherent acquisition grid.
- SliceEqSAQ's negative result closes batch-marginal quadrature as the next direction; it does not test scan-level coherence or physical conditioning.

## Preconditions

1. Full SliceEqOcc must beat B0-36, ImgOnly, and SliceHard under the locked causal protocol.
2. PROMISE12 source headers or sidecars must be audited for spacing, slice thickness, gap, orientation, center/vendor, and preprocessing provenance.
3. If only spacing is available, the method and paper must use `spacing-conditioned synthetic acquisition`, not `thickness-calibrated PSF`.
4. No PROMISE12 test statistic may choose metadata bins or sampling ranges.

User-order amendment on 2026-08-13: the causal controls remain required for a publication claim, but they no longer block one exploratory H7.6a optimization run. Because the current H5/list contract has no verified physical metadata, H7.6a implements only V1 and retains the original slice-unit profile distribution. V2/V3 remain prohibited until metadata provenance exists.

## Candidate variants

### V0: Current SliceEqOcc

Independent per-sample `sigma` and `phase` in slice units.

### V1: Scan-coherent marginally matched SliceEqOcc

Use the current marginal distribution, but draw one profile from a deterministic key

`(global_seed, epoch_or_refresh_id, case_id)`.

Every slice from a case shares that profile within the refresh window. This isolates profile correlation structure without changing marginal severity.

### V2: Spacing-conditioned physical-coordinate SliceEqOcc

Let `d_i` be trustworthy through-plane voxel spacing for case `i`. Compute tap positions in millimeters and define a predeclared target support distribution in physical coordinates. Do not interpret `d_i` as slice thickness.

### V3: Native-to-target profile composition

Only if native thickness/profile estimates are trustworthy, sample a target profile no sharper than the native profile. For a Gaussian approximation:

\[
\sigma_{extra}=\sqrt{\max(\sigma_{target}^2-\sigma_{native}^2,0)}.
\]

Use the induced discrete integration weights on the available neighboring slices. Report truncation/clamping rates and reject configurations whose intended physical support is not represented by the stack radius.

## Required controls

- V0 current IID SliceEqOcc;
- V1 scan-coherent with identical marginal distribution;
- per-case profile assignment with shuffled case IDs;
- V2/V3 with metadata shuffled across cases;
- image-only version with the identical profile;
- fixed-severity profile matched to the mean center weight;
- if compute permits, a larger-radius control shared by all relevant variants.

## Implementation contract

- The same profile weights act on image intensity and exact/pseudo occupancy.
- No confidence, uncertainty, boundary, attention, or extra pseudo-label filtering is added.
- The center hard labeled anchor, loss family, teacher, optimizer, schedule, and inference graph remain fixed.
- Case IDs and metadata are read from training data only; no test-derived range selection.
- Profile schedule is deterministic and logged per case/refresh.
- Unlabeled labels are not loaded.
- Endpoint clamping is logged; a physically conditioned claim is not allowed if most support is truncated.

## Evaluation

Primary endpoint:

- robustness AUC across predeclared synthetic through-plane degradation levels on an untouched validation/external set.

Secondary endpoints:

- ordinary patient-level Dice/NSD;
- method × spacing/thickness interaction;
- leave-center/protocol-out performance where provenance permits;
- apex/mid/base performance;
- axial prediction continuity;
- calibration between profile severity and occupancy/prediction change.

## Pass rule

H7.6 passes only if a physically/coherently conditioned variant improves robustness AUC over V0 across at least three seeds and one external domain, with a positive paired 95% interval, while matching or improving ordinary Dice. Metadata shuffling must remove or materially reduce the effect.

If only ordinary in-distribution Dice improves, or metadata shuffle performs equally well, treat the change as hyperparameterization rather than a protocol-conditioned contribution.

## Publication claim if successful

Safe claim:

> SliceEq can condition acquisition-equivalent supervision on scan geometry and maintain a coherent virtual acquisition across a volume, improving robustness to through-plane protocol shift without changing the inference model.

Unsafe claim:

> The exact scanner PSF is reconstructed from NIfTI spacing.
