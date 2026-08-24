# H5 BMER Implementation Protocol

Status: **locked before implementation**  
Locked: 2026-08-11  
Git note: the current environment has no `git` executable, so this timestamped file and
the research log provide the available pre-implementation record.

## Scope

Add parallel `train_bmer.py` / `test_bmer.py` entry points and `utils/bmer.py`. Do not
modify `train_baseline.py`, `test_baseline.py`, `train_coda.py`, `test_coda.py`, or the
dataset/sampler implementation.

The BMER training entry reuses the current CoDA entry's fixed deployment defaults and
baseline implementation components. The only method change is the labeled input tensor
during self-training.

## Frozen baseline defaults

- data root: `/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source`;
- model: 2-D binary U-Net;
- patch: `256 x 256`;
- pretrain/self-train: `10,000 / 30,000` iterations;
- batch: `24 = 12 labeled + 12 unlabeled`;
- label count: `7`; seed: `1337`;
- SGD LR `0.01`, momentum `0.9`, weight decay `1e-4`;
- EMA `0.99`; original consistency ramp and 1,000-step unsupervised cutoff;
- original RandomGenerator, TwoStreamBatchSampler, validation interval, checkpoint
  format, LCC hard pseudo-label, CE+Dice, and student inference.

Only the experiment name changes to `BMER_PROMISE12` so checkpoints do not collide.

## Fixed BMER v1 defaults

- ribbon radius: `8` pixels;
- angular/tangential sectors: `16`;
- normalized slice-position bins: `3`;
- per-sample application probability: `0.5`;
- interpolation strength: uniform `[0.5, 1.0]`;
- minimum predicted foreground pixels for a donor: `32`;
- deterministic bank batch size: `24`;
- taper: cosine from the boundary to exact zero at the ribbon edge;
- robust intensity units: median and `1.4826 * MAD`, with standard-deviation fallback;
- output clamp: recipient image minimum/maximum;
- RNG: independent CPU `torch.Generator` seeded by the existing seed.

These parameters are exposed for later ablation but the defaults above define BMER v1.

## Frozen bank lifecycle

1. Run the unchanged supervised pretraining stage.
2. Load the resulting checkpoint into a separate inference-only U-Net copy.
3. Set only that disposable bank model to eval mode; this does not alter the training
   student or EMA teacher state/mode.
4. Deterministically resize every unlabeled slice with the current nearest-neighbor
   convention; never read its hidden H5 label for bank construction.
5. Predict the baseline 2-D LCC mask and extract a standardized two-sided profile field
   partitioned by angular sector and signed-distance bin.
6. Bin donors by normalized within-volume slice position, freeze the bank, and save it
   under the self-training snapshot for provenance.
7. Destroy the disposable model before self-training.

Empty/full/tiny predicted masks are skipped. A bank with zero valid donors is a hard,
descriptive failure rather than an identity fallback.

## Self-training invariant

For each original two-stream batch:

```text
x_l_aug = BMER(x_l, y_l, frozen_U_bank, independent_rng)
x_student = concat(x_l_aug, x_u)
y_l_target = y_l
y_u_target = LCC(argmax(teacher(x_u)))
loss = original supervised CE+Dice + original ramp * original hard-U CE+Dice
```

The EMA teacher still receives original `x_u`. BMER never changes `x_u`, `y_l`, the
pseudo-target, loss, consistency weight, EMA update, optimizer, or inference graph.
BMER begins at self-training iteration zero because it is an exact-GT labeled
augmentation; the baseline's first 1,000-step unsupervised cutoff remains unchanged.

## Renderer definition

For each valid binary mask:

- signed distance is `EDT(foreground) - EDT(background)`;
- tangential coordinate is approximated by polar angle around the mask centroid and
  quantized into 16 sectors;
- a profile cell is the mean standardized low-pass intensity for one sector/distance
  bin; missing cells are linearly interpolated along distance, then fall back to the
  slice-global profile;
- recipient residual is preserved by adding the sampled-donor minus recipient-profile
  lookup, multiplied by the cosine taper and sampled strength;
- pixels outside the ribbon are copied exactly.

This is the 2-D baseline-compatible approximation to the pre-registered boundary
manifold. It does not claim a full 3-D surface parameterization.

## Diagnostics

Log bank size/counts by position bin, application fraction, valid-mask fraction,
strength, changed-pixel fraction, mean absolute change, and original/augmented/delta
images. Save the bank configuration, donor case names, and tensors.

## Acceptance tests

1. Original baseline and CoDA file hashes remain unchanged.
2. Utility rejects invalid shapes/configurations and skips invalid masks.
3. Profile extraction/sampling is finite and deterministic under a fixed generator.
4. Same-image donor reconstruction is identity within numerical tolerance.
5. A different valid donor changes boundary pixels while all outside-band pixels remain
   bitwise identical.
6. Output shape/dtype/device are preserved and output is detached.
7. Position-bin fallback is deterministic and bank save/load is lossless.
8. AST contract confirms BMER defaults and that the student concatenates augmented L
   with original U while teacher input remains original U.
9. All existing CPU tests continue to pass.

Passing these tests establishes implementation fidelity only. It is not H5 mechanism or
segmentation evidence.

