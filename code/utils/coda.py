"""CoDA-MT utilities for corruption-coupled dense pseudo-targets.

This module is deliberately independent of the PROMISE12 data loader.  It only
operates on the unlabeled tensor that the original Baseline already produced,
so case lists, sampling order, and labeled/unlabeled membership stay unchanged.

The target coupling follows the label-smoothing form used by supervised
mollification, but extends it to a spatial field estimated from the realized
degradation and to EMA pseudo-targets for dense prediction.
"""

import torch
import torch.nn.functional as F


SUPPORTED_AUGMENTATIONS = ("resolution", "gaussian_noise")


def _check_image_batch(images):
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("spatial dimensions must both be at least 2")
    if not images.is_floating_point():
        raise TypeError("images must be floating-point tensors")


def _check_range(name, low, high, lower_bound=0.0, upper_bound=None):
    if low < lower_bound or high < low:
        raise ValueError("invalid {} range [{}, {}]".format(name, low, high))
    if upper_bound is not None and high > upper_bound:
        raise ValueError("{} must not exceed {}".format(name, upper_bound))


def _sample_scalar(images, low, high, generator=None):
    if low == high:
        return torch.as_tensor(low, dtype=images.dtype, device=images.device)
    value = torch.rand((), dtype=images.dtype, device=images.device,
                       generator=generator)
    return low + (high - low) * value


def _local_average(values, kernel_size):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return F.avg_pool2d(values, kernel_size=kernel_size, stride=1,
                        padding=kernel_size // 2,
                        count_include_pad=False)


def _sobel_energy(images):
    """Return channel-averaged squared Sobel-gradient energy."""
    _check_image_batch(images)
    sobel = images.new_tensor([
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    ]) / 8.0
    channels = images.shape[1]
    weight = sobel[:, None].repeat(channels, 1, 1, 1)
    padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
    gradients = F.conv2d(padded, weight, groups=channels)
    gradients = gradients.reshape(images.shape[0], channels, 2,
                                  images.shape[-2], images.shape[-1])
    return gradients.square().sum(dim=2).mean(dim=1, keepdim=True)


@torch.no_grad()
def resolution_degradation(images, scale_min=0.25, scale_max=0.75,
                           evidence_kernel=7, generator=None, eps=1e-6):
    """Downsample-upsample a batch and measure local gradient-energy loss."""
    _check_image_batch(images)
    _check_range("resolution scale", scale_min, scale_max,
                 lower_bound=0.0, upper_bound=1.0)
    if scale_min == 0.0:
        raise ValueError("resolution scale must be greater than zero")

    scale = _sample_scalar(images, scale_min, scale_max, generator)
    height, width = images.shape[-2:]
    low_size = (max(1, int(round(height * float(scale)))),
                max(1, int(round(width * float(scale)))))
    low_resolution = F.interpolate(images, size=low_size, mode="bilinear",
                                   align_corners=False)
    strong_images = F.interpolate(low_resolution, size=(height, width),
                                  mode="bilinear", align_corners=False)

    weak_energy = _local_average(_sobel_energy(images), evidence_kernel)
    strong_energy = _local_average(_sobel_energy(strong_images), evidence_kernel)
    evidence_loss = (weak_energy - strong_energy).clamp_min(0.0)
    gamma = (evidence_loss / (weak_energy + eps)).clamp_(0.0, 1.0)

    metadata = {
        "family": "resolution",
        "family_id": images.new_tensor(0.0),
        "severity": (1.0 - scale).detach(),
        "scale": scale.detach(),
    }
    return strong_images.detach(), gamma.detach(), metadata


@torch.no_grad()
def gaussian_noise_degradation(images, std_min=0.05, std_max=0.20,
                               evidence_kernel=7, generator=None, eps=1e-6):
    """Add slice-standard-deviation-scaled noise and measure realized loss.

    Gamma is the local noise-to-signal ratio multiplied by local evidence
    presence.  The latter keeps homogeneous zero-information background from
    being softened merely because its raw signal-to-noise ratio is undefined.
    """
    _check_image_batch(images)
    _check_range("noise standard deviation", std_min, std_max,
                 lower_bound=0.0)

    relative_std = _sample_scalar(images, std_min, std_max, generator)
    slice_std = images.flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
    safe_slice_std = slice_std.clamp_min(eps)
    noise = torch.randn(images.shape, dtype=images.dtype,
                        device=images.device, generator=generator)
    noise = noise * (relative_std * slice_std)
    strong_images = images + noise

    local_mean = _local_average(images, evidence_kernel)
    local_second_moment = _local_average(images.square(), evidence_kernel)
    local_variance = (local_second_moment - local_mean.square()).clamp_min(0.0)
    signal_rms = torch.sqrt(local_variance)
    noise_rms = torch.sqrt(_local_average(noise.square(), evidence_kernel))
    local_noise_ratio = (noise_rms / (signal_rms + eps)).clamp_(0.0, 1.0)
    evidence_presence = (signal_rms / safe_slice_std).clamp_(0.0, 1.0)
    gamma = (local_noise_ratio * evidence_presence).clamp_(0.0, 1.0)

    metadata = {
        "family": "gaussian_noise",
        "family_id": images.new_tensor(1.0),
        "severity": relative_std.detach(),
        "relative_std": relative_std.detach(),
    }
    return strong_images.detach(), gamma.detach(), metadata


@torch.no_grad()
def evidence_augment(images, augmentations=SUPPORTED_AUGMENTATIONS,
                     resolution_scale=(0.25, 0.75), noise_std=(0.05, 0.20),
                     evidence_kernel=7, generator=None):
    """Sample one coordinate-preserving degradation for the unlabeled batch."""
    _check_image_batch(images)
    if isinstance(augmentations, str):
        augmentations = tuple(item.strip() for item in augmentations.split(",")
                              if item.strip())
    else:
        augmentations = tuple(augmentations)
    if not augmentations:
        raise ValueError("at least one CoDA augmentation is required")
    unknown = set(augmentations) - set(SUPPORTED_AUGMENTATIONS)
    if unknown:
        raise ValueError("unsupported CoDA augmentations: {}".format(
            ", ".join(sorted(unknown))))

    family_index = int(torch.randint(len(augmentations), (),
                                     device=images.device,
                                     generator=generator).item())
    family = augmentations[family_index]
    if family == "resolution":
        strong, gamma, metadata = resolution_degradation(
            images, resolution_scale[0], resolution_scale[1],
            evidence_kernel, generator)
    else:
        strong, gamma, metadata = gaussian_noise_degradation(
            images, noise_std[0], noise_std[1], evidence_kernel, generator)
    metadata["sampled_family_id"] = images.new_tensor(float(family_index))
    return strong, gamma, metadata


@torch.no_grad()
def lcc_preserve_teacher_probabilities(teacher_logits, lcc_labels):
    """Apply the Baseline's binary LCC prior without hardening its interior."""
    if teacher_logits.ndim != 4 or teacher_logits.shape[1] != 2:
        raise ValueError("CoDA version 1 expects binary [B, 2, H, W] logits")
    expected_shape = (teacher_logits.shape[0], teacher_logits.shape[2],
                      teacher_logits.shape[3])
    if tuple(lcc_labels.shape) != expected_shape:
        raise ValueError("lcc_labels must have shape [B, H, W]")

    probabilities = torch.softmax(teacher_logits, dim=1)
    foreground_mask = (lcc_labels == 1).unsqueeze(1).to(probabilities.dtype)
    foreground = probabilities[:, 1:2] * foreground_mask
    background = 1.0 - foreground
    return torch.cat((background, foreground), dim=1).detach()


@torch.no_grad()
def couple_pseudo_target(pseudo_target, gamma):
    """Relax each dense pseudo-target toward uniform according to gamma."""
    if pseudo_target.ndim != 4:
        raise ValueError("pseudo_target must have shape [B, C, H, W]")
    if gamma.ndim != 4 or gamma.shape[1] != 1:
        raise ValueError("gamma must have shape [B, 1, H, W]")
    if (pseudo_target.shape[0], pseudo_target.shape[2], pseudo_target.shape[3]) != \
            (gamma.shape[0], gamma.shape[2], gamma.shape[3]):
        raise ValueError("pseudo_target and gamma spatial shapes must match")

    gamma = gamma.to(dtype=pseudo_target.dtype).clamp(0.0, 1.0)
    uniform = torch.full_like(pseudo_target, 1.0 / pseudo_target.shape[1])
    coupled = (1.0 - gamma) * pseudo_target + gamma * uniform
    coupled = coupled / coupled.sum(dim=1, keepdim=True).clamp_min(1e-7)
    return coupled.detach()


def soft_cross_entropy(logits, soft_target):
    """Cross-entropy for a per-pixel target distribution."""
    if logits.shape != soft_target.shape:
        raise ValueError("logits and soft_target must have identical shapes")
    return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def soft_dice_loss(probabilities, soft_target, smooth=1e-10):
    """Soft-target counterpart of the Baseline's squared-denominator Dice."""
    if probabilities.shape != soft_target.shape:
        raise ValueError("probabilities and soft_target must have identical shapes")
    class_losses = []
    for class_index in range(probabilities.shape[1]):
        prediction = probabilities[:, class_index]
        target = soft_target[:, class_index]
        intersection = torch.sum(prediction * target)
        denominator = torch.sum(prediction.square()) + torch.sum(target.square())
        class_losses.append(1.0 - (2.0 * intersection + smooth) /
                            (denominator + smooth))
    return torch.stack(class_losses).mean()
