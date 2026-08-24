"""Orbit-Balanced Augmentation utilities.

OBA constructs two views from opposite coordinates of the same sampled
augmentation.  The functions in this module only transform image tensors; they do not
inspect labels, teacher predictions, or model state.
"""

import torch
import torch.nn.functional as F


SUPPORTED_AUGMENTATIONS = ("log_gamma", "smooth_bias", "gaussian_noise")


def _check_image_batch(images):
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    if images.shape[0] < 1:
        raise ValueError("images must contain at least one sample")
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("spatial dimensions must both be at least 2")
    if not images.is_floating_point():
        raise TypeError("images must be floating-point tensors")
    if not torch.isfinite(images).all():
        raise ValueError("images must be finite")


def _check_range(name, low, high, lower_bound=0.0):
    if low < lower_bound or high < low:
        raise ValueError("invalid {} range [{}, {}]".format(name, low, high))


def _sample_magnitudes(images, low, high, generator=None):
    shape = (images.shape[0], 1, 1, 1)
    if low == high:
        return torch.full(shape, low, dtype=images.dtype, device=images.device)
    values = torch.rand(shape, dtype=images.dtype, device=images.device,
                        generator=generator)
    return low + (high - low) * values


def _normalize_per_sample(images, eps):
    flat = images.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    span = maximum - minimum
    valid = span > eps
    normalized = (images - minimum) / span.clamp_min(eps)
    normalized = normalized.clamp(0.0, 1.0)
    return normalized, minimum, span, valid


def _restore_per_sample(normalized, minimum, span, valid, original):
    restored = minimum + normalized * span
    return torch.where(valid, restored, original)


@torch.no_grad()
def log_gamma_pair(images, magnitude_min=0.10, magnitude_max=0.40,
                   generator=None, eps=1e-6):
    """Apply reciprocal gamma exponents around the identity in log-gamma space."""
    _check_image_batch(images)
    _check_range("log-gamma magnitude", magnitude_min, magnitude_max)
    normalized, minimum, span, valid = _normalize_per_sample(images, eps)
    magnitude = _sample_magnitudes(
        images, magnitude_min, magnitude_max, generator)
    exponent_plus = torch.exp(magnitude)
    exponent_minus = torch.exp(-magnitude)
    plus_normalized = normalized.pow(exponent_plus)
    minus_normalized = normalized.pow(exponent_minus)
    plus = _restore_per_sample(
        plus_normalized, minimum, span, valid, images)
    minus = _restore_per_sample(
        minus_normalized, minimum, span, valid, images)
    metadata = {
        "family": "log_gamma",
        "severity": magnitude.flatten(),
        "plus_exponent": exponent_plus.flatten(),
        "minus_exponent": exponent_minus.flatten(),
    }
    return plus.detach(), minus.detach(), metadata


def _smooth_unit_fields(images, grid_size, generator):
    if grid_size < 2:
        raise ValueError("bias grid size must be at least 2")
    fields = torch.randn(
        (images.shape[0], 1, grid_size, grid_size),
        dtype=images.dtype, device=images.device, generator=generator)
    fields = F.interpolate(fields, size=images.shape[-2:], mode="bicubic",
                           align_corners=False)
    fields = fields - fields.mean(dim=(2, 3), keepdim=True)
    rms = fields.square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1e-6)
    fields = (fields / rms).clamp(-3.0, 3.0) / 3.0
    return fields


@torch.no_grad()
def smooth_bias_pair(images, magnitude_min=0.10, magnitude_max=0.35,
                     grid_size=8, generator=None, eps=1e-5):
    """Apply opposite smooth offsets in normalized-intensity logit space."""
    _check_image_batch(images)
    _check_range("smooth-bias magnitude", magnitude_min, magnitude_max)
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")

    normalized, minimum, span, valid = _normalize_per_sample(images, eps)
    magnitude = _sample_magnitudes(
        images, magnitude_min, magnitude_max, generator)
    unit_field = _smooth_unit_fields(images, grid_size, generator)
    coordinate = magnitude * unit_field

    interior = (normalized > eps) & (normalized < 1.0 - eps)
    base_logit = torch.logit(normalized.clamp(eps, 1.0 - eps))
    plus_normalized = torch.sigmoid(base_logit + coordinate)
    minus_normalized = torch.sigmoid(base_logit - coordinate)
    plus_normalized = torch.where(interior, plus_normalized, normalized)
    minus_normalized = torch.where(interior, minus_normalized, normalized)
    plus = _restore_per_sample(
        plus_normalized, minimum, span, valid, images)
    minus = _restore_per_sample(
        minus_normalized, minimum, span, valid, images)
    metadata = {
        "family": "smooth_bias",
        "severity": magnitude.flatten(),
        "coordinate_rms": coordinate.flatten(1).square().mean(
            dim=1).sqrt(),
    }
    return plus.detach(), minus.detach(), metadata


@torch.no_grad()
def gaussian_noise_pair(images, magnitude_min=0.05, magnitude_max=0.15,
                        generator=None):
    """Add exactly opposite noise scaled by each slice standard deviation."""
    _check_image_batch(images)
    _check_range("Gaussian-noise magnitude", magnitude_min, magnitude_max)
    magnitude = _sample_magnitudes(
        images, magnitude_min, magnitude_max, generator)
    slice_std = images.flatten(1).std(
        dim=1, unbiased=False).view(-1, 1, 1, 1)
    direction = torch.randn(
        images.shape, dtype=images.dtype, device=images.device,
        generator=generator)
    noise = direction * magnitude * slice_std
    plus = images + noise
    minus = images - noise
    metadata = {
        "family": "gaussian_noise",
        "severity": magnitude.flatten(),
        "realized_relative_rms": (
            noise.flatten(1).square().mean(dim=1).sqrt() /
            slice_std.flatten().clamp_min(1e-6)),
    }
    return plus.detach(), minus.detach(), metadata


def _parse_augmentations(augmentations):
    if isinstance(augmentations, str):
        parsed = tuple(item.strip() for item in augmentations.split(",")
                       if item.strip())
    else:
        parsed = tuple(augmentations)
    if not parsed:
        raise ValueError("at least one OBA augmentation is required")
    if len(set(parsed)) != len(parsed):
        raise ValueError("OBA augmentation names must be unique")
    unknown = set(parsed) - set(SUPPORTED_AUGMENTATIONS)
    if unknown:
        raise ValueError("unsupported OBA augmentations: {}".format(
            ", ".join(sorted(unknown))))
    return parsed


def _pair_diagnostics(images, plus, minus):
    plus_delta = (plus - images).flatten(1)
    minus_delta = (minus - images).flatten(1)
    numerator = (plus_delta * minus_delta).sum(dim=1)
    denominator = plus_delta.norm(dim=1) * minus_delta.norm(dim=1)
    cosine = torch.where(
        denominator > 1e-12, numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(denominator))
    return {
        "plus_mean_absolute_change": plus_delta.abs().mean(),
        "minus_mean_absolute_change": minus_delta.abs().mean(),
        "displacement_cosine": cosine.mean(),
        "midpoint_drift": ((plus + minus) * 0.5 - images).abs().mean(),
        "pair_span": (plus - minus).abs().mean(),
    }


@torch.no_grad()
def orbit_balanced_augment(
        images, augmentations=SUPPORTED_AUGMENTATIONS,
        gamma_magnitude=(0.10, 0.40), bias_magnitude=(0.10, 0.35),
        noise_magnitude=(0.05, 0.15), bias_grid_size=8, generator=None):
    """Generate per-sample antithetic views from a shared nuisance coordinate."""
    _check_image_batch(images)
    augmentations = _parse_augmentations(augmentations)
    _check_range("log-gamma magnitude", *gamma_magnitude)
    _check_range("smooth-bias magnitude", *bias_magnitude)
    _check_range("Gaussian-noise magnitude", *noise_magnitude)
    if bias_grid_size < 2:
        raise ValueError("bias grid size must be at least 2")

    family_ids = torch.randint(
        len(augmentations), (images.shape[0],), device=images.device,
        generator=generator)
    plus = images.clone()
    minus = images.clone()
    severity = images.new_zeros(images.shape[0])

    for family_index, family in enumerate(augmentations):
        selected = family_ids == family_index
        if not bool(selected.any()):
            continue
        selected_images = images[selected]
        if family == "log_gamma":
            selected_plus, selected_minus, selected_metadata = log_gamma_pair(
                selected_images, gamma_magnitude[0], gamma_magnitude[1],
                generator)
        elif family == "smooth_bias":
            selected_plus, selected_minus, selected_metadata = smooth_bias_pair(
                selected_images, bias_magnitude[0], bias_magnitude[1],
                bias_grid_size, generator)
        else:
            selected_plus, selected_minus, selected_metadata = gaussian_noise_pair(
                selected_images, noise_magnitude[0], noise_magnitude[1],
                generator)
        plus[selected] = selected_plus
        minus[selected] = selected_minus
        severity[selected] = selected_metadata["severity"]

    metadata = {
        "family_ids": family_ids.detach(),
        "severity": severity.detach(),
        "augmentation_count": images.new_tensor(float(len(augmentations))),
    }
    for family_index, family in enumerate(augmentations):
        metadata["family_fraction_{}".format(family)] = (
            family_ids == family_index).to(images.dtype).mean()
    metadata.update(_pair_diagnostics(images, plus, minus))
    return plus.detach(), minus.detach(), metadata
