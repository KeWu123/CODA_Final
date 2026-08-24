"""H7.16 fixed 1.50x ordered appearance transform for SliceEqOcc-OAAC."""

import torch


OAAC_SCALE = 1.50
LOG_GAMMA_RANGE = (-0.30, 0.30)
LOG_CONTRAST_RANGE = (-0.225, 0.225)
BRIGHTNESS_SPAN_RANGE = (-0.15, 0.15)


def _uniform_per_sample(images, value_range, generator):
    low, high = value_range
    if high < low:
        raise ValueError('appearance range must be ordered')
    shape = (images.shape[0], 1, 1, 1)
    values = torch.rand(
        shape, dtype=images.dtype, device=images.device,
        generator=generator)
    return low + (high - low) * values


@torch.no_grad()
def ordered_appearance_transform(images, generator=None, epsilon=1e-6):
    """Apply one monotonic gamma/contrast/brightness transform per sample.

    The transform changes no coordinates, performs no clipping and has a
    positive intensity derivative for every nonconstant input. Targets are not
    accepted by this function, making accidental target modification impossible.
    """
    if images.ndim != 4:
        raise ValueError('OAAC images must have shape [B,C,H,W]')
    if images.shape[0] < 1 or images.shape[1] < 1 or \
            images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError('OAAC requires a nonempty spatial image batch')
    if not images.is_floating_point():
        raise TypeError('OAAC images must be floating point')
    if not torch.isfinite(images).all():
        raise FloatingPointError('OAAC images contain non-finite values')
    if epsilon <= 0.0:
        raise ValueError('epsilon must be positive')

    flattened = images.flatten(1)
    minimum = flattened.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flattened.max(dim=1).values.view(-1, 1, 1, 1)
    span = maximum - minimum
    valid = span > 0.0
    safe_span = torch.where(valid, span, torch.ones_like(span))
    normalized = (images - minimum) / safe_span

    log_gamma = _uniform_per_sample(
        images, LOG_GAMMA_RANGE, generator)
    gamma = torch.exp(log_gamma)
    gamma_image = minimum + safe_span * normalized.pow(gamma)

    log_contrast = _uniform_per_sample(
        images, LOG_CONTRAST_RANGE, generator)
    contrast = torch.exp(log_contrast)
    transformed_mean = gamma_image.mean(dim=(1, 2, 3), keepdim=True)
    contrasted = transformed_mean + contrast * (
        gamma_image - transformed_mean)

    brightness_fraction = _uniform_per_sample(
        images, BRIGHTNESS_SPAN_RANGE, generator)
    transformed = contrasted + brightness_fraction * safe_span
    transformed = torch.where(valid, transformed, images)

    if not torch.isfinite(transformed).all():
        raise FloatingPointError('OAAC produced non-finite values')

    normalized_change = (
        (transformed - images).abs().flatten(1).mean(dim=1) /
        safe_span.flatten())
    metadata = {
        'appearance_abs_log_gamma_mean': log_gamma.abs().mean(),
        'appearance_abs_log_contrast_mean': log_contrast.abs().mean(),
        'appearance_abs_brightness_fraction_mean': (
            brightness_fraction.abs().mean()),
        'appearance_normalized_absolute_change': normalized_change.mean(),
        'appearance_active_sample_fraction': (
            normalized_change > 1e-7).to(images.dtype).mean(),
        'appearance_below_source_min_fraction': (
            transformed < minimum).to(images.dtype).mean(),
        'appearance_above_source_max_fraction': (
            transformed > maximum).to(images.dtype).mean(),
    }
    return transformed.detach(), metadata
