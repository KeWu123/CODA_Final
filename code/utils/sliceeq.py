"""Paired slice-profile re-acquisition operator used by SliceEq."""

import torch
import torch.nn.functional as F


def sample_slice_profiles(batch_size, offsets, sigma_range, phase_range,
                          device, generator=None):
    """Sample normalized Gaussian slice profiles independently per sample."""
    sigma_min, sigma_max = sigma_range
    phase_min, phase_max = phase_range
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError('invalid SliceEq sigma range')
    if phase_max < phase_min:
        raise ValueError('invalid SliceEq phase range')

    sigma = torch.rand(
        batch_size, device=device, generator=generator)
    sigma = sigma * (sigma_max - sigma_min) + sigma_min
    phase = torch.rand(
        batch_size, device=device, generator=generator)
    phase = phase * (phase_max - phase_min) + phase_min
    offset_tensor = torch.as_tensor(
        offsets, dtype=sigma.dtype, device=device).view(1, -1)
    logits = -0.5 * (
        (offset_tensor - phase.view(-1, 1)) /
        sigma.view(-1, 1)).square()
    weights = torch.softmax(logits, dim=1)
    _validate_weights(weights)
    return weights, sigma, phase


def paired_slice_reacquisition(image_stack, hard_target_stack, weights,
                               num_classes):
    """Apply one profile to image signal and one-hot tissue occupancy.

    Args:
        image_stack: `[B, K, C, H, W]` float tensor.
        hard_target_stack: `[B, K, H, W]` integer tensor.
        weights: `[B, K]` normalized nonnegative tensor.
        num_classes: target class count.
    """
    if image_stack.ndim != 5:
        raise ValueError('image_stack must have shape [B,K,C,H,W]')
    if hard_target_stack.ndim != 4:
        raise ValueError('hard_target_stack must have shape [B,K,H,W]')
    if weights.ndim != 2:
        raise ValueError('weights must have shape [B,K]')
    if image_stack.shape[:2] != hard_target_stack.shape[:2] or \
            image_stack.shape[:2] != weights.shape:
        raise ValueError('SliceEq batch/stack dimensions do not match')
    if image_stack.shape[-2:] != hard_target_stack.shape[-2:]:
        raise ValueError('SliceEq spatial dimensions do not match')
    if num_classes < 2:
        raise ValueError('num_classes must be at least two')

    _validate_weights(weights)
    if hard_target_stack.min().item() < 0 or \
            hard_target_stack.max().item() >= num_classes:
        raise ValueError('hard targets are outside the class range')

    image_weights = weights.view(
        weights.shape[0], weights.shape[1], 1, 1, 1)
    reacquired_image = (image_stack * image_weights).sum(dim=1)

    occupancy = F.one_hot(
        hard_target_stack.long(), num_classes=num_classes)
    occupancy = occupancy.permute(0, 1, 4, 2, 3).to(image_stack.dtype)
    target_weights = weights.view(
        weights.shape[0], weights.shape[1], 1, 1, 1)
    reacquired_occupancy = (occupancy * target_weights).sum(dim=1)
    reacquired_target = reacquired_occupancy.argmax(dim=1)

    if not torch.isfinite(reacquired_image).all():
        raise FloatingPointError('SliceEq produced a non-finite image')
    if not torch.isfinite(reacquired_occupancy).all():
        raise FloatingPointError('SliceEq produced non-finite occupancy')
    return reacquired_image, reacquired_target, reacquired_occupancy


def reacquisition_diagnostics(image_stack, target_stack, reacquired_image,
                              reacquired_target, weights, sigma, phase):
    """Return detached observables needed to determine operator activity."""
    center = image_stack.shape[1] // 2
    center_image = image_stack[:, center]
    center_target = target_stack[:, center]
    return {
        'sigma_mean': sigma.detach().mean(),
        'absolute_phase_mean': phase.detach().abs().mean(),
        'center_weight_mean': weights[:, center].detach().mean(),
        'image_absolute_change': (
            reacquired_image.detach() - center_image.detach()).abs().mean(),
        'target_changed_fraction': (
            reacquired_target.detach() != center_target.detach()).float().mean(),
        'center_foreground_fraction': (
            center_target.detach() > 0).float().mean(),
        'reacquired_foreground_fraction': (
            reacquired_target.detach() > 0).float().mean(),
    }


def _validate_weights(weights):
    if not torch.isfinite(weights).all():
        raise FloatingPointError('SliceEq weights contain non-finite values')
    if (weights < 0).any():
        raise ValueError('SliceEq weights must be nonnegative')
    sums = weights.sum(dim=1)
    if not torch.allclose(
            sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError('SliceEq weights must sum to one')

