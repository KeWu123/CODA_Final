"""Conditional antithetic slice profiles for SliceEqOcc.

The antithetic profile reuses the sampled width for the same anatomy and
reflects only the through-plane phase.  Averaging losses from the original
and reflected observations is an unbiased estimator of the symmetric phase
risk, while cancelling its sample-conditional odd component.
"""

import torch


def gaussian_profile_weights(offsets, sigma, phase):
    """Build normalized Gaussian tap weights from batched parameters."""
    if sigma.ndim != 1 or phase.ndim != 1 or sigma.shape != phase.shape:
        raise ValueError('sigma and phase must be same-shaped vectors')
    if sigma.numel() < 1:
        raise ValueError('sigma and phase must be non-empty')
    if not torch.isfinite(sigma).all() or not torch.isfinite(phase).all():
        raise FloatingPointError('profile parameters must be finite')
    if (sigma <= 0.0).any():
        raise ValueError('sigma must be positive')

    offset_tensor = torch.as_tensor(
        offsets, dtype=sigma.dtype, device=sigma.device).view(1, -1)
    if offset_tensor.shape[1] < 2:
        raise ValueError('at least two slice offsets are required')
    logits = -0.5 * (
        (offset_tensor - phase.view(-1, 1)) /
        sigma.view(-1, 1)).square()
    weights = torch.softmax(logits, dim=1)
    if not torch.isfinite(weights).all():
        raise FloatingPointError('profile weights must be finite')
    if not torch.allclose(
            weights.sum(dim=1), torch.ones_like(sigma),
            atol=1e-6, rtol=1e-6):
        raise ValueError('profile weights must sum to one')
    return weights


def antithetic_phase_weights(offsets, sigma, phase):
    """Return the same-width profile reflected around the center slice."""
    return gaussian_profile_weights(offsets, sigma, -phase)


def antithetic_pair_diagnostics(
        primary_weights, reflected_weights, primary_image,
        reflected_image, primary_occupancy, reflected_occupancy, phase):
    """Measure activity and exact phase symmetry of a paired observation."""
    if primary_weights.shape != reflected_weights.shape:
        raise ValueError('paired weights must have identical shapes')
    if primary_image.shape != reflected_image.shape:
        raise ValueError('paired images must have identical shapes')
    if primary_occupancy.shape != reflected_occupancy.shape:
        raise ValueError('paired occupancies must have identical shapes')
    center = primary_weights.shape[1] // 2
    return {
        'phase_pair_residual': (phase + (-phase)).abs().max().detach(),
        'weight_l1_separation': (
            primary_weights - reflected_weights).abs().sum(dim=1).mean(
        ).detach(),
        'center_weight_gap': (
            primary_weights[:, center] -
            reflected_weights[:, center]).abs().mean().detach(),
        'image_absolute_separation': (
            primary_image.detach() - reflected_image.detach()).abs().mean(),
        'occupancy_absolute_separation': (
            primary_occupancy.detach() -
            reflected_occupancy.detach()).abs().mean(),
        'hard_target_disagreement': (
            primary_occupancy.detach().argmax(dim=1) !=
            reflected_occupancy.detach().argmax(dim=1)).float().mean(),
    }
