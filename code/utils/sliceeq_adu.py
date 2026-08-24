"""Acquisition-aligned dropout reliability for SliceEqOcc-ADU."""

import torch
import torch.nn.functional as F

from utils.sliceeq_occ import validate_occupancy
from utils.sliceeq_reliability import (
    jensen_shannon_map, reliability_from_js, restore_buffers,
    snapshot_buffers)


def isolated_stochastic_teacher_forward(module, inputs, seed):
    """Run an extra CUDA forward without persistent buffer or RNG changes."""
    if not module.training:
        raise RuntimeError('ADU preserves the train-mode teacher policy')
    if inputs.device.type != 'cuda':
        raise ValueError('ADU isolated teacher forward requires CUDA inputs')
    device_index = inputs.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    buffer_snapshot = snapshot_buffers(module)
    try:
        with torch.random.fork_rng(devices=[device_index]):
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(seed)
                with torch.no_grad():
                    output = module(inputs)
    finally:
        restore_buffers(module, buffer_snapshot)
    return output


def acquisition_aligned_reliability(first_occupancy, second_occupancy):
    """Return mean occupancy, operator-space JS, and detached reliability.

    Both occupancies must already have passed through the same SliceEq profile.
    Consequently, agreement on a modeled acquisition-derived fractional
    occupancy has zero JS disagreement and retains unit reliability.
    """
    if first_occupancy.shape != second_occupancy.shape:
        raise ValueError('ADU occupancies must have identical shapes')
    validate_occupancy(first_occupancy)
    validate_occupancy(second_occupancy)
    first = first_occupancy.detach()
    second = second_occupancy.detach()
    mean_occupancy = 0.5 * (first + second)
    validate_occupancy(mean_occupancy)
    divergence = jensen_shannon_map(first, second)
    reliability = reliability_from_js(
        divergence, num_classes=first.shape[1])
    return (
        mean_occupancy.detach(), divergence.detach(), reliability.detach())


def _validate_reliability(reliability, occupancy):
    if reliability.ndim != 3 or reliability.shape != (
            occupancy.shape[0], occupancy.shape[2], occupancy.shape[3]):
        raise ValueError('reliability must have shape [B,H,W]')
    if not torch.isfinite(reliability).all():
        raise FloatingPointError('reliability contains non-finite values')
    if (reliability < 0.0).any() or (reliability > 1.0).any():
        raise ValueError('reliability must lie in [0,1]')


def reliability_weighted_soft_cross_entropy(
        logits, occupancy, reliability):
    """Soft CE normalized by the detached operator-space reliability mass."""
    if logits.shape != occupancy.shape:
        raise ValueError('logits and occupancy must have identical shapes')
    validate_occupancy(occupancy)
    _validate_reliability(reliability, occupancy)
    target = occupancy.to(dtype=logits.dtype).detach()
    weight = reliability.to(
        device=logits.device, dtype=logits.dtype).detach()
    denominator = weight.sum()
    if denominator.item() <= 0.0:
        return logits.sum() * 0.0
    pixel_loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    return (pixel_loss * weight).sum() / denominator


def reliability_weighted_soft_dice_loss(
        probabilities, occupancy, reliability, smooth=1e-10):
    """Squared-denominator soft Dice under detached reliability weights."""
    if probabilities.shape != occupancy.shape:
        raise ValueError(
            'probabilities and occupancy must have identical shapes')
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError('probabilities contain non-finite values')
    validate_occupancy(occupancy)
    _validate_reliability(reliability, occupancy)
    target = occupancy.to(dtype=probabilities.dtype).detach()
    weight = reliability.to(
        device=probabilities.device, dtype=probabilities.dtype).detach()
    if weight.sum().item() <= 0.0:
        return probabilities.sum() * 0.0

    class_losses = []
    for class_index in range(probabilities.shape[1]):
        prediction = probabilities[:, class_index]
        class_target = target[:, class_index]
        intersection = torch.sum(weight * prediction * class_target)
        denominator = torch.sum(weight * prediction.square()) + \
            torch.sum(weight * class_target.square())
        class_losses.append(
            1.0 - (2.0 * intersection + smooth) /
            (denominator + smooth))
    return torch.stack(class_losses).mean()


def reliability_weighted_soft_segmentation_loss(
        logits, occupancy, reliability):
    """Return the baseline-compatible average of ADU-weighted CE and Dice."""
    probabilities = torch.softmax(logits, dim=1)
    loss_ce = reliability_weighted_soft_cross_entropy(
        logits, occupancy, reliability)
    loss_dice = reliability_weighted_soft_dice_loss(
        probabilities, occupancy, reliability)
    return 0.5 * (loss_ce + loss_dice), loss_ce, loss_dice


def adu_diagnostics(first_occupancy, second_occupancy, mean_occupancy,
                    divergence, reliability, epsilon=1e-7):
    """Detached activity checks for the parameter-free ADU intervention."""
    validate_occupancy(first_occupancy)
    validate_occupancy(second_occupancy)
    validate_occupancy(mean_occupancy)
    _validate_reliability(reliability, mean_occupancy)
    if divergence.shape != reliability.shape or \
            not torch.isfinite(divergence).all():
        raise ValueError('ADU divergence must be finite with shape [B,H,W]')

    reliability = reliability.detach()
    divergence = divergence.detach()
    first = first_occupancy.detach()
    second = second_occupancy.detach()
    mean = mean_occupancy.detach()
    weight_sum = reliability.sum()
    weight_square_sum = reliability.square().sum()
    effective_fraction = weight_sum.square() / (
        reliability.numel() * weight_square_sum).clamp_min(epsilon)
    fractional = mean.max(dim=1).values < 1.0 - 1e-6
    if fractional.any():
        fractional_weight = reliability[fractional].mean()
    else:
        fractional_weight = reliability.new_tensor(1.0)

    return {
        'js_mean': divergence.mean(),
        'js_max': divergence.max(),
        'positive_js_fraction': (divergence > epsilon).float().mean(),
        'reliability_mean': reliability.mean(),
        'effective_sample_fraction': effective_fraction,
        'pseudo_fractional_support_mean_weight': fractional_weight,
        'occupancy_absolute_disagreement': (first - second).abs().mean(),
        'occupancy_hard_disagreement': (
            first.argmax(dim=1) != second.argmax(dim=1)).float().mean(),
        'mean_target_absolute_change_from_primary':
            (mean - first).abs().mean(),
    }
