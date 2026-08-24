"""Fractional-occupancy supervision for the independent SliceEqOcc version."""

import math

import torch
import torch.nn.functional as F


def validate_occupancy(occupancy, atol=1e-6):
    """Validate a dense per-pixel class occupancy distribution."""
    if occupancy.ndim != 4 or occupancy.shape[1] < 2:
        raise ValueError(
            'occupancy must have shape [B,C,H,W] with at least two classes')
    if not torch.isfinite(occupancy).all():
        raise FloatingPointError('occupancy contains non-finite values')
    if (occupancy < -atol).any() or (occupancy > 1.0 + atol).any():
        raise ValueError('occupancy values must lie in [0,1]')
    sums = occupancy.sum(dim=1)
    if not torch.allclose(
            sums, torch.ones_like(sums), atol=atol, rtol=atol):
        raise ValueError('occupancy must sum to one over classes')


def soft_cross_entropy(logits, occupancy):
    """Cross-entropy against acquisition-derived fractional occupancy."""
    if logits.shape != occupancy.shape:
        raise ValueError('logits and occupancy must have identical shapes')
    validate_occupancy(occupancy)
    target = occupancy.to(dtype=logits.dtype).detach()
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def soft_dice_loss(probabilities, occupancy, smooth=1e-10):
    """Squared-denominator Dice extended to fractional targets."""
    if probabilities.shape != occupancy.shape:
        raise ValueError(
            'probabilities and occupancy must have identical shapes')
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError('probabilities contain non-finite values')
    validate_occupancy(occupancy)
    target = occupancy.to(dtype=probabilities.dtype).detach()
    class_losses = []
    for class_index in range(probabilities.shape[1]):
        prediction = probabilities[:, class_index]
        class_target = target[:, class_index]
        intersection = torch.sum(prediction * class_target)
        denominator = torch.sum(prediction.square()) + \
            torch.sum(class_target.square())
        class_losses.append(
            1.0 - (2.0 * intersection + smooth) /
            (denominator + smooth))
    return torch.stack(class_losses).mean()


def soft_segmentation_loss(logits, occupancy):
    """Return the baseline-compatible mean of soft CE and soft Dice."""
    probabilities = torch.softmax(logits, dim=1)
    loss_ce = soft_cross_entropy(logits, occupancy)
    loss_dice = soft_dice_loss(probabilities, occupancy)
    return 0.5 * (loss_ce + loss_dice), loss_ce, loss_dice


def occupancy_diagnostics(occupancy, center_target):
    """Measure whether fractional supervision is active and localized."""
    validate_occupancy(occupancy)
    if center_target.ndim != 3:
        raise ValueError('center_target must have shape [B,H,W]')
    if occupancy.shape[0] != center_target.shape[0] or \
            occupancy.shape[-2:] != center_target.shape[-2:]:
        raise ValueError('occupancy and center_target dimensions do not match')
    if center_target.min().item() < 0 or \
            center_target.max().item() >= occupancy.shape[1]:
        raise ValueError('center_target is outside the class range')

    detached = occupancy.detach()
    center_one_hot = F.one_hot(
        center_target.long(), num_classes=occupancy.shape[1])
    center_one_hot = center_one_hot.permute(0, 3, 1, 2).to(detached.dtype)
    entropy = -(detached.clamp_min(1e-7).log() * detached).sum(dim=1)
    entropy = entropy / math.log(occupancy.shape[1])
    maximum = detached.max(dim=1).values
    hard_target = detached.argmax(dim=1)
    foreground = detached[:, 1:].sum(dim=1)

    return {
        'normalized_occupancy_entropy': entropy.mean(),
        'fractional_pixel_fraction': (
            maximum < 1.0 - 1e-6).float().mean(),
        'occupancy_deviation_from_center': (
            0.5 * (detached - center_one_hot).abs().sum(dim=1)).mean(),
        'hard_target_changed_fraction': (
            hard_target != center_target).float().mean(),
        'foreground_occupancy_mean': foreground.mean(),
        'center_foreground_fraction': (
            center_target > 0).float().mean(),
    }

