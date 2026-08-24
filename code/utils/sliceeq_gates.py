"""Mechanism diagnostics for the preregistered SliceEq H7.3 gates."""

import math

import torch
import torch.nn.functional as F

from utils.sliceeq_occ import validate_occupancy


def acquisition_residual(occupancy, center_target):
    """Return detached total-variation distance from the central hard target."""
    validate_occupancy(occupancy)
    if center_target.ndim != 3:
        raise ValueError('center_target must have shape [B,H,W]')
    if occupancy.shape[0] != center_target.shape[0] or \
            occupancy.shape[-2:] != center_target.shape[-2:]:
        raise ValueError('occupancy and center_target dimensions do not match')
    if center_target.min().item() < 0 or \
            center_target.max().item() >= occupancy.shape[1]:
        raise ValueError('center_target is outside the class range')

    center_one_hot = F.one_hot(
        center_target.long(), num_classes=occupancy.shape[1])
    center_one_hot = center_one_hot.permute(0, 3, 1, 2).to(
        device=occupancy.device, dtype=occupancy.dtype)
    return 0.5 * (
        occupancy.detach() - center_one_hot).abs().sum(dim=1)


def normalized_weighted_soft_cross_entropy(logits, occupancy, weights):
    """Per-sample normalized soft CE under a detached spatial measure.

    Samples with zero weight have an exactly zero contribution, as required by
    the H7.3 formulation.
    """
    if logits.shape != occupancy.shape:
        raise ValueError('logits and occupancy must have identical shapes')
    if weights.shape != logits.shape[:1] + logits.shape[-2:]:
        raise ValueError('weights must have shape [B,H,W]')
    validate_occupancy(occupancy)
    if not torch.isfinite(weights).all():
        raise FloatingPointError('weights contain non-finite values')
    if (weights < 0).any():
        raise ValueError('weights must be nonnegative')

    target = occupancy.to(dtype=logits.dtype).detach()
    detached_weights = weights.to(dtype=logits.dtype).detach()
    pixel_loss = -(
        target * F.log_softmax(logits, dim=1)).sum(dim=1)
    numerator = (pixel_loss * detached_weights).flatten(1).sum(dim=1)
    denominator = detached_weights.flatten(1).sum(dim=1)
    active = denominator > 0
    safe_denominator = denominator.clamp_min(
        torch.finfo(logits.dtype).eps)
    per_sample = numerator / safe_denominator
    per_sample = torch.where(active, per_sample, torch.zeros_like(per_sample))
    return per_sample.mean()


def gradient_pixel_norm(gradient):
    """L2 norm over class logits at each pixel."""
    if gradient.ndim != 4:
        raise ValueError('gradient must have shape [B,C,H,W]')
    return gradient.square().sum(dim=1).sqrt()


def per_sample_gradient_cosine(first, second, active_samples=None):
    """Return cosines and unit-vector distances for nonzero sample gradients."""
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError('gradients must have matched [B,C,H,W] shapes')
    first_flat = first.flatten(1)
    second_flat = second.flatten(1)
    first_norm = first_flat.norm(dim=1)
    second_norm = second_flat.norm(dim=1)
    valid = (first_norm > 0) & (second_norm > 0)
    if active_samples is not None:
        if active_samples.shape != valid.shape:
            raise ValueError('active_samples must have shape [B]')
        valid = valid & active_samples.bool()
    if not valid.any():
        return [], []

    cosine = F.cosine_similarity(
        first_flat[valid], second_flat[valid], dim=1).clamp(-1.0, 1.0)
    unit_distance = (2.0 - 2.0 * cosine).clamp_min(0.0).sqrt()
    return cosine.detach().cpu().tolist(), \
        unit_distance.detach().cpu().tolist()


class PearsonAccumulator:
    """Numerically stable-enough sufficient statistics for bounded gate data."""

    def __init__(self):
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, first, second):
        first = first.detach().double().reshape(-1)
        second = second.detach().double().reshape(-1)
        if first.numel() != second.numel():
            raise ValueError('Pearson inputs must have the same number of values')
        if first.numel() == 0:
            return
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise FloatingPointError('Pearson inputs contain non-finite values')
        self.count += int(first.numel())
        self.sum_x += float(first.sum().item())
        self.sum_y += float(second.sum().item())
        self.sum_x2 += float(first.square().sum().item())
        self.sum_y2 += float(second.square().sum().item())
        self.sum_xy += float((first * second).sum().item())

    def correlation(self):
        if self.count < 2:
            return None
        count = float(self.count)
        covariance = self.sum_xy - self.sum_x * self.sum_y / count
        variance_x = self.sum_x2 - self.sum_x * self.sum_x / count
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / count
        denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
        if denominator <= 0.0:
            return None
        return covariance / denominator

