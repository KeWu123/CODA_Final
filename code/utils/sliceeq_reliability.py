"""Reliability utilities for the SliceEq H7.10 zero-training gate."""

from contextlib import contextmanager
import math
import types

import numpy as np
import torch
import torch.nn as nn

from utils.sliceeq_occ import validate_occupancy


INDEPENDENT_DROPOUT = 'independent'
STACK_SHARED_DROPOUT = 'stack_shared'
SUPPORTED_DROPOUT_MODES = (
    INDEPENDENT_DROPOUT, STACK_SHARED_DROPOUT)


def structured_stack_dropout(inputs, probability, stack_size, mode,
                             training=True, inplace=False):
    """Apply paired independent or stack-shared elementwise dropout.

    Both modes draw a full tensor of uniforms. The shared mode uses the center
    slice's mask for every slice in a stack but still consumes the same number
    of random values as the independent mode. Resetting the RNG before each
    mode therefore aligns all later dropout draws in the network.
    """
    if mode not in SUPPORTED_DROPOUT_MODES:
        raise ValueError('unsupported dropout mode: {}'.format(mode))
    if not isinstance(stack_size, int) or stack_size < 1:
        raise ValueError('stack_size must be a positive integer')
    if probability < 0.0 or probability >= 1.0:
        raise ValueError('dropout probability must lie in [0,1)')
    if not training or probability == 0.0:
        return inputs
    if inputs.ndim < 1 or inputs.shape[0] % stack_size != 0:
        raise ValueError(
            'dropout batch must contain complete contiguous stacks')

    uniforms = torch.rand_like(inputs)
    keep = uniforms >= probability
    if mode == STACK_SHARED_DROPOUT:
        grouped = keep.reshape(
            inputs.shape[0] // stack_size, stack_size, *inputs.shape[1:])
        center = stack_size // 2
        grouped = grouped[:, center:center + 1].expand_as(grouped)
        keep = grouped.reshape_as(inputs)
    output = inputs * keep.to(dtype=inputs.dtype) / (1.0 - probability)
    if inplace:
        return inputs.copy_(output)
    return output


@contextmanager
def temporary_stack_dropout(model, mode, stack_size):
    """Temporarily route every nn.Dropout through the paired operator."""
    if mode not in SUPPORTED_DROPOUT_MODES:
        raise ValueError('unsupported dropout mode: {}'.format(mode))
    patched = []
    for module in model.modules():
        if not isinstance(module, nn.Dropout):
            continue
        had_instance_forward = 'forward' in module.__dict__
        original_instance_forward = module.__dict__.get('forward')

        def replacement(dropout_module, inputs):
            return structured_stack_dropout(
                inputs,
                probability=dropout_module.p,
                stack_size=stack_size,
                mode=mode,
                training=dropout_module.training,
                inplace=dropout_module.inplace)

        module.forward = types.MethodType(replacement, module)
        patched.append(
            (module, had_instance_forward, original_instance_forward))
    try:
        yield len(patched)
    finally:
        for module, had_instance_forward, original in patched:
            if had_instance_forward:
                module.forward = original
            else:
                del module.__dict__['forward']


def snapshot_buffers(model):
    """Clone all model buffers so train-mode BN analysis is reversible."""
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
    }


def restore_buffers(model, snapshot):
    """Restore an exact buffer snapshot without changing parameters."""
    current = dict(model.named_buffers())
    if set(current) != set(snapshot):
        raise ValueError('model buffers differ from the recorded snapshot')
    with torch.no_grad():
        for name, value in snapshot.items():
            current[name].copy_(value)


def jensen_shannon_map(first, second):
    """Per-pixel Jensen--Shannon divergence between two occupancies."""
    if first.shape != second.shape:
        raise ValueError('occupancy shapes must match')
    validate_occupancy(first)
    validate_occupancy(second)
    # Entropy differences can be much smaller than the individual entropies.
    # Compute the diagnostic in float64 even when training tensors are float32.
    first = first.detach().to(dtype=torch.float64)
    second = second.detach().to(dtype=torch.float64)
    mean = 0.5 * (first + second)

    def entropy(distribution):
        safe_log = distribution.clamp_min(
            torch.finfo(distribution.dtype).tiny).log()
        return -(distribution * safe_log).sum(dim=1)

    divergence = entropy(mean) - 0.5 * (
        entropy(first) + entropy(second))
    return divergence.clamp_min(0.0)


def reliability_from_js(divergence, num_classes):
    """Map JS divergence to detached reliability in [0,1]."""
    if divergence.ndim != 3:
        raise ValueError('divergence must have shape [B,H,W]')
    if num_classes < 2:
        raise ValueError('num_classes must be at least two')
    if not torch.isfinite(divergence).all() or (divergence < 0).any():
        raise ValueError('divergence must be finite and nonnegative')
    maximum = math.log(float(num_classes))
    return (1.0 - divergence.detach() / maximum).clamp(0.0, 1.0)


def occupancy_brier_map(candidate, exact):
    """Mean squared occupancy error over classes at each pixel."""
    if candidate.shape != exact.shape:
        raise ValueError('candidate and exact occupancy shapes must match')
    validate_occupancy(candidate)
    validate_occupancy(exact)
    return (candidate.detach() - exact.detach()).square().mean(dim=1)


def soft_dice_error_per_sample(candidate, exact, smooth=1e-10):
    """Squared-denominator soft-Dice error for every sample."""
    if candidate.shape != exact.shape or candidate.ndim != 4:
        raise ValueError('occupancies must have matched [B,C,H,W] shapes')
    validate_occupancy(candidate)
    validate_occupancy(exact)
    intersection = (candidate * exact).flatten(2).sum(dim=2)
    denominator = candidate.square().flatten(2).sum(dim=2) + \
        exact.square().flatten(2).sum(dim=2)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - score.mean(dim=1)


def binary_dice_per_sample(prediction, target, smooth=1e-10):
    """Foreground Dice for matched hard masks."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('hard masks must have matched [B,H,W] shapes')
    prediction = (prediction > 0).to(dtype=torch.float32)
    target = (target > 0).to(
        device=prediction.device, dtype=prediction.dtype)
    intersection = (prediction * target).flatten(1).sum(dim=1)
    denominator = prediction.flatten(1).sum(dim=1) + \
        target.flatten(1).sum(dim=1)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def normalized_weighted_mean(values, weights, mask=None):
    """Return a scalar mean under detached nonnegative reliability weights."""
    if values.shape != weights.shape:
        raise ValueError('values and weights must have identical shapes')
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
        raise FloatingPointError('weighted mean inputs must be finite')
    if (weights < 0).any():
        raise ValueError('weights must be nonnegative')
    active = torch.ones_like(values, dtype=torch.bool) if mask is None \
        else mask.bool()
    if active.shape != values.shape:
        raise ValueError('mask must match values')
    active_weights = weights.detach()[active]
    if active_weights.numel() == 0:
        return None
    denominator = active_weights.sum()
    if denominator.item() <= 0.0:
        return None
    return (values.detach()[active] * active_weights).sum() / denominator


def spearman_correlation(first, second):
    """Spearman correlation with average ranks for ties."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size != second.size:
        raise ValueError('Spearman inputs must have equal length')
    if first.size < 2 or not np.isfinite(first).all() or \
            not np.isfinite(second).all():
        return None

    def average_ranks(values):
        order = np.argsort(values, kind='mergesort')
        sorted_values = values[order]
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and \
                    sorted_values[end] == sorted_values[start]:
                end += 1
            average = 0.5 * (start + end - 1) + 1.0
            ranks[order[start:end]] = average
            start = end
        return ranks

    ranked_first = average_ranks(first)
    ranked_second = average_ranks(second)
    centered_first = ranked_first - ranked_first.mean()
    centered_second = ranked_second - ranked_second.mean()
    denominator = math.sqrt(
        float(np.dot(centered_first, centered_first)) *
        float(np.dot(centered_second, centered_second)))
    if denominator <= 0.0:
        return None
    return float(np.dot(centered_first, centered_second) / denominator)


def top_fraction_error_ratio(scores, errors, fraction=0.20):
    """Mean error in the exact highest-score fraction divided by the rest."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if scores.size != errors.size:
        raise ValueError('score and error arrays must have equal length')
    if not 0.0 < fraction < 1.0:
        raise ValueError('fraction must lie strictly between zero and one')
    if scores.size < 2 or not np.isfinite(scores).all() or \
            not np.isfinite(errors).all() or (errors < 0).any():
        return None
    top_count = max(1, int(math.ceil(fraction * scores.size)))
    if top_count >= scores.size:
        return None
    order = np.argsort(-scores, kind='mergesort')
    top_error = float(errors[order[:top_count]].mean())
    remaining_error = float(errors[order[top_count:]].mean())
    if remaining_error <= 0.0:
        return None
    return top_error / remaining_error
