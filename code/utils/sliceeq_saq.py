"""Batch-stratified acquisition quadrature for SliceEq.

The sampler preserves the continuous uniform profile domain through a fixed
2x2 Gauss-Legendre rule.  Within each 12-sample branch every quadrature node
is assigned exactly three times, while a seeded permutation prevents node
identity from becoming coupled to dataloader order or anatomy.
"""

import math

import torch


QUADRATURE_NODE_COUNT = 4


def gauss_legendre_profile_nodes(sigma_range, phase_range):
    """Return the fixed 2x2 nodes for two independent uniform variables."""
    sigma_min, sigma_max = sigma_range
    phase_min, phase_max = phase_range
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError('invalid SliceEq sigma range')
    if phase_max < phase_min:
        raise ValueError('invalid SliceEq phase range')

    inverse_sqrt_three = 1.0 / math.sqrt(3.0)
    sigma_midpoint = 0.5 * (sigma_min + sigma_max)
    sigma_half_width = 0.5 * (sigma_max - sigma_min)
    phase_midpoint = 0.5 * (phase_min + phase_max)
    phase_half_width = 0.5 * (phase_max - phase_min)
    sigma_nodes = (
        sigma_midpoint - sigma_half_width * inverse_sqrt_three,
        sigma_midpoint + sigma_half_width * inverse_sqrt_three,
    )
    phase_nodes = (
        phase_midpoint - phase_half_width * inverse_sqrt_three,
        phase_midpoint + phase_half_width * inverse_sqrt_three,
    )
    return tuple(
        (sigma, phase)
        for sigma in sigma_nodes
        for phase in phase_nodes)


def sample_stratified_slice_profiles(
        batch_size, offsets, sigma_range, phase_range, device,
        generator=None):
    """Assign every 2x2 quadrature node equally within one batch branch.

    Returns normalized Gaussian profile weights, sigma, phase, and node id.
    The only randomness is a permutation of the balanced node table.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if batch_size % QUADRATURE_NODE_COUNT != 0:
        raise ValueError(
            'batch_size must be divisible by four for balanced quadrature')
    if len(offsets) < 1:
        raise ValueError('offsets must not be empty')

    nodes = gauss_legendre_profile_nodes(sigma_range, phase_range)
    node_table = torch.as_tensor(
        nodes, dtype=torch.float32, device=device)
    repeats = batch_size // QUADRATURE_NODE_COUNT
    node_ids = torch.arange(
        QUADRATURE_NODE_COUNT, device=device, dtype=torch.long)
    node_ids = node_ids.repeat_interleave(repeats)
    permutation = torch.randperm(
        batch_size, device=device, generator=generator)
    node_ids = node_ids[permutation]
    sigma = node_table[node_ids, 0]
    phase = node_table[node_ids, 1]

    offset_tensor = torch.as_tensor(
        offsets, dtype=sigma.dtype, device=device).view(1, -1)
    logits = -0.5 * (
        (offset_tensor - phase.view(-1, 1)) /
        sigma.view(-1, 1)).square()
    weights = torch.softmax(logits, dim=1)
    if not torch.isfinite(weights).all():
        raise FloatingPointError(
            'SliceEqSAQ profile weights contain non-finite values')
    if not torch.allclose(
            weights.sum(dim=1), torch.ones_like(sigma),
            atol=1e-6, rtol=1e-6):
        raise ValueError('SliceEqSAQ profile weights must sum to one')
    return weights, sigma, phase, node_ids


def quadrature_assignment_diagnostics(node_ids):
    """Report coverage and imbalance without inspecting image or target."""
    if node_ids.ndim != 1 or node_ids.numel() < 1:
        raise ValueError('node_ids must be a non-empty vector')
    if node_ids.dtype != torch.long:
        raise ValueError('node_ids must use torch.long dtype')
    if node_ids.min().item() < 0 or \
            node_ids.max().item() >= QUADRATURE_NODE_COUNT:
        raise ValueError('node_ids are outside the quadrature node range')
    counts = torch.bincount(
        node_ids, minlength=QUADRATURE_NODE_COUNT).to(torch.float32)
    expected = node_ids.numel() / float(QUADRATURE_NODE_COUNT)
    return {
        'quadrature_node_coverage': (counts > 0).float().mean(),
        'quadrature_max_count_deviation': (
            counts - expected).abs().max() / node_ids.numel(),
    }
