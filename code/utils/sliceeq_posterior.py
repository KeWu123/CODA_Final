"""Posterior-commutation operators for the SliceEq H7.4 fidelity gate."""

import torch

from utils.sliceeq_occ import validate_occupancy


def profile_weighted_distribution(distribution_stack, weights):
    """Apply a normalized slice profile to a stack of class distributions."""
    if distribution_stack.ndim != 5:
        raise ValueError(
            'distribution_stack must have shape [B,K,C,H,W]')
    if weights.ndim != 2:
        raise ValueError('weights must have shape [B,K]')
    if distribution_stack.shape[:2] != weights.shape:
        raise ValueError('distribution stack and weight dimensions differ')
    if not torch.isfinite(distribution_stack).all():
        raise FloatingPointError('distribution stack contains non-finite values')
    if (distribution_stack < -1e-6).any() or \
            (distribution_stack > 1.0 + 1e-6).any():
        raise ValueError('distribution values must lie in [0,1]')
    sums = distribution_stack.sum(dim=2)
    if not torch.allclose(
            sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError('each stack distribution must sum to one')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('profile weights must be finite and nonnegative')
    weight_sums = weights.sum(dim=1)
    if not torch.allclose(
            weight_sums, torch.ones_like(weight_sums),
            atol=1e-6, rtol=1e-6):
        raise ValueError('profile weights must sum to one')

    expanded_weights = weights.view(
        weights.shape[0], weights.shape[1], 1, 1, 1)
    output = (distribution_stack * expanded_weights).sum(dim=1)
    validate_occupancy(output)
    return output


def topology_gate_binary_posterior(posterior_stack, lcc_stack):
    """Keep soft foreground probability only inside the hard foreground LCC."""
    if posterior_stack.ndim != 5 or posterior_stack.shape[2] != 2:
        raise ValueError(
            'posterior_stack must have binary shape [B,K,2,H,W]')
    if lcc_stack.shape != posterior_stack.shape[:2] + \
            posterior_stack.shape[-2:]:
        raise ValueError('lcc_stack must have shape [B,K,H,W]')
    if not torch.isfinite(posterior_stack).all():
        raise FloatingPointError('posterior stack contains non-finite values')
    if (lcc_stack < 0).any() or (lcc_stack > 1).any():
        raise ValueError('lcc_stack must be binary')
    sums = posterior_stack.sum(dim=2)
    if not torch.allclose(
            sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError('posterior distributions must sum to one')

    foreground = posterior_stack[:, :, 1] * lcc_stack.to(
        dtype=posterior_stack.dtype)
    gated = torch.stack((1.0 - foreground, foreground), dim=2)
    if not torch.isfinite(gated).all():
        raise FloatingPointError('topology-gated posterior is non-finite')
    return gated


def distribution_residual(reacquired_distribution, center_distribution):
    """Total-variation acquisition change between two class distributions."""
    if reacquired_distribution.shape != center_distribution.shape:
        raise ValueError('distribution shapes must match')
    validate_occupancy(reacquired_distribution)
    validate_occupancy(center_distribution)
    return 0.5 * (
        reacquired_distribution.detach() -
        center_distribution.detach()).abs().sum(dim=1)


def occupancy_brier_map(candidate, exact):
    """Mean squared occupancy error over classes at each pixel."""
    if candidate.shape != exact.shape:
        raise ValueError('candidate and exact occupancy shapes must match')
    validate_occupancy(candidate)
    validate_occupancy(exact)
    return (candidate.detach() - exact.detach()).square().mean(dim=1)
