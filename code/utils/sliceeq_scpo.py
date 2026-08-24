"""Slab-coherent hard pseudo occupancy for SliceEqOcc-SCPO."""

import numpy as np
import torch
from skimage.measure import label as connected_components


def slab_largest_connected_component(hard_stack):
    """Keep one 26-connected foreground component in each three-slice slab.

    Args:
        hard_stack: binary tensor with shape ``[B,3,H,W]``.

    Returns:
        A tensor on the original device/dtype and the number of foreground
        components observed in every raw slab.
    """
    if hard_stack.ndim != 4 or hard_stack.shape[1] != 3:
        raise ValueError('SCPO expects a binary [B,3,H,W] tensor')
    if not torch.isfinite(hard_stack).all():
        raise FloatingPointError('SCPO hard stack contains non-finite values')
    if ((hard_stack != 0) & (hard_stack != 1)).any():
        raise ValueError('SCPO supports binary hard pseudo masks only')

    foreground = hard_stack.detach().cpu().numpy().astype(np.uint8)
    coherent = np.zeros_like(foreground, dtype=np.uint8)
    component_counts = []
    for batch_index in range(foreground.shape[0]):
        labels = connected_components(
            foreground[batch_index], connectivity=3)
        component_count = int(labels.max())
        component_counts.append(component_count)
        if component_count == 0:
            continue
        counts = np.bincount(labels.reshape(-1))[1:]
        largest_label = int(np.argmax(counts)) + 1
        coherent[batch_index] = labels == largest_label

    coherent_tensor = torch.as_tensor(
        coherent, device=hard_stack.device, dtype=hard_stack.dtype)
    count_tensor = torch.as_tensor(
        component_counts, device=hard_stack.device, dtype=torch.float32)
    return coherent_tensor, count_tensor


def scpo_diagnostics(raw_stack, parent_2d_stack, coherent_stack,
                     raw_component_counts, epsilon=1e-7):
    """Return detached activity diagnostics without changing training."""
    expected_shape = raw_stack.shape
    if expected_shape != parent_2d_stack.shape or \
            expected_shape != coherent_stack.shape:
        raise ValueError('SCPO diagnostic stacks must have identical shapes')
    if raw_component_counts.shape != (raw_stack.shape[0],):
        raise ValueError('raw component counts must have shape [B]')

    raw = raw_stack.detach() > 0
    parent = parent_2d_stack.detach() > 0
    coherent = coherent_stack.detach() > 0
    changed = parent != coherent
    removed = parent & ~coherent
    added = ~parent & coherent
    parent_mass = parent.float().sum()
    coherent_mass = coherent.float().sum()
    retained_components = coherent.flatten(1).any(dim=1).float()
    active_slabs = changed.flatten(1).any(dim=1).float()

    return {
        'changed_pixel_fraction': changed.float().mean(),
        'removed_parent_foreground_fraction': removed.float().mean(),
        'added_parent_foreground_fraction': added.float().mean(),
        'coherent_to_parent_foreground_mass_ratio': (
            coherent_mass / parent_mass.clamp_min(epsilon)),
        'raw_foreground_fraction': raw.float().mean(),
        'parent_2d_foreground_fraction': parent.float().mean(),
        'coherent_foreground_fraction': coherent.float().mean(),
        'raw_component_count_mean': raw_component_counts.detach().mean(),
        'retained_component_count_mean': retained_components.mean(),
        'active_slab_fraction': active_slabs.mean(),
    }
