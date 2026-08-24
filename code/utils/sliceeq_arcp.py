"""H5-only axial-response calibration for SliceEq three-tap profiles.

This module does not estimate scanner slice thickness or a physical PSF.  It
normalizes the *observed effect* of the existing acquisition-inspired profile
using only neighboring training images.  Labels, predictions, losses, and
validation/test data are deliberately absent from the reference estimator.
"""

from collections import OrderedDict
import hashlib
import json
import math
import os

import h5py
import numpy as np
from scipy.ndimage import zoom
import torch

from dataloaders.sliceeq_dataset import parse_slice_name


REFERENCE_VERSION = 'arcp_h5_patient_balanced_gram_v1'
PARENT_CENTER_WEIGHT_MIN = 0.4850
PARENT_CENTER_WEIGHT_MAX = 0.8553
ACTIVITY_MARGIN = 0.05
DEFAULT_EPSILON = 1e-8


def _read_nonempty_lines(path):
    with open(path, 'r', encoding='utf-8-sig') as stream:
        return [line.strip() for line in stream if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _resize_image(image, output_size):
    if image.ndim != 2:
        raise ValueError('ARCP expects 2-D H5 image slices')
    height, width = image.shape
    if height < 1 or width < 1:
        raise ValueError('ARCP received an empty H5 image')
    factors = (output_size[0] / height, output_size[1] / width)
    return zoom(image, factors, order=0).astype(np.float64, copy=False)


def _case_groups(sample_list):
    groups = OrderedDict()
    for sample_name in sample_list:
        case_name, slice_index = parse_slice_name(sample_name)
        groups.setdefault(case_name, []).append((slice_index, sample_name))
    for case_name, entries in groups.items():
        entries.sort(key=lambda item: item[0])
        indices = [item[0] for item in entries]
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            raise ValueError(
                'ARCP requires contiguous slices for {}'.format(case_name))
    return groups


def _load_case_images(root_path, entries, output_size):
    images = []
    for _, sample_name in entries:
        path = os.path.join(
            root_path, 'data', 'slices', sample_name + '.h5')
        with h5py.File(path, 'r') as stream:
            if 'image' not in stream:
                raise KeyError('ARCP H5 image is missing: {}'.format(path))
            # Intentionally access only `image`; labels are outside this
            # estimator's data contract.
            image = stream['image'][:]
        images.append(_resize_image(image, output_size))
    return np.stack(images, axis=0)


def _numpy_response_gram(image_stack, epsilon=DEFAULT_EPSILON):
    if image_stack.shape[0] != 3 or image_stack.ndim != 3:
        raise ValueError('ARCP NumPy stack must have shape [3,H,W]')
    previous, center, following = image_stack
    centered = center - center.mean(dtype=np.float64)
    scale_squared = np.mean(centered * centered, dtype=np.float64)
    if not np.isfinite(scale_squared) or scale_squared <= epsilon:
        return None
    first = 0.5 * (following - previous)
    second = previous - 2.0 * center + following
    matrix = np.asarray([
        [np.mean(first * first, dtype=np.float64),
         np.mean(first * second, dtype=np.float64)],
        [np.mean(first * second, dtype=np.float64),
         np.mean(second * second, dtype=np.float64)],
    ], dtype=np.float64) / scale_squared
    if not np.isfinite(matrix).all():
        return None
    return 0.5 * (matrix + matrix.T)


def calibrate_profile_weights_numpy(
        image_stack, weights, reference_matrix,
        center_weight_min=PARENT_CENTER_WEIGHT_MIN,
        center_weight_max=PARENT_CENTER_WEIGHT_MAX,
        epsilon=DEFAULT_EPSILON):
    """NumPy equivalent of :func:`calibrate_profile_weights` for gates."""
    image_stack = np.asarray(image_stack, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    reference_matrix = np.asarray(reference_matrix, dtype=np.float64)
    if image_stack.ndim != 3 or image_stack.shape[0] != 3:
        raise ValueError('ARCP NumPy image stack must have shape [3,H,W]')
    if weights.shape != (3,) or reference_matrix.shape != (2, 2):
        raise ValueError('ARCP NumPy weight/reference shapes are invalid')
    if not np.isfinite(weights).all() or (weights < 0.0).any() or \
            abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError('ARCP NumPy parent weights are invalid')
    matrix = _numpy_response_gram(image_stack, epsilon=epsilon)
    duplicate = np.array_equal(image_stack[0], image_stack[1]) or \
        np.array_equal(image_stack[1], image_stack[2])
    moment = np.asarray([
        weights[2] - weights[0],
        0.5 * (weights[0] + weights[2]),
    ], dtype=np.float64)
    reference_squared = max(
        float(moment @ reference_matrix @ moment), 0.0)
    if matrix is None or duplicate:
        current_squared = reference_squared
        raw_alpha = 1.0
        alpha = 1.0
        eligible = False
        at_lower = False
        at_upper = False
    else:
        current_squared = max(float(moment @ matrix @ moment), 0.0)
        raw_alpha = math.sqrt(
            (reference_squared + epsilon) /
            (current_squared + epsilon))
        neighbor_mass = 1.0 - weights[1]
        lower = (1.0 - center_weight_max) / neighbor_mass
        upper = (1.0 - center_weight_min) / neighbor_mass
        alpha = min(max(raw_alpha, lower), upper)
        eligible = True
        at_lower = abs(alpha - lower) <= 1e-6
        at_upper = abs(alpha - upper) <= 1e-6
    identity = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    calibrated = identity + alpha * (weights - identity)
    if not np.isfinite(calibrated).all() or calibrated.min() < -1e-7 or \
            abs(calibrated.sum() - 1.0) > 1e-6:
        raise FloatingPointError('ARCP NumPy calibration produced bad weights')
    return calibrated, {
        'alpha': float(alpha),
        'raw_alpha': float(raw_alpha),
        'eligible': bool(eligible),
        'duplicate_support': bool(duplicate),
        'at_lower_bound': bool(at_lower),
        'at_upper_bound': bool(at_upper),
        'effect_before': math.sqrt(max(current_squared, 0.0)),
        'effect_after': float(alpha) * math.sqrt(max(current_squared, 0.0)),
        'effect_reference': math.sqrt(max(reference_squared, 0.0)),
    }


def compute_patient_balanced_reference(root_path, output_size=(256, 256),
                                       epsilon=DEFAULT_EPSILON):
    """Compute one image-only axial Gram reference from training H5 files.

    Interior stacks are averaged within each patient first; patient matrices
    are then averaged with equal weight.  Endpoints are excluded because the
    parent loader duplicates endpoint slices there.
    """
    if len(output_size) != 2 or min(output_size) < 2:
        raise ValueError('ARCP output_size must contain two positive sizes')
    if epsilon <= 0.0:
        raise ValueError('ARCP epsilon must be positive')
    list_path = os.path.join(root_path, 'train_slices.list')
    sample_list = _read_nonempty_lines(list_path)
    if not sample_list:
        raise ValueError('ARCP train_slices.list is empty')
    groups = _case_groups(sample_list)
    case_matrices = []
    case_records = []
    for case_name, entries in groups.items():
        if len(entries) < 3:
            raise ValueError(
                'ARCP case has fewer than three slices: {}'.format(case_name))
        images = _load_case_images(root_path, entries, output_size)
        matrices = []
        for index in range(1, images.shape[0] - 1):
            matrix = _numpy_response_gram(
                images[index - 1:index + 2], epsilon=epsilon)
            if matrix is not None:
                matrices.append(matrix)
        if not matrices:
            raise ValueError(
                'ARCP case has no nondegenerate interior stack: {}'.format(
                    case_name))
        case_matrix = np.mean(np.stack(matrices), axis=0, dtype=np.float64)
        case_matrices.append(case_matrix)
        case_records.append({
            'case': case_name,
            'slice_count': len(entries),
            'eligible_interior_stacks': len(matrices),
            'matrix': case_matrix.tolist(),
        })
    reference = np.mean(
        np.stack(case_matrices), axis=0, dtype=np.float64)
    reference = 0.5 * (reference + reference.T)
    eigenvalues = np.linalg.eigvalsh(reference)
    if not np.isfinite(reference).all() or eigenvalues.min() < -1e-7:
        raise FloatingPointError('ARCP reference matrix is not finite PSD')
    return {
        'version': REFERENCE_VERSION,
        'root_path': os.path.abspath(root_path),
        'train_slices_path': os.path.abspath(list_path),
        'train_slices_sha256': _sha256(list_path),
        'output_size': [int(output_size[0]), int(output_size[1])],
        'epsilon': float(epsilon),
        'patient_count': len(case_records),
        'eligible_interior_stacks': int(sum(
            item['eligible_interior_stacks'] for item in case_records)),
        'reference_matrix': reference.tolist(),
        'reference_eigenvalues': eigenvalues.tolist(),
        'cases': case_records,
    }


def save_reference_report(report, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def reference_tensor(report, device, dtype=torch.float32):
    matrix = torch.as_tensor(
        report['reference_matrix'], dtype=dtype, device=device)
    if matrix.shape != (2, 2) or not torch.isfinite(matrix).all():
        raise ValueError('ARCP reference matrix must be finite 2x2')
    return matrix


def response_gram(image_stack, epsilon=DEFAULT_EPSILON):
    """Return per-stack normalized `[B,2,2]` axial response matrices."""
    if image_stack.ndim != 5 or image_stack.shape[1] != 3:
        raise ValueError('ARCP image_stack must have shape [B,3,C,H,W]')
    if not image_stack.is_floating_point():
        raise TypeError('ARCP image_stack must be floating point')
    if epsilon <= 0.0:
        raise ValueError('ARCP epsilon must be positive')
    if not torch.isfinite(image_stack).all():
        raise FloatingPointError('ARCP image_stack contains non-finite values')
    previous, center, following = (
        image_stack[:, 0], image_stack[:, 1], image_stack[:, 2])
    first = 0.5 * (following - previous)
    second = previous - 2.0 * center + following
    center_flat = center.flatten(1)
    centered = center_flat - center_flat.mean(dim=1, keepdim=True)
    scale_squared = centered.square().mean(dim=1)
    first_flat = first.flatten(1)
    second_flat = second.flatten(1)
    c11 = first_flat.square().mean(dim=1)
    c12 = (first_flat * second_flat).mean(dim=1)
    c22 = second_flat.square().mean(dim=1)
    denominator = scale_squared + epsilon
    matrix = torch.stack((c11, c12, c12, c22), dim=1).reshape(-1, 2, 2)
    matrix = matrix / denominator.view(-1, 1, 1)
    valid = torch.logical_and(
        scale_squared > epsilon,
        torch.isfinite(matrix).flatten(1).all(dim=1))
    return matrix, valid


def _quadratic_form(matrix, vector):
    return torch.einsum('bi,bij,bj->b', vector, matrix, vector)


@torch.no_grad()
def calibrate_profile_weights(
        image_stack, weights, reference_matrix,
        center_weight_min=PARENT_CENTER_WEIGHT_MIN,
        center_weight_max=PARENT_CENTER_WEIGHT_MAX,
        activity_margin=ACTIVITY_MARGIN, epsilon=DEFAULT_EPSILON):
    """Calibrate parent weights along the identity-to-profile ray.

    The function is image-only.  It returns detached calibrated weights and
    scalar diagnostics; it never accepts a label, prediction, or loss.
    """
    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError('ARCP weights must have shape [B,3]')
    if weights.shape[0] != image_stack.shape[0]:
        raise ValueError('ARCP image/weight batch sizes differ')
    if reference_matrix.shape != (2, 2):
        raise ValueError('ARCP reference_matrix must have shape [2,2]')
    if not 0.0 < center_weight_min < center_weight_max < 1.0:
        raise ValueError('ARCP center-weight bounds are invalid')
    if activity_margin <= 0.0 or epsilon <= 0.0:
        raise ValueError('ARCP margins and epsilon must be positive')
    if not torch.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError('ARCP parent weights must be finite and nonnegative')
    if not torch.allclose(
            weights.sum(dim=1), torch.ones_like(weights[:, 0]),
            atol=1e-6, rtol=1e-6):
        raise ValueError('ARCP parent weights must sum to one')
    if not torch.isfinite(reference_matrix).all():
        raise ValueError('ARCP reference matrix must be finite')

    matrix, valid = response_gram(image_stack, epsilon=epsilon)
    reference = reference_matrix.to(
        dtype=image_stack.dtype, device=image_stack.device)
    reference = reference.unsqueeze(0).expand(image_stack.shape[0], -1, -1)
    moment = torch.stack((
        weights[:, 2] - weights[:, 0],
        0.5 * (weights[:, 0] + weights[:, 2])), dim=1)
    current_squared = _quadratic_form(matrix, moment).clamp_min(0.0)
    reference_squared = _quadratic_form(reference, moment).clamp_min(0.0)
    raw_alpha = torch.sqrt(
        (reference_squared + epsilon) / (current_squared + epsilon))

    center = weights[:, 1]
    neighbor_mass = 1.0 - center
    if (neighbor_mass <= 0.0).any():
        raise ValueError('ARCP parent profile must be nonidentity')
    lower_alpha = (1.0 - center_weight_max) / neighbor_mass
    upper_alpha = (1.0 - center_weight_min) / neighbor_mass
    alpha = torch.maximum(raw_alpha, lower_alpha)
    alpha = torch.minimum(alpha, upper_alpha)

    duplicate_support = torch.logical_or(
        (image_stack[:, 0] == image_stack[:, 1]).flatten(1).all(dim=1),
        (image_stack[:, 1] == image_stack[:, 2]).flatten(1).all(dim=1))
    eligible = torch.logical_and(valid, ~duplicate_support)
    alpha = torch.where(eligible, alpha, torch.ones_like(alpha))

    identity = torch.zeros_like(weights)
    identity[:, 1] = 1.0
    calibrated = identity + alpha.unsqueeze(1) * (weights - identity)
    if not torch.isfinite(calibrated).all() or (calibrated < -1e-7).any():
        raise FloatingPointError('ARCP produced invalid profile weights')
    if not torch.allclose(
            calibrated.sum(dim=1), torch.ones_like(calibrated[:, 0]),
            atol=1e-6, rtol=1e-6):
        raise FloatingPointError('ARCP weights no longer sum to one')

    effective_before = torch.sqrt(current_squared.clamp_min(0.0))
    effective_after = alpha * effective_before
    at_lower = torch.logical_and(
        eligible, (alpha - lower_alpha).abs() <= 1e-6)
    at_upper = torch.logical_and(
        eligible, (alpha - upper_alpha).abs() <= 1e-6)
    metadata = {
        'arcp_alpha_mean': alpha.mean(),
        'arcp_alpha_std': alpha.std(unbiased=False),
        'arcp_abs_alpha_minus_one_mean': (alpha - 1.0).abs().mean(),
        'arcp_active_sample_fraction': torch.logical_and(
            eligible, (alpha - 1.0).abs() >= activity_margin
        ).to(image_stack.dtype).mean(),
        'arcp_lower_bound_fraction': at_lower.to(image_stack.dtype).mean(),
        'arcp_upper_bound_fraction': at_upper.to(image_stack.dtype).mean(),
        'arcp_eligible_fraction': eligible.to(image_stack.dtype).mean(),
        'arcp_duplicate_support_fraction': (
            duplicate_support.to(image_stack.dtype).mean()),
        'arcp_parent_center_weight_mean': center.mean(),
        'arcp_calibrated_center_weight_mean': calibrated[:, 1].mean(),
        'arcp_effect_before_mean': effective_before.mean(),
        'arcp_effect_after_mean': effective_after.mean(),
        'arcp_effect_reference_mean': torch.sqrt(
            reference_squared.clamp_min(0.0)).mean(),
    }
    return calibrated.detach(), metadata
