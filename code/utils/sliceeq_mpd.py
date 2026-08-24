"""H7.19 robust moment-profile design and frozen sampler utilities.

The design is computed before segmentation training from exact labels in the
locked labeled-training subset.  The runtime sampler is global and
case-independent: it never reads an image, target, model prediction, loss,
validation metric or iteration number.
"""

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize


SCHEMA_VERSION = 'h7.19-mpd-v1'
GRID_SIDE = 21
SIGMA_RANGE = (0.45, 0.85)
PHASE_RANGE = (-0.25, 0.25)
MOMENT_TOLERANCE = 0.02
IMAGE_RESIDUAL_TOLERANCE = 0.05
DENSITY_RATIO_CAP = 3.0
ENTROPY_FRACTION_MIN = 0.70
UTILITY_OPTIMUM_FRACTION = 0.99
NUMERICAL_EPSILON = 1e-12


class DesignError(RuntimeError):
    """Raised when the locked convex-design contract cannot be verified."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix='.mpd-', suffix='.json.tmp', dir=directory)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(
                payload, stream, indent=2, sort_keys=True,
                allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def midpoint_profile_grid(grid_side=GRID_SIDE):
    if grid_side != GRID_SIDE:
        raise ValueError('H7.19 locks a 21x21 profile grid')
    sigma = SIGMA_RANGE[0] + (
        np.arange(grid_side, dtype=np.float64) + 0.5) * (
            (SIGMA_RANGE[1] - SIGMA_RANGE[0]) / grid_side)
    phase = PHASE_RANGE[0] + (
        np.arange(grid_side, dtype=np.float64) + 0.5) * (
            (PHASE_RANGE[1] - PHASE_RANGE[0]) / grid_side)
    sigmas = np.repeat(sigma, grid_side)
    phases = np.tile(phase, grid_side)
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    logits = -0.5 * (
        (offsets[None, :] - phases[:, None]) /
        sigmas[:, None]) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    parent = np.full(weights.shape[0], 1.0 / weights.shape[0])
    return sigmas, phases, weights, parent


def profile_moments(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError('profile weights must have shape [G,3]')
    neighbor_mass = weights[:, 0] + weights[:, 2]
    directional_mass = weights[:, 2] - weights[:, 0]
    if np.any(neighbor_mass <= 0.0):
        raise ValueError('profile neighbor mass must be positive')
    phase_ratio = directional_mass / neighbor_mass
    features = np.stack((
        neighbor_mass,
        neighbor_mass ** 2,
        directional_mass ** 2,
    ), axis=1)
    return neighbor_mass, phase_ratio, features


def pattern_counts(previous, center, following):
    previous = np.asarray(previous) > 0
    center = np.asarray(center) > 0
    following = np.asarray(following) > 0
    if previous.shape != center.shape or center.shape != following.shape:
        raise ValueError('three labels must have identical shapes')
    codes = (
        previous.astype(np.uint8) * 4 +
        center.astype(np.uint8) * 2 +
        following.astype(np.uint8))
    return np.bincount(codes.reshape(-1), minlength=8).astype(np.float64)


def occupancy_metrics_from_patterns(counts, weights, epsilon=1e-7):
    """Evaluate all profiles using the eight possible binary label patterns."""
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if counts.shape != (8,):
        raise ValueError('pattern counts must have shape [8]')
    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError('weights must have shape [G,3]')
    patterns = np.asarray([
        [(code >> 2) & 1, (code >> 1) & 1, code & 1]
        for code in range(8)], dtype=np.float64)
    foreground = weights @ patterns.T
    foreground = np.clip(foreground, 0.0, 1.0)
    center = patterns[:, 1]
    opportunity = np.logical_or(
        patterns[:, 0] != center, patterns[:, 2] != center)
    hard_foreground = foreground > 0.5
    retained = hard_foreground == center[None, :].astype(bool)
    p = np.clip(foreground, epsilon, 1.0 - epsilon)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    opportunity_count = float(counts[opportunity].sum())
    if opportunity_count <= 0.0:
        utility = np.full(weights.shape[0], np.nan, dtype=np.float64)
    else:
        utility = (
            entropy * retained * opportunity[None, :] * counts[None, :]
        ).sum(axis=1) / opportunity_count
    total = max(float(counts.sum()), 1.0)
    hard_change = (
        (hard_foreground != center[None, :].astype(bool)) *
        counts[None, :]
    ).sum(axis=1) / total
    fractional = np.logical_and(
        foreground > epsilon, foreground < 1.0 - epsilon)
    fractional_support = (
        fractional * counts[None, :]).sum(axis=1) / total
    center_mass = float((counts * center).sum())
    profile_mass = (foreground * counts[None, :]).sum(axis=1)
    mass_error = np.abs(profile_mass - center_mass) / max(center_mass, 1.0)
    return {
        'utility': utility,
        'hard_change': hard_change,
        'fractional_support': fractional_support,
        'foreground_mass_error': mass_error,
        'opportunity_pixels': int(opportunity_count),
    }


def normalized_axial_gram(image_stack, epsilon=1e-8):
    image_stack = np.asarray(image_stack, dtype=np.float64)
    if image_stack.shape[0] != 3:
        raise ValueError('image stack must have three slices')
    previous, center, following = image_stack
    first = 0.5 * (following - previous)
    second = previous - 2.0 * center + following
    center_scale = math.sqrt(float(np.mean((center - center.mean()) ** 2)))
    center_scale = max(center_scale, epsilon)
    return np.asarray([
        [np.mean(first * first), np.mean(first * second)],
        [np.mean(first * second), np.mean(second * second)],
    ], dtype=np.float64) / (center_scale ** 2)


def normalized_profile_residuals(weights, gram):
    weights = np.asarray(weights, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape != (2, 2):
        raise ValueError('axial Gram matrix must be 2x2')
    vectors = np.stack((
        weights[:, 2] - weights[:, 0],
        0.5 * (weights[:, 0] + weights[:, 2]),
    ), axis=1)
    squared = np.einsum('gi,ij,gj->g', vectors, gram, vectors)
    return np.sqrt(np.maximum(squared, 0.0))


def _mirror_projection(sigmas, phases):
    sigmas = np.asarray(sigmas, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    if sigmas.shape != phases.shape:
        raise ValueError('sigma and phase grids must have identical shapes')
    groups = []
    used = set()
    for index, phase in enumerate(phases):
        if index in used:
            continue
        candidates = np.where(np.logical_and(
            np.isclose(sigmas, sigmas[index], atol=1e-12),
            np.isclose(phases, -phase, atol=1e-12)))[0]
        if candidates.size != 1:
            raise DesignError('profile grid is not uniquely phase symmetric')
        mirror = int(candidates[0])
        used.add(index)
        used.add(mirror)
        groups.append((index,) if index == mirror else (index, mirror))
    projection = np.zeros((phases.size, len(groups)), dtype=np.float64)
    for group_index, members in enumerate(groups):
        for member in members:
            projection[member, group_index] = 1.0 / len(members)
    return projection, groups


def distribution_entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = probabilities > 0.0
    return float(-np.sum(
        probabilities[positive] * np.log(probabilities[positive])))


def js_divergence(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)

    def _kl(a, b):
        positive = a > 0.0
        return float(np.sum(a[positive] * np.log(a[positive] / b[positive])))

    return 0.5 * _kl(left, middle) + 0.5 * _kl(right, middle)


def _relative_bounds(reference, tolerance):
    reference = np.asarray(reference, dtype=np.float64)
    return reference * (1.0 - tolerance), reference * (1.0 + tolerance)


def _constraint_diagnostics(
        q, utilities, residuals, features, parent, utility_floor=None):
    parent_moments = parent @ features
    moments = q @ features
    lower_moments, upper_moments = _relative_bounds(
        parent_moments, MOMENT_TOLERANCE)
    parent_residuals = residuals @ parent
    designed_residuals = residuals @ q
    lower_residuals, upper_residuals = _relative_bounds(
        parent_residuals, IMAGE_RESIDUAL_TOLERANCE)
    utility = utilities @ q
    entropy = distribution_entropy(q)
    parent_entropy = distribution_entropy(parent)
    checks = {
        'simplex': bool(
            np.all(q >= -1e-10) and abs(float(q.sum()) - 1.0) <= 1e-8),
        'density_cap': bool(np.all(
            q <= DENSITY_RATIO_CAP * parent + 1e-9)),
        'moment_budget': bool(np.all(
            moments >= lower_moments - 1e-8) and np.all(
                moments <= upper_moments + 1e-8)),
        'image_residual_budget': bool(np.all(
            designed_residuals >= lower_residuals - 1e-8) and np.all(
                designed_residuals <= upper_residuals + 1e-8)),
        'entropy_floor': bool(
            entropy >= ENTROPY_FRACTION_MIN * parent_entropy - 1e-8),
    }
    if utility_floor is not None:
        checks['utility_floor'] = bool(
            np.all(utility >= utility_floor - 1e-8))
    return {
        'checks': checks,
        'all_pass': all(checks.values()),
        'worst_utility': float(np.min(utility)),
        'utilities': utility.tolist(),
        'moments': moments.tolist(),
        'parent_moments': parent_moments.tolist(),
        'image_residuals': designed_residuals.tolist(),
        'parent_image_residuals': parent_residuals.tolist(),
        'entropy': entropy,
        'parent_entropy': parent_entropy,
        'max_density_ratio': float(np.max(q / parent)),
    }


def design_robust_distribution(
        utilities, residuals, weights, sigmas, phases, parent=None):
    """Solve the locked two-stage H7.19 robust profile design."""
    utilities = np.asarray(utilities, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    profile_count = weights.shape[0]
    if utilities.ndim != 2 or utilities.shape[1] != profile_count:
        raise ValueError('utilities must have shape [S,G]')
    if residuals.ndim != 2 or residuals.shape[1] != profile_count:
        raise ValueError('residuals must have shape [R,G]')
    if utilities.shape[0] == 0 or residuals.shape[0] == 0:
        raise DesignError('design requires active utility and residual strata')
    if not np.isfinite(utilities).all() or not np.isfinite(residuals).all():
        raise DesignError('design inputs contain non-finite values')
    if np.any(utilities < 0.0) or np.any(residuals < 0.0):
        raise DesignError('design inputs must be nonnegative')
    if parent is None:
        parent = np.full(profile_count, 1.0 / profile_count)
    parent = np.asarray(parent, dtype=np.float64)
    if parent.shape != (profile_count,) or np.any(parent <= 0.0):
        raise ValueError('parent distribution must be strictly positive')
    parent = parent / parent.sum()
    _, _, features = profile_moments(weights)
    projection, groups = _mirror_projection(sigmas, phases)
    group_parent = np.asarray([
        parent[np.asarray(group, dtype=np.int64)].sum()
        for group in groups], dtype=np.float64)
    grouped_utilities = utilities @ projection
    grouped_residuals = residuals @ projection
    grouped_features = projection.T @ features
    parent_moments = parent @ features
    lower_moments, upper_moments = _relative_bounds(
        parent_moments, MOMENT_TOLERANCE)
    parent_residuals = residuals @ parent
    lower_residuals, upper_residuals = _relative_bounds(
        parent_residuals, IMAGE_RESIDUAL_TOLERANCE)
    entropy_floor = ENTROPY_FRACTION_MIN * distribution_entropy(parent)
    group_caps = np.asarray([
        len(group) * DENSITY_RATIO_CAP / profile_count for group in groups
    ], dtype=np.float64)

    def _q(z):
        return projection @ z

    def _entropy_from_z(z):
        return distribution_entropy(_q(z))

    constraints_stage1 = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1.0},
        {'type': 'ineq', 'fun': lambda x: (
            grouped_utilities @ x[:-1] - x[-1])},
        {'type': 'ineq', 'fun': lambda x: (
            grouped_features.T @ x[:-1] - lower_moments)},
        {'type': 'ineq', 'fun': lambda x: (
            upper_moments - grouped_features.T @ x[:-1])},
        {'type': 'ineq', 'fun': lambda x: (
            grouped_residuals @ x[:-1] - lower_residuals)},
        {'type': 'ineq', 'fun': lambda x: (
            upper_residuals - grouped_residuals @ x[:-1])},
        {'type': 'ineq', 'fun': lambda x: (
            _entropy_from_z(x[:-1]) - entropy_floor)},
    ]
    initial_t = float(np.min(grouped_utilities @ group_parent))
    initial = np.concatenate((group_parent, [initial_t]))
    bounds_stage1 = [
        (0.0, float(cap)) for cap in group_caps] + [(0.0, None)]
    result1 = minimize(
        lambda x: -x[-1], initial, method='SLSQP',
        bounds=bounds_stage1, constraints=constraints_stage1,
        options={'ftol': 1e-11, 'maxiter': 3000, 'disp': False})
    if not result1.success:
        raise DesignError(
            'stage-one robust design failed: {}'.format(result1.message))
    t_star = float(result1.x[-1])
    utility_floor = UTILITY_OPTIMUM_FRACTION * t_star

    def _kl_from_z(z):
        q = _q(z)
        positive = q > 0.0
        return float(np.sum(q[positive] * np.log(q[positive] / parent[positive])))

    constraints_stage2 = [
        {'type': 'eq', 'fun': lambda z: np.sum(z) - 1.0},
        {'type': 'ineq', 'fun': lambda z: (
            grouped_utilities @ z - utility_floor)},
        {'type': 'ineq', 'fun': lambda z: (
            grouped_features.T @ z - lower_moments)},
        {'type': 'ineq', 'fun': lambda z: (
            upper_moments - grouped_features.T @ z)},
        {'type': 'ineq', 'fun': lambda z: (
            grouped_residuals @ z - lower_residuals)},
        {'type': 'ineq', 'fun': lambda z: (
            upper_residuals - grouped_residuals @ z)},
        {'type': 'ineq', 'fun': lambda z: (
            _entropy_from_z(z) - entropy_floor)},
    ]
    result2 = minimize(
        _kl_from_z, result1.x[:-1], method='SLSQP',
        bounds=[(0.0, float(cap)) for cap in group_caps],
        constraints=constraints_stage2,
        options={'ftol': 1e-12, 'maxiter': 3000, 'disp': False})
    if not result2.success:
        raise DesignError(
            'stage-two KL projection failed: {}'.format(result2.message))
    q = _q(result2.x)
    q = np.maximum(q, 0.0)
    q /= q.sum()
    diagnostics = _constraint_diagnostics(
        q, utilities, residuals, features, parent,
        utility_floor=utility_floor)
    mirror_error = float(np.max(np.abs(
        q.reshape(GRID_SIDE, GRID_SIDE) -
        q.reshape(GRID_SIDE, GRID_SIDE)[:, ::-1])))
    diagnostics['mirror_error'] = mirror_error
    diagnostics['checks']['phase_mirror'] = mirror_error <= 1e-10
    diagnostics['all_pass'] = all(diagnostics['checks'].values())
    if not diagnostics['all_pass']:
        raise DesignError(
            'optimized distribution violates locked constraints: {}'.format(
                diagnostics['checks']))
    return {
        'probabilities': q,
        't_star': t_star,
        'utility_floor': utility_floor,
        'kl_to_parent': _kl_from_z(result2.x),
        'stage1_iterations': int(result1.nit),
        'stage2_iterations': int(result2.nit),
        'diagnostics': diagnostics,
    }


def distribution_hash(sigmas, phases, weights, probabilities):
    payload = {
        'schema_version': SCHEMA_VERSION,
        'sigmas': [float(value) for value in sigmas],
        'phases': [float(value) for value in phases],
        'weights': [[float(value) for value in row] for row in weights],
        'probabilities': [float(value) for value in probabilities],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _read_locked_labeled_training_data(root_path, output_size=(256, 256)):
    """Read only the locked seven-patient labeled-training prefix.

    This pre-pass intentionally does not construct a validation/test dataset and
    never reads an unlabeled label.  The returned arrays are resized with the
    same nearest-neighbour contract used by the SliceEq training dataset.
    """
    import h5py
    from scipy.ndimage import zoom

    from dataloaders.sliceeq_dataset import (
        build_neighbor_table, parse_slice_name)

    list_path = os.path.join(root_path, 'train_slices.list')
    if not os.path.isfile(list_path):
        raise FileNotFoundError('missing training slice list: {}'.format(
            list_path))
    with open(list_path, 'r', encoding='utf-8-sig') as stream:
        all_names = [line.strip() for line in stream if line.strip()]
    if len(all_names) < 192:
        raise DesignError('PROMISE12 training slice list is unexpectedly short')
    labeled_names = all_names[:191]
    labeled_cases = [parse_slice_name(name)[0] for name in labeled_names]
    case_order = list(dict.fromkeys(labeled_cases))
    if len(case_order) != 7:
        raise DesignError(
            'H7.19 requires exactly seven labeled patients in first 191 slices')
    remaining_cases = {
        parse_slice_name(name)[0] for name in all_names[191:]}
    overlap = sorted(set(case_order) & remaining_cases)
    if overlap:
        raise DesignError(
            'labeled/unlabeled boundary splits a patient: {}'.format(overlap))
    neighbor_table = build_neighbor_table(labeled_names, radius=1)

    cache = {}
    content_digest = hashlib.sha256()
    for name in labeled_names:
        path = os.path.join(root_path, 'data', 'slices', name + '.h5')
        if not os.path.isfile(path):
            raise FileNotFoundError('missing labeled H5 slice: {}'.format(path))
        with h5py.File(path, 'r') as stream:
            image = stream['image'][:]
            label = stream['label'][:]
        if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
            raise DesignError('invalid labeled H5 arrays: {}'.format(path))
        if image.shape != tuple(output_size):
            factors = (
                output_size[0] / image.shape[0],
                output_size[1] / image.shape[1])
            image = zoom(image, factors, order=0)
            label = zoom(label, factors, order=0)
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label > 0, dtype=np.uint8)
        if image.shape != tuple(output_size) or label.shape != tuple(output_size):
            raise DesignError('resized arrays violate locked patch size')
        cache[name] = (image, label)
        content_digest.update(name.encode('utf-8'))
        content_digest.update(image.tobytes(order='C'))
        content_digest.update(label.tobytes(order='C'))

    positions = {}
    for case_name in case_order:
        indices = [
            index for index, name in enumerate(labeled_names)
            if parse_slice_name(name)[0] == case_name]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise DesignError('patient slices are not contiguous: {}'.format(
                case_name))
        for position, global_index in enumerate(indices):
            third = min(2, (3 * position) // len(indices))
            positions[global_index] = (case_name, int(third))
    return {
        'all_names': all_names,
        'labeled_names': labeled_names,
        'case_order': case_order,
        'neighbor_table': neighbor_table,
        'cache': cache,
        'positions': positions,
        'train_slices_sha256': sha256_file(list_path),
        'labeled_content_sha256': content_digest.hexdigest(),
    }


def collect_exact_design_statistics(root_path, output_size=(256, 256)):
    """Build patient-by-index-third exact occupancy/profile statistics."""
    data = _read_locked_labeled_training_data(root_path, output_size)
    sigmas, phases, weights, parent = midpoint_profile_grid()
    strata = [
        (case_name, third)
        for case_name in data['case_order'] for third in range(3)]
    stratum_index = {key: index for index, key in enumerate(strata)}
    profile_count = weights.shape[0]
    utility_sum = np.zeros((len(strata), profile_count), dtype=np.float64)
    residual_sum = np.zeros_like(utility_sum)
    hard_change_sum = np.zeros_like(utility_sum)
    fractional_sum = np.zeros_like(utility_sum)
    mass_error_sum = np.zeros_like(utility_sum)
    active_count = np.zeros(len(strata), dtype=np.int64)
    slice_count = np.zeros(len(strata), dtype=np.int64)
    opportunity_pixels = np.zeros(len(strata), dtype=np.int64)
    clamped_count = np.zeros(len(strata), dtype=np.int64)

    for index, (neighbor_names, was_clamped) in enumerate(
            data['neighbor_table']):
        images = np.stack([
            data['cache'][name][0] for name in neighbor_names], axis=0)
        labels = np.stack([
            data['cache'][name][1] for name in neighbor_names], axis=0)
        metrics = occupancy_metrics_from_patterns(
            pattern_counts(labels[0], labels[1], labels[2]), weights)
        residual = normalized_profile_residuals(
            weights, normalized_axial_gram(images))
        stratum = stratum_index[data['positions'][index]]
        slice_count[stratum] += 1
        clamped_count[stratum] += int(was_clamped)
        residual_sum[stratum] += residual
        hard_change_sum[stratum] += metrics['hard_change']
        fractional_sum[stratum] += metrics['fractional_support']
        mass_error_sum[stratum] += metrics['foreground_mass_error']
        if metrics['opportunity_pixels'] > 0:
            utility_sum[stratum] += metrics['utility']
            active_count[stratum] += 1
            opportunity_pixels[stratum] += metrics['opportunity_pixels']

    if int(slice_count.sum()) != 191 or np.any(slice_count == 0):
        raise DesignError('incomplete H7.19 patient-stratum coverage')
    active_strata = active_count > 0
    for case_name in data['case_order']:
        patient_rows = np.asarray([
            key[0] == case_name for key in strata], dtype=bool)
        if not np.any(active_strata[patient_rows]):
            raise DesignError(
                'labeled patient has no axial label opportunity: {}'.format(
                    case_name))
    utilities = np.zeros_like(utility_sum)
    utilities[active_strata] = (
        utility_sum[active_strata] / active_count[active_strata, None])
    residuals = residual_sum / slice_count[:, None]
    hard_change = hard_change_sum / slice_count[:, None]
    fractional = fractional_sum / slice_count[:, None]
    mass_error = mass_error_sum / slice_count[:, None]
    if not all(np.isfinite(value).all() for value in (
            utilities, residuals, hard_change, fractional, mass_error)):
        raise DesignError('non-finite exact training statistic')
    return {
        'sigmas': sigmas,
        'phases': phases,
        'weights': weights,
        'parent': parent,
        'utilities': utilities,
        'residuals': residuals,
        'hard_change': hard_change,
        'fractional_support': fractional,
        'foreground_mass_error': mass_error,
        'strata': strata,
        'active_slice_count': active_count,
        'slice_count': slice_count,
        'opportunity_pixels': opportunity_pixels,
        'active_strata': active_strata,
        'clamped_slice_count': clamped_count,
        'case_order': data['case_order'],
        'train_slices_sha256': data['train_slices_sha256'],
        'labeled_content_sha256': data['labeled_content_sha256'],
    }


def build_direct_design_artifact(root_path, output_path, protocol_path=None):
    """Design q on all seven labeled patients and atomically freeze it.

    This is the user-authorized direct-training path.  It deliberately skips
    the previously proposed LOPO gate, but still fails closed if the locked
    mathematical design is infeasible or violates any of its safety budgets.
    """
    statistics = collect_exact_design_statistics(root_path)
    active_strata = statistics['active_strata']
    result = design_robust_distribution(
        statistics['utilities'][active_strata], statistics['residuals'],
        statistics['weights'], statistics['sigmas'], statistics['phases'],
        parent=statistics['parent'])
    q = result['probabilities']
    p0 = statistics['parent']
    stratum_names = [
        '{}:index-third-{}'.format(case_name, third)
        for case_name, third in statistics['strata']]
    parent_utility = statistics['utilities'] @ p0
    designed_utility = statistics['utilities'] @ q
    active_stratum_names = [
        name for name, active in zip(stratum_names, active_strata)
        if active]
    empty_stratum_names = [
        name for name, active in zip(stratum_names, active_strata)
        if not active]

    def _expectation(values, distribution):
        return (values @ distribution).tolist()

    distribution_sha = distribution_hash(
        statistics['sigmas'], statistics['phases'],
        statistics['weights'], q)
    report = {
        'schema_version': SCHEMA_VERSION,
        'method': 'SliceEqOcc-OAAC-Strong-MPD',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'execution_mode': 'user_override_skip_lopo_direct_full_training',
        'decision': 'exploratory_direct_design_ready',
        'training_authorized': True,
        'evidence_scope': (
            'all-seven-labeled-training-patient design; no LOPO gate; '
            'no validation/test/model output used'),
        'data_firewall': {
            'labeled_slices_read': 191,
            'labeled_patients_read': 7,
            'unlabeled_labels_read': 0,
            'validation_or_test_read': False,
            'model_predictions_read': False,
            'train_slices_sha256': statistics['train_slices_sha256'],
            'labeled_image_label_content_sha256':
                statistics['labeled_content_sha256'],
            'patient_ids': statistics['case_order'],
        },
        'locked_design': {
            'grid': '21x21 midpoint sigma-phase grid',
            'sigma_range': list(SIGMA_RANGE),
            'phase_range': list(PHASE_RANGE),
            'moment_tolerance': MOMENT_TOLERANCE,
            'image_residual_tolerance': IMAGE_RESIDUAL_TOLERANCE,
            'density_ratio_cap': DENSITY_RATIO_CAP,
            'entropy_fraction_min': ENTROPY_FRACTION_MIN,
            'utility_optimum_fraction': UTILITY_OPTIMUM_FRACTION,
            'protocol_path': protocol_path,
            'protocol_sha256': sha256_file(protocol_path)
                if protocol_path and os.path.isfile(protocol_path) else None,
        },
        'full_design': {
            'probabilities': [float(value) for value in q],
            'distribution_sha256': distribution_sha,
            't_star': result['t_star'],
            'utility_floor': result['utility_floor'],
            'kl_to_parent': result['kl_to_parent'],
            'stage1_iterations': result['stage1_iterations'],
            'stage2_iterations': result['stage2_iterations'],
            'diagnostics': result['diagnostics'],
        },
        'patient_strata': {
            'names': stratum_names,
            'active_for_rfi_optimization': active_strata.tolist(),
            'active_rfi_strata': active_stratum_names,
            'structurally_empty_rfi_strata': empty_stratum_names,
            'empty_stratum_policy': (
                'excluded only from max-min RFI because opportunity denominator '
                'is zero; retained in image-residual constraints and reporting'),
            'slice_count': statistics['slice_count'].tolist(),
            'active_slice_count': statistics['active_slice_count'].tolist(),
            'opportunity_pixels': statistics['opportunity_pixels'].tolist(),
            'clamped_slice_count': statistics['clamped_slice_count'].tolist(),
            'parent_expected_rfi': parent_utility.tolist(),
            'designed_expected_rfi': designed_utility.tolist(),
            'relative_rfi_change': (
                (designed_utility - parent_utility) /
                np.maximum(parent_utility, NUMERICAL_EPSILON)).tolist(),
            'parent_expected_hard_change': _expectation(
                statistics['hard_change'], p0),
            'designed_expected_hard_change': _expectation(
                statistics['hard_change'], q),
            'parent_expected_fractional_support': _expectation(
                statistics['fractional_support'], p0),
            'designed_expected_fractional_support': _expectation(
                statistics['fractional_support'], q),
            'parent_expected_foreground_mass_error': _expectation(
                statistics['foreground_mass_error'], p0),
            'designed_expected_foreground_mass_error': _expectation(
                statistics['foreground_mass_error'], q),
        },
        'conditions': dict(result['diagnostics']['checks']),
    }
    report['conditions']['solver_and_design_constraints'] = bool(
        result['diagnostics']['all_pass'])
    if not all(report['conditions'].values()):
        raise DesignError('direct profile design failed a locked condition')
    atomic_json_dump(report, output_path)
    validated = validate_design_artifact(report)
    validated['artifact_sha256'] = sha256_file(output_path)
    validated['report'] = report
    return validated


def validate_design_artifact(report):
    """Validate either a historical gate-pass or direct-training artifact."""
    if report.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported H7.19 artifact schema')
    allowed = {
        'pass',
        'exploratory_direct_design_ready',
    }
    if report.get('decision') not in allowed or not report.get(
            'training_authorized', False):
        raise ValueError('H7.19 artifact does not authorize training')
    design = report.get('full_design', {})
    sigmas, phases, weights, parent = midpoint_profile_grid()
    probabilities = np.asarray(
        design.get('probabilities', []), dtype=np.float64)
    if probabilities.shape != parent.shape:
        raise ValueError('H7.19 artifact distribution has wrong length')
    if np.any(probabilities < 0.0) or not np.isfinite(probabilities).all():
        raise ValueError('H7.19 probabilities are invalid')
    if abs(float(probabilities.sum()) - 1.0) > 1e-8:
        raise ValueError('H7.19 probabilities do not sum to one')
    mirror_error = np.max(np.abs(
        probabilities.reshape(GRID_SIDE, GRID_SIDE) -
        probabilities.reshape(GRID_SIDE, GRID_SIDE)[:, ::-1]))
    if mirror_error > 1e-10:
        raise ValueError('H7.19 probabilities violate phase symmetry')
    expected_hash = distribution_hash(
        sigmas, phases, weights, probabilities)
    if design.get('distribution_sha256') != expected_hash:
        raise ValueError('H7.19 distribution hash mismatch')
    if not report.get('conditions') or not all(
            report['conditions'].values()):
        raise ValueError('H7.19 artifact has a failed design condition')
    return {
        'sigmas': sigmas,
        'phases': phases,
        'weights': weights,
        'parent': parent,
        'probabilities': probabilities,
        'distribution_sha256': expected_hash,
    }


def validate_gate_artifact(report):
    if report.get('decision') != 'pass' or not report.get(
            'training_authorized', False):
        raise ValueError('H7.19 gate did not authorize training')
    return validate_design_artifact(report)


def load_gate_artifact(path):
    with open(path, 'r', encoding='utf-8') as stream:
        report = json.load(stream)
    design = validate_gate_artifact(report)
    design['artifact_sha256'] = sha256_file(path)
    design['report'] = report
    return design


def load_design_artifact(path):
    with open(path, 'r', encoding='utf-8') as stream:
        report = json.load(stream)
    design = validate_design_artifact(report)
    design['artifact_sha256'] = sha256_file(path)
    design['report'] = report
    return design


def sample_frozen_profiles(
        batch_size, weights, sigmas, phases, probabilities,
        device, generator=None):
    """Sample a global discrete profile distribution with a private RNG."""
    import torch

    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    weight_tensor = torch.as_tensor(
        weights, dtype=torch.float32, device=device)
    sigma_tensor = torch.as_tensor(
        sigmas, dtype=torch.float32, device=device)
    phase_tensor = torch.as_tensor(
        phases, dtype=torch.float32, device=device)
    probability_tensor = torch.as_tensor(
        probabilities, dtype=torch.float32, device=device)
    indices = torch.multinomial(
        probability_tensor, batch_size, replacement=True,
        generator=generator)
    sampled_weights = weight_tensor.index_select(0, indices)
    if not torch.isfinite(sampled_weights).all() or \
            not torch.allclose(
                sampled_weights.sum(dim=1),
                torch.ones(batch_size, device=device),
                atol=1e-6, rtol=1e-6):
        raise RuntimeError('H7.19 sampler produced invalid weights')
    return (
        sampled_weights,
        sigma_tensor.index_select(0, indices),
        phase_tensor.index_select(0, indices),
    )


class FrozenProfileSampler:
    """Device-cached parent-compatible sampler for one immutable global q."""

    def __init__(self, design):
        self._weights = np.asarray(design['weights'], dtype=np.float32)
        self._sigmas = np.asarray(design['sigmas'], dtype=np.float32)
        self._phases = np.asarray(design['phases'], dtype=np.float32)
        self._probabilities = np.asarray(
            design['probabilities'], dtype=np.float32)
        self.distribution_sha256 = design['distribution_sha256']
        self._cache = {}

    def _tensors(self, device):
        import torch

        key = str(torch.device(device))
        if key not in self._cache:
            self._cache[key] = (
                torch.as_tensor(self._weights, device=device),
                torch.as_tensor(self._sigmas, device=device),
                torch.as_tensor(self._phases, device=device),
                torch.as_tensor(self._probabilities, device=device),
            )
        return self._cache[key]

    def __call__(
            self, batch_size, offsets, sigma_range, phase_range, device,
            generator=None):
        import torch

        if tuple(float(value) for value in offsets) != (-1.0, 0.0, 1.0):
            raise ValueError('MPD locks three offsets [-1,0,1]')
        if tuple(float(value) for value in sigma_range) != SIGMA_RANGE:
            raise ValueError('MPD locks the parent sigma support')
        if tuple(float(value) for value in phase_range) != PHASE_RANGE:
            raise ValueError('MPD locks the parent phase support')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        weights, sigmas, phases, probabilities = self._tensors(device)
        indices = torch.multinomial(
            probabilities, batch_size, replacement=True,
            generator=generator)
        sampled_weights = weights.index_select(0, indices)
        if not torch.isfinite(sampled_weights).all() or not torch.allclose(
                sampled_weights.sum(dim=1),
                torch.ones(batch_size, device=device),
                atol=1e-6, rtol=1e-6):
            raise RuntimeError('MPD sampler produced invalid weights')
        return (
            sampled_weights,
            sigmas.index_select(0, indices),
            phases.index_select(0, indices),
        )
