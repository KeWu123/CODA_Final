"""Offline robustness audits for the frozen SliceEq MPD design.

This module deliberately lives outside the training path.  It reads the same
seven labeled PROMISE12 patients used by MPD calibration, recomputes exact
design statistics, and evaluates numerical/design robustness without loading
a network checkpoint or reading validation/test data.
"""

import csv
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize

from utils import sliceeq_mpd as mpd


AUDIT_SCHEMA_VERSION = 'sliceeq-mpd-offline-audit-v1'
DEFAULT_GRID_SIDES = (11, 21, 31)
DEFAULT_AXIAL_BINS = (2, 3, 4)
DEFAULT_MOMENT_TOLERANCES = (0.01, 0.02, 0.05)
DEFAULT_RESIDUAL_TOLERANCES = (0.025, 0.05, 0.10)
DEFAULT_UTILITY_FRACTIONS = (0.98, 0.99, 0.995)


def midpoint_profile_grid(grid_side):
    """Build an odd midpoint grid over the locked MPD support."""
    grid_side = int(grid_side)
    if grid_side < 3 or grid_side % 2 == 0:
        raise ValueError('grid_side must be an odd integer >= 3')
    sigma = mpd.SIGMA_RANGE[0] + (
        np.arange(grid_side, dtype=np.float64) + 0.5) * (
            (mpd.SIGMA_RANGE[1] - mpd.SIGMA_RANGE[0]) / grid_side)
    phase = mpd.PHASE_RANGE[0] + (
        np.arange(grid_side, dtype=np.float64) + 0.5) * (
            (mpd.PHASE_RANGE[1] - mpd.PHASE_RANGE[0]) / grid_side)
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


def _relative_bounds(reference, tolerance):
    reference = np.asarray(reference, dtype=np.float64)
    return reference * (1.0 - tolerance), reference * (1.0 + tolerance)


def _minimum_normalized_slack(values, lower, upper):
    scale = np.maximum(np.abs(upper - lower), mpd.NUMERICAL_EPSILON)
    lower_slack = np.min((values - lower) / scale)
    upper_slack = np.min((upper - values) / scale)
    return float(lower_slack), float(upper_slack)


def _constraint_slacks(
        q, utilities, residuals, features, parent, utility_floor,
        moment_tolerance, residual_tolerance, density_ratio_cap,
        entropy_fraction_min, include_density_cap, include_entropy_floor):
    parent_moments = parent @ features
    moments = q @ features
    lower_moments, upper_moments = _relative_bounds(
        parent_moments, moment_tolerance)
    parent_residuals = residuals @ parent
    designed_residuals = residuals @ q
    lower_residuals, upper_residuals = _relative_bounds(
        parent_residuals, residual_tolerance)
    utility_values = utilities @ q
    moment_lower, moment_upper = _minimum_normalized_slack(
        moments, lower_moments, upper_moments)
    residual_lower, residual_upper = _minimum_normalized_slack(
        designed_residuals, lower_residuals, upper_residuals)
    parent_entropy = mpd.distribution_entropy(parent)
    entropy_fraction = (
        mpd.distribution_entropy(q) /
        max(parent_entropy, mpd.NUMERICAL_EPSILON))
    max_density_ratio = float(np.max(q / parent))
    utility_scale = max(abs(float(utility_floor)), mpd.NUMERICAL_EPSILON)
    utility_slack = float(
        np.min(utility_values - utility_floor) / utility_scale)
    slacks = {
        'simplex_absolute': abs(float(q.sum()) - 1.0),
        'nonnegative_min_probability': float(np.min(q)),
        'utility_relative': utility_slack,
        'moment_lower_normalized': moment_lower,
        'moment_upper_normalized': moment_upper,
        'residual_lower_normalized': residual_lower,
        'residual_upper_normalized': residual_upper,
        'max_density_ratio': max_density_ratio,
        'density_cap_absolute': (
            float(density_ratio_cap - max_density_ratio)
            if include_density_cap else None),
        'entropy_fraction': float(entropy_fraction),
        'entropy_floor_absolute': (
            float(entropy_fraction - entropy_fraction_min)
            if include_entropy_floor else None),
    }
    active_tolerance = 1e-5
    slacks['active_constraints'] = {
        'utility_floor': utility_slack <= active_tolerance,
        'moment_lower': moment_lower <= active_tolerance,
        'moment_upper': moment_upper <= active_tolerance,
        'residual_lower': residual_lower <= active_tolerance,
        'residual_upper': residual_upper <= active_tolerance,
        'density_cap': bool(
            include_density_cap and
            slacks['density_cap_absolute'] <= active_tolerance),
        'entropy_floor': bool(
            include_entropy_floor and
            slacks['entropy_floor_absolute'] <= active_tolerance),
    }
    return slacks


def design_distribution(
        utilities, residuals, weights, sigmas, phases, parent=None,
        moment_tolerance=mpd.MOMENT_TOLERANCE,
        residual_tolerance=mpd.IMAGE_RESIDUAL_TOLERANCE,
        density_ratio_cap=mpd.DENSITY_RATIO_CAP,
        entropy_fraction_min=mpd.ENTROPY_FRACTION_MIN,
        utility_optimum_fraction=mpd.UTILITY_OPTIMUM_FRACTION,
        include_density_cap=True, include_entropy_floor=True):
    """Generalized copy of the frozen two-stage design for audit only."""
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
        raise mpd.DesignError('audit design needs utility and residual strata')
    if not np.isfinite(utilities).all() or not np.isfinite(residuals).all():
        raise mpd.DesignError('audit design inputs contain non-finite values')
    if parent is None:
        parent = np.full(profile_count, 1.0 / profile_count)
    parent = np.asarray(parent, dtype=np.float64)
    parent /= parent.sum()
    _, _, features = mpd.profile_moments(weights)
    projection, groups = mpd._mirror_projection(sigmas, phases)
    group_parent = np.asarray([
        parent[np.asarray(group, dtype=np.int64)].sum()
        for group in groups], dtype=np.float64)
    grouped_utilities = utilities @ projection
    grouped_residuals = residuals @ projection
    grouped_features = projection.T @ features
    parent_moments = parent @ features
    lower_moments, upper_moments = _relative_bounds(
        parent_moments, moment_tolerance)
    parent_residuals = residuals @ parent
    lower_residuals, upper_residuals = _relative_bounds(
        parent_residuals, residual_tolerance)
    entropy_floor = (
        entropy_fraction_min * mpd.distribution_entropy(parent))

    def _q(z):
        return projection @ z

    def _entropy(z):
        return mpd.distribution_entropy(_q(z))

    constraints1 = [
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
    ]
    if include_entropy_floor:
        constraints1.append({
            'type': 'ineq',
            'fun': lambda x: _entropy(x[:-1]) - entropy_floor,
        })
    if include_density_cap:
        caps = density_ratio_cap * group_parent
        group_bounds = [(0.0, float(cap)) for cap in caps]
    else:
        group_bounds = [(0.0, 1.0) for _ in groups]
    initial_t = float(np.min(grouped_utilities @ group_parent))
    initial = np.concatenate((group_parent, [initial_t]))
    result1 = minimize(
        lambda x: -x[-1], initial, method='SLSQP',
        bounds=group_bounds + [(0.0, None)], constraints=constraints1,
        options={'ftol': 1e-11, 'maxiter': 3000, 'disp': False})
    if not result1.success:
        raise mpd.DesignError(
            'audit stage-one design failed: {}'.format(result1.message))
    t_star = float(result1.x[-1])
    utility_floor = float(utility_optimum_fraction * t_star)

    def _kl(z):
        q = _q(z)
        positive = q > 0.0
        return float(np.sum(
            q[positive] * np.log(q[positive] / parent[positive])))

    constraints2 = [
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
    ]
    if include_entropy_floor:
        constraints2.append({
            'type': 'ineq',
            'fun': lambda z: _entropy(z) - entropy_floor,
        })
    result2 = minimize(
        _kl, result1.x[:-1], method='SLSQP', bounds=group_bounds,
        constraints=constraints2,
        options={'ftol': 1e-12, 'maxiter': 3000, 'disp': False})
    if not result2.success:
        raise mpd.DesignError(
            'audit stage-two projection failed: {}'.format(result2.message))
    q = np.maximum(_q(result2.x), 0.0)
    q /= q.sum()
    grid_side = int(round(math.sqrt(profile_count)))
    if grid_side * grid_side != profile_count:
        raise ValueError('profile count is not a square grid')
    mirror_error = float(np.max(np.abs(
        q.reshape(grid_side, grid_side) -
        q.reshape(grid_side, grid_side)[:, ::-1])))
    slacks = _constraint_slacks(
        q, utilities, residuals, features, parent, utility_floor,
        moment_tolerance, residual_tolerance, density_ratio_cap,
        entropy_fraction_min, include_density_cap, include_entropy_floor)
    slacks['mirror_error'] = mirror_error
    checks = {
        'simplex': bool(
            np.min(q) >= -1e-10 and abs(float(q.sum()) - 1.0) <= 1e-8),
        'utility_floor': slacks['utility_relative'] >= -1e-7,
        'moment_budget': bool(
            slacks['moment_lower_normalized'] >= -1e-7 and
            slacks['moment_upper_normalized'] >= -1e-7),
        'residual_budget': bool(
            slacks['residual_lower_normalized'] >= -1e-7 and
            slacks['residual_upper_normalized'] >= -1e-7),
        'density_cap': bool(
            not include_density_cap or
            slacks['density_cap_absolute'] >= -1e-7),
        'entropy_floor': bool(
            not include_entropy_floor or
            slacks['entropy_floor_absolute'] >= -1e-7),
        'phase_mirror': mirror_error <= 1e-10,
    }
    if not all(checks.values()):
        raise mpd.DesignError(
            'audit solution violates constraints: {}'.format(checks))
    return {
        'probabilities': q,
        't_star': t_star,
        'utility_floor': utility_floor,
        'kl_to_parent': _kl(result2.x),
        'stage1_iterations': int(result1.nit),
        'stage2_iterations': int(result2.nit),
        'checks': checks,
        'slacks': slacks,
    }


def collect_slice_statistics(data, grid_side):
    """Compute per-slice exact statistics so binning/LOPO can be repeated."""
    sigmas, phases, weights, parent = midpoint_profile_grid(grid_side)
    profile_count = weights.shape[0]
    slice_count = len(data['labeled_names'])
    utilities = np.zeros((slice_count, profile_count), dtype=np.float64)
    residuals = np.zeros_like(utilities)
    hard_change = np.zeros_like(utilities)
    fractional_support = np.zeros_like(utilities)
    mass_error = np.zeros_like(utilities)
    active = np.zeros(slice_count, dtype=bool)
    opportunity_pixels = np.zeros(slice_count, dtype=np.int64)
    clamped = np.zeros(slice_count, dtype=bool)
    case_names = []
    case_positions = np.zeros(slice_count, dtype=np.int64)
    case_lengths = np.zeros(slice_count, dtype=np.int64)
    indices_by_case = {}
    for index in range(slice_count):
        case_name = data['positions'][index][0]
        indices_by_case.setdefault(case_name, []).append(index)
        case_names.append(case_name)
    for case_name, indices in indices_by_case.items():
        for position, index in enumerate(indices):
            case_positions[index] = position
            case_lengths[index] = len(indices)
    for index, item in enumerate(data['neighbor_table']):
        neighbor_names, was_clamped = item
        images = np.stack([
            data['cache'][name][0] for name in neighbor_names], axis=0)
        labels = np.stack([
            data['cache'][name][1] for name in neighbor_names], axis=0)
        metrics = mpd.occupancy_metrics_from_patterns(
            mpd.pattern_counts(labels[0], labels[1], labels[2]), weights)
        residuals[index] = mpd.normalized_profile_residuals(
            weights, mpd.normalized_axial_gram(images))
        hard_change[index] = metrics['hard_change']
        fractional_support[index] = metrics['fractional_support']
        mass_error[index] = metrics['foreground_mass_error']
        opportunity_pixels[index] = metrics['opportunity_pixels']
        active[index] = metrics['opportunity_pixels'] > 0
        clamped[index] = bool(was_clamped)
        if active[index]:
            utilities[index] = metrics['utility']
    arrays = (utilities, residuals, hard_change, fractional_support, mass_error)
    if not all(np.isfinite(value).all() for value in arrays):
        raise mpd.DesignError('non-finite per-slice audit statistic')
    return {
        'grid_side': int(grid_side),
        'sigmas': sigmas,
        'phases': phases,
        'weights': weights,
        'parent': parent,
        'utilities': utilities,
        'residuals': residuals,
        'hard_change': hard_change,
        'fractional_support': fractional_support,
        'foreground_mass_error': mass_error,
        'active': active,
        'opportunity_pixels': opportunity_pixels,
        'clamped': clamped,
        'case_names': np.asarray(case_names),
        'case_positions': case_positions,
        'case_lengths': case_lengths,
        'case_order': list(data['case_order']),
    }


def aggregate_statistics(slice_statistics, axial_bins, included_cases=None):
    """Aggregate per-slice values into patient-by-relative-position strata."""
    axial_bins = int(axial_bins)
    if axial_bins < 1:
        raise ValueError('axial_bins must be positive')
    case_order = slice_statistics['case_order']
    if included_cases is None:
        included_cases = case_order
    included_cases = [case for case in case_order if case in included_cases]
    if not included_cases:
        raise ValueError('included_cases is empty')
    strata = [
        (case_name, axial_bin)
        for case_name in included_cases for axial_bin in range(axial_bins)]
    profile_count = slice_statistics['weights'].shape[0]
    outputs = {
        key: np.zeros((len(strata), profile_count), dtype=np.float64)
        for key in (
            'utilities', 'residuals', 'hard_change',
            'fractional_support', 'foreground_mass_error')}
    slice_count = np.zeros(len(strata), dtype=np.int64)
    active_count = np.zeros(len(strata), dtype=np.int64)
    opportunity_pixels = np.zeros(len(strata), dtype=np.int64)
    clamped_count = np.zeros(len(strata), dtype=np.int64)
    stratum_index = {key: index for index, key in enumerate(strata)}
    for index, case_name in enumerate(slice_statistics['case_names']):
        if case_name not in included_cases:
            continue
        relative_bin = min(
            axial_bins - 1,
            (axial_bins * int(slice_statistics['case_positions'][index])) //
            int(slice_statistics['case_lengths'][index]))
        row = stratum_index[(case_name, relative_bin)]
        slice_count[row] += 1
        clamped_count[row] += int(slice_statistics['clamped'][index])
        for key in (
                'residuals', 'hard_change', 'fractional_support',
                'foreground_mass_error'):
            outputs[key][row] += slice_statistics[key][index]
        if slice_statistics['active'][index]:
            outputs['utilities'][row] += slice_statistics['utilities'][index]
            active_count[row] += 1
            opportunity_pixels[row] += int(
                slice_statistics['opportunity_pixels'][index])
    if np.any(slice_count == 0):
        raise mpd.DesignError(
            'axial binning created an empty patient stratum')
    active_strata = active_count > 0
    outputs['utilities'][active_strata] /= active_count[active_strata, None]
    for key in (
            'residuals', 'hard_change', 'fractional_support',
            'foreground_mass_error'):
        outputs[key] /= slice_count[:, None]
    outputs.update({
        'strata': strata,
        'slice_count': slice_count,
        'active_slice_count': active_count,
        'active_strata': active_strata,
        'opportunity_pixels': opportunity_pixels,
        'clamped_slice_count': clamped_count,
        'case_order': included_cases,
    })
    return outputs


def summarize_design(name, design, statistics, parent, reference=None):
    q = design['probabilities']
    active = statistics['active_strata']
    utilities = statistics['utilities'][active]
    parent_utility = utilities @ parent
    designed_utility = utilities @ q
    relative_change = (
        (designed_utility - parent_utility) /
        np.maximum(parent_utility, mpd.NUMERICAL_EPSILON))
    phase_ratio = None
    features = None
    if 'weights' in statistics:
        _, phase_ratio, features = mpd.profile_moments(statistics['weights'])
    row = {
        'variant': name,
        'status': 'ok',
        'profile_count': int(q.size),
        'worst_utility': float(np.min(designed_utility)),
        'parent_worst_utility': float(np.min(parent_utility)),
        'worst_relative_gain': float(np.min(relative_change)),
        'median_relative_gain': float(np.median(relative_change)),
        'mean_relative_gain': float(np.mean(relative_change)),
        'entropy_fraction': float(
            mpd.distribution_entropy(q) /
            mpd.distribution_entropy(parent)),
        'max_density_ratio': float(np.max(q / parent)),
        'kl_to_parent': float(design['kl_to_parent']),
        't_star': float(design['t_star']),
        'stage1_iterations': int(design['stage1_iterations']),
        'stage2_iterations': int(design['stage2_iterations']),
    }
    if reference is not None and reference.shape == q.shape:
        row['js_to_reference'] = float(mpd.js_divergence(q, reference))
        row['max_abs_to_reference'] = float(np.max(np.abs(q - reference)))
    else:
        row['js_to_reference'] = None
        row['max_abs_to_reference'] = None
    if features is not None:
        row['neighbor_mass_mean'] = float((q @ features)[0])
        row['neighbor_mass_second_moment'] = float((q @ features)[1])
        row['directional_mass_second_moment'] = float((q @ features)[2])
        row['absolute_phase_ratio_mean'] = float(q @ np.abs(phase_ratio))
    return row


def _design_for_statistics(slice_statistics, statistics, **kwargs):
    active = statistics['active_strata']
    result = design_distribution(
        statistics['utilities'][active], statistics['residuals'],
        slice_statistics['weights'], slice_statistics['sigmas'],
        slice_statistics['phases'], parent=slice_statistics['parent'],
        **kwargs)
    statistics = dict(statistics)
    statistics['weights'] = slice_statistics['weights']
    return result, statistics


def _failed_row(name, error):
    return {
        'variant': name,
        'status': 'failed',
        'error': '{}: {}'.format(type(error).__name__, error),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key != 'probabilities':
                keys.append(key)
    with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key) for key in keys
            })


def _fmt(value, digits=6):
    if value is None:
        return '-'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (float, np.floating)):
        return ('{:.%df}' % digits).format(float(value))
    return str(value)


def _markdown_table(rows, columns):
    if not rows:
        return '_No rows._\n'
    lines = [
        '| ' + ' | '.join(label for _, label in columns) + ' |',
        '|' + '|'.join('---' for _ in columns) + '|',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(
            _fmt(row.get(key)) for key, _ in columns) + ' |')
    return '\n'.join(lines) + '\n'


def write_audit_outputs(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tables = report['tables']
    for name in (
            'grid_convergence', 'axial_bins', 'tolerance_sensitivity',
            'constraint_activity', 'lopo_stability'):
        _write_csv(os.path.join(output_dir, name + '.csv'), tables[name])
    mpd.atomic_json_dump(
        report['config'], os.path.join(output_dir, 'audit_config.json'))
    mpd.atomic_json_dump(
        report['baseline_design'],
        os.path.join(output_dir, 'baseline_design.json'))
    mpd.atomic_json_dump(
        report['distribution_comparison'],
        os.path.join(output_dir, 'distribution_comparison.json'))
    mpd.atomic_json_dump(report, os.path.join(output_dir, 'audit_results.json'))

    baseline = report['baseline_design']['summary']
    lopo_ok = [
        row for row in tables['lopo_stability'] if row['status'] == 'ok']
    lopo_js = [row['js_to_full'] for row in lopo_ok]
    lopo_gain_pass = sum(
        row['heldout_worst_relative_gain'] >= 0.10 for row in lopo_ok)
    historical_gate = bool(
        lopo_js and np.median(lopo_js) < 0.05 and max(lopo_js) < 0.10 and
        lopo_gain_pass >= 5)
    if not report['config']['run_lopo']:
        historical_gate_text = 'not evaluated (--skip_lopo)'
    else:
        historical_gate_text = 'pass' if historical_gate else 'not passed'
    lines = [
        '# SliceEq MPD offline robustness audit',
        '',
        'This is a training-independent numerical audit. It does not report '
        'segmentation Dice and does not authorize model selection.',
        '',
        '## Data firewall',
        '',
        '- Labeled train patients read: **7**',
        '- Labeled train slices read: **191**',
        '- Unlabeled labels read: **0**',
        '- Validation/test data read: **No**',
        '- Checkpoints/predictions/losses read: **No**',
        '',
        '## Locked-design reproduction',
        '',
        '- Status: **{}**'.format(baseline['status']),
        '- Worst expected RFI: **{}**'.format(
            _fmt(baseline.get('worst_utility'))),
        '- Worst relative RFI gain over the uniform parent: **{}**'.format(
            _fmt(baseline.get('worst_relative_gain'))),
        '- Entropy fraction: **{}**'.format(
            _fmt(baseline.get('entropy_fraction'))),
        '- Maximum density ratio: **{}**'.format(
            _fmt(baseline.get('max_density_ratio'))),
        '',
        '## Discretization convergence',
        '',
        _markdown_table(tables['grid_convergence'], [
            ('variant', 'Variant'), ('status', 'Status'),
            ('worst_utility', 'Worst RFI'),
            ('worst_relative_gain', 'Worst gain'),
            ('entropy_fraction', 'Entropy fraction'),
            ('neighbor_mass_mean', 'E[neighbor mass]')]),
        '## Axial stratification sensitivity',
        '',
        _markdown_table(tables['axial_bins'], [
            ('variant', 'Variant'), ('status', 'Status'),
            ('js_to_reference', 'JS to 3-bin'),
            ('worst_relative_gain', 'Worst gain'),
            ('entropy_fraction', 'Entropy fraction')]),
        '## Constraint sensitivity',
        '',
        _markdown_table(tables['tolerance_sensitivity'], [
            ('variant', 'Variant'), ('status', 'Status'),
            ('js_to_reference', 'JS to locked'),
            ('worst_relative_gain', 'Worst gain'),
            ('max_density_ratio', 'Max density ratio')]),
        '## Leave-one-patient-out stability',
        '',
        _markdown_table(tables['lopo_stability'], [
            ('held_out_case', 'Held-out patient'), ('status', 'Status'),
            ('js_to_full', 'JS to full'),
            ('heldout_worst_relative_gain', 'Held-out worst gain'),
            ('heldout_median_relative_gain', 'Held-out median gain')]),
        'Historical preregistered diagnostic gate (not a new model-selection '
        'rule): **{}**. Median JS < 0.05, maximum JS < 0.10, and at least '
        '5/7 held-out patients with >=10% worst-stratum RFI gain.'.format(
            historical_gate_text),
        '',
        '## Interpretation boundary',
        '',
        'These tables test whether the train-only profile distribution is '
        'numerically stable and whether its conclusions depend strongly on '
        'discretization or arbitrary tolerances. They do not replace repeated '
        'training, external-dataset evaluation, calibration analysis, or '
        'segmentation significance tests.',
        '',
    ]
    with open(
            os.path.join(output_dir, 'audit_summary.md'),
            'w', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))


def run_offline_audit(
        root_path, output_dir, reference_artifact=None,
        grid_sides=DEFAULT_GRID_SIDES, axial_bins=DEFAULT_AXIAL_BINS,
        moment_tolerances=DEFAULT_MOMENT_TOLERANCES,
        residual_tolerances=DEFAULT_RESIDUAL_TOLERANCES,
        utility_fractions=DEFAULT_UTILITY_FRACTIONS,
        output_size=(256, 256), run_lopo=True):
    """Run the complete train-only MPD audit and write reproducible outputs."""
    root_path = os.path.abspath(root_path)
    output_dir = os.path.abspath(output_dir)
    sides = sorted(set(int(value) for value in grid_sides) | {21})
    bins_to_run = sorted(set(int(value) for value in axial_bins) | {3})
    data = mpd._read_locked_labeled_training_data(root_path, output_size)
    slice_by_side = {
        side: collect_slice_statistics(data, side) for side in sides}
    baseline_slices = slice_by_side[21]
    baseline_stats = aggregate_statistics(baseline_slices, 3)
    baseline_design, baseline_stats_for_summary = _design_for_statistics(
        baseline_slices, baseline_stats)
    baseline_q = baseline_design['probabilities']
    baseline_summary = summarize_design(
        'locked_grid21_bins3', baseline_design,
        baseline_stats_for_summary, baseline_slices['parent'])

    distributions = {
        'locked_grid21_bins3': baseline_q.tolist(),
    }
    grid_rows = []
    for side in sides:
        name = 'grid_{}x{}'.format(side, side)
        try:
            statistics = aggregate_statistics(slice_by_side[side], 3)
            design, stats_for_summary = _design_for_statistics(
                slice_by_side[side], statistics)
            row = summarize_design(
                name, design, stats_for_summary,
                slice_by_side[side]['parent'],
                baseline_q if side == 21 else None)
            row['grid_side'] = side
            grid_rows.append(row)
            distributions[name] = design['probabilities'].tolist()
        except Exception as error:
            row = _failed_row(name, error)
            row['grid_side'] = side
            grid_rows.append(row)

    bin_rows = []
    for bin_count in bins_to_run:
        name = 'axial_bins_{}'.format(bin_count)
        try:
            statistics = aggregate_statistics(baseline_slices, bin_count)
            design, stats_for_summary = _design_for_statistics(
                baseline_slices, statistics)
            row = summarize_design(
                name, design, stats_for_summary,
                baseline_slices['parent'], baseline_q)
            row['axial_bin_count'] = bin_count
            bin_rows.append(row)
            distributions[name] = design['probabilities'].tolist()
        except Exception as error:
            row = _failed_row(name, error)
            row['axial_bin_count'] = bin_count
            bin_rows.append(row)

    sensitivity_rows = []
    variants = []
    for value in moment_tolerances:
        variants.append((
            'moment_tolerance_{:.4g}'.format(value),
            {'moment_tolerance': float(value)}))
    for value in residual_tolerances:
        variants.append((
            'residual_tolerance_{:.4g}'.format(value),
            {'residual_tolerance': float(value)}))
    for value in utility_fractions:
        variants.append((
            'utility_fraction_{:.4g}'.format(value),
            {'utility_optimum_fraction': float(value)}))
    variants.extend([
        ('without_density_cap', {'include_density_cap': False}),
        ('without_entropy_floor', {'include_entropy_floor': False}),
    ])
    for name, kwargs in variants:
        try:
            design, stats_for_summary = _design_for_statistics(
                baseline_slices, baseline_stats, **kwargs)
            row = summarize_design(
                name, design, stats_for_summary,
                baseline_slices['parent'], baseline_q)
            sensitivity_rows.append(row)
            distributions[name] = design['probabilities'].tolist()
        except Exception as error:
            sensitivity_rows.append(_failed_row(name, error))

    constraint_rows = []
    activity_keys = {
        'utility_relative': 'utility_floor',
        'moment_lower_normalized': 'moment_lower',
        'moment_upper_normalized': 'moment_upper',
        'residual_lower_normalized': 'residual_lower',
        'residual_upper_normalized': 'residual_upper',
        'density_cap_absolute': 'density_cap',
        'entropy_floor_absolute': 'entropy_floor',
    }
    for key, value in baseline_design['slacks'].items():
        if key == 'active_constraints':
            continue
        constraint_rows.append({
            'constraint': key,
            'value': value,
            'active': baseline_design['slacks']['active_constraints'].get(
                activity_keys.get(key, ''), False),
        })

    lopo_rows = []
    if run_lopo:
        for held_out in baseline_slices['case_order']:
            name = 'lopo_without_{}'.format(held_out)
            try:
                training_cases = [
                    case for case in baseline_slices['case_order']
                    if case != held_out]
                training_stats = aggregate_statistics(
                    baseline_slices, 3, training_cases)
                design, _ = _design_for_statistics(
                    baseline_slices, training_stats)
                q = design['probabilities']
                heldout_stats = aggregate_statistics(
                    baseline_slices, 3, [held_out])
                active = heldout_stats['active_strata']
                utilities = heldout_stats['utilities'][active]
                parent_values = utilities @ baseline_slices['parent']
                designed_values = utilities @ q
                gains = (
                    (designed_values - parent_values) /
                    np.maximum(parent_values, mpd.NUMERICAL_EPSILON))
                row = {
                    'variant': name,
                    'held_out_case': held_out,
                    'status': 'ok',
                    'js_to_full': float(mpd.js_divergence(q, baseline_q)),
                    'max_abs_to_full': float(np.max(np.abs(q - baseline_q))),
                    'heldout_worst_relative_gain': float(np.min(gains)),
                    'heldout_median_relative_gain': float(np.median(gains)),
                    'heldout_mean_relative_gain': float(np.mean(gains)),
                    'heldout_active_strata': int(np.sum(active)),
                }
                lopo_rows.append(row)
                distributions[name] = q.tolist()
            except Exception as error:
                row = _failed_row(name, error)
                row['held_out_case'] = held_out
                lopo_rows.append(row)

    reference_comparison = {
        'provided': bool(reference_artifact),
        'path': os.path.abspath(reference_artifact)
            if reference_artifact else None,
    }
    if reference_artifact:
        try:
            with open(reference_artifact, 'r', encoding='utf-8') as stream:
                artifact = json.load(stream)
            reference_q = np.asarray(
                artifact.get('full_design', {}).get('probabilities', []),
                dtype=np.float64)
            if reference_q.shape != baseline_q.shape:
                raise ValueError('reference distribution does not have 441 values')
            reference_comparison.update({
                'status': 'ok',
                'js_divergence': float(
                    mpd.js_divergence(baseline_q, reference_q)),
                'max_absolute_difference': float(
                    np.max(np.abs(baseline_q - reference_q))),
                'artifact_sha256': mpd.sha256_file(reference_artifact),
            })
        except Exception as error:
            reference_comparison.update({
                'status': 'failed',
                'error': '{}: {}'.format(type(error).__name__, error),
            })

    config = {
        'schema_version': AUDIT_SCHEMA_VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'root_path': root_path,
        'output_dir': output_dir,
        'grid_sides': sides,
        'axial_bins': bins_to_run,
        'moment_tolerances': [float(v) for v in moment_tolerances],
        'residual_tolerances': [float(v) for v in residual_tolerances],
        'utility_fractions': [float(v) for v in utility_fractions],
        'output_size': list(output_size),
        'run_lopo': bool(run_lopo),
        'data_firewall': {
            'labeled_patients_read': 7,
            'labeled_slices_read': 191,
            'patient_ids': data['case_order'],
            'unlabeled_labels_read': 0,
            'validation_or_test_read': False,
            'model_checkpoints_read': False,
            'model_predictions_or_losses_read': False,
            'train_slices_sha256': data['train_slices_sha256'],
            'labeled_content_sha256': data['labeled_content_sha256'],
        },
    }
    report = {
        'schema_version': AUDIT_SCHEMA_VERSION,
        'config': config,
        'baseline_design': {
            'summary': baseline_summary,
            'probabilities': baseline_q.tolist(),
            'solver': baseline_design,
        },
        'reference_comparison': reference_comparison,
        'tables': {
            'grid_convergence': grid_rows,
            'axial_bins': bin_rows,
            'tolerance_sensitivity': sensitivity_rows,
            'constraint_activity': constraint_rows,
            'lopo_stability': lopo_rows,
        },
        'distribution_comparison': {
            'schema_version': AUDIT_SCHEMA_VERSION,
            'distributions': distributions,
        },
    }
    report = _json_ready(report)
    write_audit_outputs(report, output_dir)
    return report
