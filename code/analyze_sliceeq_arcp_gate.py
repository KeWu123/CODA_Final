"""Zero-training H5-only gate for Axial-Response Calibrated Profiles.

The analyzer reads every training image but labels only for the locked first
191 labeled slices.  It never constructs validation/test datasets or loads a
segmentation checkpoint.
"""

import argparse
import json
import math
import os

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.stats import spearmanr

from dataloaders.sliceeq_dataset import parse_slice_name
from utils.sliceeq_arcp import (
    ACTIVITY_MARGIN, PARENT_CENTER_WEIGHT_MAX, PARENT_CENTER_WEIGHT_MIN,
    _case_groups, _load_case_images, _numpy_response_gram,
    _read_nonempty_lines, calibrate_profile_weights_numpy,
    compute_patient_balanced_reference, save_reference_report)


LABELED_SLICES = 191
SIGMA_GRID = (0.45, 0.65, 0.85)
PHASE_GRID = (-0.25, 0.0, 0.25)
PROFILE_GRID = tuple(
    (sigma, phase) for sigma in SIGMA_GRID for phase in PHASE_GRID)


def _profile_weights(sigma, phase):
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    logits = -0.5 * ((offsets - phase) / sigma) ** 2
    values = np.exp(logits - logits.max())
    return values / values.sum()


def _load_case_labels(root_path, entries, output_size):
    labels = []
    for _, sample_name in entries:
        path = os.path.join(
            root_path, 'data', 'slices', sample_name + '.h5')
        with h5py.File(path, 'r') as stream:
            if 'label' not in stream:
                raise KeyError('ARCP labeled H5 lacks label: {}'.format(path))
            label = stream['label'][:]
        factors = (
            output_size[0] / label.shape[0],
            output_size[1] / label.shape[1])
        labels.append(zoom(label, factors, order=0).astype(np.uint8))
    return np.stack(labels, axis=0)


def _safe_cv(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    return float(values.std() / mean) if mean > 1e-12 else None


def _safe_corr(left, right, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.logical_and(np.isfinite(left), np.isfinite(right))
    left, right = left[finite], right[finite]
    if left.size < 3 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    if rank:
        value = spearmanr(left, right).statistic
    else:
        value = np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else None


def _mean_finite(values):
    finite = [value for value in values if value is not None and
              math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _median_finite(values):
    finite = [value for value in values if value is not None and
              math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _at_least(value, threshold):
    return value is not None and math.isfinite(value) and value >= threshold


def _at_most(value, threshold):
    return value is not None and math.isfinite(value) and value <= threshold


def _third(index, count):
    fraction = index / max(count - 1, 1)
    if fraction < 1.0 / 3.0:
        return 'first'
    if fraction < 2.0 / 3.0:
        return 'middle'
    return 'last'


def _analyze(root_path, output_size):
    reference_report = compute_patient_balanced_reference(
        root_path, output_size=output_size)
    reference = np.asarray(
        reference_report['reference_matrix'], dtype=np.float64)
    list_path = os.path.join(root_path, 'train_slices.list')
    sample_list = _read_nonempty_lines(list_path)
    labeled_names = set(sample_list[:LABELED_SLICES])
    groups = _case_groups(sample_list)

    case_summaries = []
    labeled_summaries = []
    all_parent_centers = []
    all_calibrated_centers = []
    for case_name, entries in groups.items():
        images = _load_case_images(root_path, entries, output_size)
        names = [name for _, name in entries]
        labeled_flags = [name in labeled_names for name in names]
        if any(labeled_flags) and not all(labeled_flags):
            raise RuntimeError(
                'ARCP labeled boundary crosses case {}'.format(case_name))
        labels = _load_case_labels(
            root_path, entries, output_size) if all(labeled_flags) else None

        profile_parent_effects = {profile: [] for profile in PROFILE_GRID}
        profile_arcp_effects = {profile: [] for profile in PROFILE_GRID}
        profile_axial_response = {profile: [] for profile in PROFILE_GRID}
        alpha_values = []
        active_values = []
        lower_values = []
        upper_values = []
        eligible_values = []
        occupancy_records = []
        third_parent_residual = {key: [] for key in ('first', 'middle', 'last')}
        third_arcp_residual = {key: [] for key in ('first', 'middle', 'last')}
        parent_fractional_pixels = 0
        arcp_fractional_pixels = 0

        for index in range(1, images.shape[0] - 1):
            stack = images[index - 1:index + 2]
            matrix = _numpy_response_gram(stack)
            if matrix is None:
                continue
            axial_response = math.sqrt(max(float(np.trace(matrix)), 0.0))
            for profile in PROFILE_GRID:
                weights = _profile_weights(*profile)
                calibrated, metadata = calibrate_profile_weights_numpy(
                    stack, weights, reference)
                profile_parent_effects[profile].append(
                    metadata['effect_before'])
                profile_arcp_effects[profile].append(
                    metadata['effect_after'])
                profile_axial_response[profile].append(axial_response)
                alpha_values.append(metadata['alpha'])
                eligible_values.append(float(metadata['eligible']))
                active_values.append(float(
                    metadata['eligible'] and
                    abs(metadata['alpha'] - 1.0) >= ACTIVITY_MARGIN))
                lower_values.append(float(metadata['at_lower_bound']))
                upper_values.append(float(metadata['at_upper_bound']))
                all_parent_centers.append(float(weights[1]))
                all_calibrated_centers.append(float(calibrated[1]))

                if labels is not None:
                    mask_stack = (labels[index - 1:index + 2] > 0).astype(
                        np.float64)
                    parent_occupancy = np.tensordot(
                        weights, mask_stack, axes=(0, 0))
                    arcp_occupancy = np.tensordot(
                        calibrated, mask_stack, axes=(0, 0))
                    center_mask = mask_stack[1]
                    parent_residual = float(np.mean(
                        np.abs(parent_occupancy - center_mask)))
                    arcp_residual = float(np.mean(
                        np.abs(arcp_occupancy - center_mask)))
                    parent_fractional_pixels += int(np.logical_and(
                        parent_occupancy > 1e-7,
                        parent_occupancy < 1.0 - 1e-7).sum())
                    arcp_fractional_pixels += int(np.logical_and(
                        arcp_occupancy > 1e-7,
                        arcp_occupancy < 1.0 - 1e-7).sum())
                    axial_third = _third(index, images.shape[0])
                    third_parent_residual[axial_third].append(parent_residual)
                    third_arcp_residual[axial_third].append(arcp_residual)
                    occupancy_records.append({
                        'parent_image': metadata['effect_before'],
                        'arcp_image': metadata['effect_after'],
                        'parent_occupancy': parent_residual,
                        'arcp_occupancy': arcp_residual,
                    })

        parent_cvs = []
        arcp_cvs = []
        parent_correlations = []
        arcp_correlations = []
        for profile in PROFILE_GRID:
            parent_cvs.append(_safe_cv(profile_parent_effects[profile]))
            arcp_cvs.append(_safe_cv(profile_arcp_effects[profile]))
            parent_correlations.append(_safe_corr(
                profile_parent_effects[profile],
                profile_axial_response[profile]))
            arcp_correlations.append(_safe_corr(
                profile_arcp_effects[profile],
                profile_axial_response[profile]))
        case_summary = {
            'case': case_name,
            'is_labeled_case': labels is not None,
            'eligible_fraction': _mean_finite(eligible_values),
            'active_fraction': _mean_finite(active_values),
            'lower_bound_fraction': _mean_finite(lower_values),
            'upper_bound_fraction': _mean_finite(upper_values),
            'alpha_mean': _mean_finite(alpha_values),
            'parent_cv_median': _median_finite(parent_cvs),
            'arcp_cv_median': _median_finite(arcp_cvs),
            'parent_abs_axial_correlation_median': _median_finite([
                abs(value) if value is not None else None
                for value in parent_correlations]),
            'arcp_abs_axial_correlation_median': _median_finite([
                abs(value) if value is not None else None
                for value in arcp_correlations]),
        }
        case_summaries.append(case_summary)

        if labels is not None:
            parent_image = [item['parent_image'] for item in occupancy_records]
            arcp_image = [item['arcp_image'] for item in occupancy_records]
            parent_occ = [item['parent_occupancy'] for item in occupancy_records]
            arcp_occ = [item['arcp_occupancy'] for item in occupancy_records]
            third_retention = {}
            for key in ('first', 'middle', 'last'):
                parent_mean = _mean_finite(third_parent_residual[key])
                arcp_mean = _mean_finite(third_arcp_residual[key])
                third_retention[key] = (
                    arcp_mean / parent_mean if parent_mean and
                    parent_mean > 1e-12 else None)
            labeled_summaries.append({
                'case': case_name,
                'fractional_support_retention': (
                    arcp_fractional_pixels / parent_fractional_pixels
                    if parent_fractional_pixels else None),
                'third_occupancy_residual_retention': third_retention,
                'parent_image_occupancy_spearman': _safe_corr(
                    parent_image, parent_occ, rank=True),
                'arcp_image_occupancy_spearman': _safe_corr(
                    arcp_image, arcp_occ, rank=True),
            })

    parent_cv = _median_finite([
        item['parent_cv_median'] for item in case_summaries])
    arcp_cv = _median_finite([
        item['arcp_cv_median'] for item in case_summaries])
    parent_corr = _median_finite([
        item['parent_abs_axial_correlation_median']
        for item in case_summaries])
    arcp_corr = _median_finite([
        item['arcp_abs_axial_correlation_median']
        for item in case_summaries])
    fractional_retention = _mean_finite([
        item['fractional_support_retention'] for item in labeled_summaries])
    first_retention = _mean_finite([
        item['third_occupancy_residual_retention']['first']
        for item in labeled_summaries])
    last_retention = _mean_finite([
        item['third_occupancy_residual_retention']['last']
        for item in labeled_summaries])
    parent_spearman = _median_finite([
        item['parent_image_occupancy_spearman']
        for item in labeled_summaries])
    arcp_spearman = _median_finite([
        item['arcp_image_occupancy_spearman']
        for item in labeled_summaries])
    summary = {
        'patient_balanced_active_fraction': _mean_finite([
            item['active_fraction'] for item in case_summaries]),
        'patient_balanced_lower_bound_fraction': _mean_finite([
            item['lower_bound_fraction'] for item in case_summaries]),
        'patient_balanced_upper_bound_fraction': _mean_finite([
            item['upper_bound_fraction'] for item in case_summaries]),
        'parent_cv_median': parent_cv,
        'arcp_cv_median': arcp_cv,
        'cv_reduction': (
            1.0 - arcp_cv / parent_cv if parent_cv and parent_cv > 0 else None),
        'parent_abs_axial_correlation_median': parent_corr,
        'arcp_abs_axial_correlation_median': arcp_corr,
        'abs_axial_correlation_reduction': (
            1.0 - arcp_corr / parent_corr
            if parent_corr and parent_corr > 0 else None),
        'labeled_fractional_support_retention': fractional_retention,
        'labeled_first_third_residual_retention': first_retention,
        'labeled_last_third_residual_retention': last_retention,
        'parent_image_occupancy_spearman_median': parent_spearman,
        'arcp_image_occupancy_spearman_median': arcp_spearman,
        'mean_center_weight_shift': abs(
            float(np.mean(all_calibrated_centers)) -
            float(np.mean(all_parent_centers))),
    }
    conditions = {
        'active_fraction_at_least_050':
            _at_least(summary['patient_balanced_active_fraction'], 0.50),
        'lower_bound_fraction_at_most_025':
            _at_most(summary['patient_balanced_lower_bound_fraction'], 0.25),
        'upper_bound_fraction_at_most_025':
            _at_most(summary['patient_balanced_upper_bound_fraction'], 0.25),
        'cv_reduction_at_least_020':
            _at_least(summary['cv_reduction'], 0.20),
        'axial_correlation_reduction_at_least_030':
            _at_least(summary['abs_axial_correlation_reduction'], 0.30),
        'fractional_support_retention_at_least_090':
            _at_least(fractional_retention, 0.90),
        'first_third_residual_retention_at_least_090':
            _at_least(first_retention, 0.90),
        'last_third_residual_retention_at_least_090':
            _at_least(last_retention, 0.90),
        'image_occupancy_spearman_not_decreased':
            parent_spearman is not None and arcp_spearman is not None and
            arcp_spearman >= parent_spearman,
        'mean_center_weight_shift_below_003':
            _at_most(summary['mean_center_weight_shift'], 0.03),
    }
    return {
        'analysis': 'H7.18 ARCP H5-only zero-training gate',
        'data_contract': {
            'lists_read': ['train_slices.list'],
            'image_scope': 'all training H5 images',
            'label_scope': 'first 191 locked labeled training slices only',
            'validation_or_test_read': False,
            'profile_grid': [list(item) for item in PROFILE_GRID],
            'center_weight_bounds': [
                PARENT_CENTER_WEIGHT_MIN, PARENT_CENTER_WEIGHT_MAX],
        },
        'reference': reference_report,
        'summary': summary,
        'conditions': conditions,
        'decision': 'pass' if all(conditions.values()) else 'fail',
        'cases': case_summaries,
        'labeled_cases': labeled_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', type=str,
        default=(
            '/home/aiteam/zhengtaoma/Baseline/data/'
            'PROMISE12_h5_training_source'))
    parser.add_argument('--output_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument(
        '--output_json', type=str,
        default=(
            '../model/SliceEqOccOAACStrongARCP_PROMISE12_7_labeled/'
            'analysis/h7_18_arcp_gate.json'))
    args = parser.parse_args()
    report = _analyze(args.root_path, tuple(args.output_size))
    save_reference_report(report, args.output_json)
    print(json.dumps({
        'decision': report['decision'],
        'summary': report['summary'],
        'conditions': report['conditions'],
        'output_json': os.path.abspath(args.output_json),
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
