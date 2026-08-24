"""Deterministically reaggregate the archived H7.10 patient/pair records.

This is a post-run audit, not a replacement for rerunning the final analyzer.
It uses only statistics already present in the immutable uploaded JSON.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


LOCKED_STEPS = (18000, 24000, 30000)
LOCKED_PATIENTS = (
    'Case04', 'Case08', 'Case15', 'Case23',
    'Case25', 'Case35', 'Case48')
ADU_KEYS = (
    'spearman_js_error', 'top20_error_ratio',
    'weighted_brier_reduction', 'fractional_support_mean_weight')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _median(values):
    if not values or not all(_finite(value) for value in values):
        raise ValueError('median input must be nonempty and finite')
    return statistics.median(values)


def _adu_pair_quality(pair):
    return (
        pair.get('eligible_pixels', 0) > 0 and
        pair.get('positive_js_fraction', 0.0) > 0.0 and
        pair.get('js_standard_deviation', 0.0) > 0.0 and
        pair.get('convexity_identity_max_abs_error', math.inf) <= 1e-6 and
        all(_finite(pair.get(key)) for key in ADU_KEYS))


def _adu_pass(metrics, thresholds):
    return (
        metrics['spearman_js_error'] >= thresholds['adu_min_spearman'] and
        metrics['top20_error_ratio'] >=
        thresholds['adu_min_top20_error_ratio'] and
        metrics['weighted_brier_reduction'] >=
        thresholds['adu_min_weighted_brier_reduction'] and
        metrics['fractional_support_mean_weight'] >=
        thresholds['adu_min_fractional_weight'])


def _sct_values(report):
    observed = report['observed']
    non_clamped = report['non_clamped']['observed']
    return {
        'residual_variance_reduction':
            observed['residual_variance_reduction'],
        'residual_brier_reduction': observed['residual_brier_reduction'],
        'full_brier_ratio': observed['full_brier_ratio'],
        'center_dice_delta': observed['center_dice_delta'],
        'non_clamped_residual_variance_reduction':
            non_clamped['residual_variance_reduction'],
        'non_clamped_residual_brier_reduction':
            non_clamped['residual_brier_reduction'],
        'non_clamped_full_brier_ratio':
            non_clamped['full_brier_ratio'],
        'non_clamped_center_dice_delta':
            non_clamped['center_dice_delta'],
    }


def _sct_pass(metrics, thresholds):
    for prefix in ('', 'non_clamped_'):
        if not (
                metrics[prefix + 'residual_variance_reduction'] >=
                thresholds['sct_min_residual_variance_reduction'] and
                metrics[prefix + 'residual_brier_reduction'] >=
                thresholds['sct_min_residual_brier_reduction'] and
                metrics[prefix + 'full_brier_ratio'] <=
                thresholds['sct_max_full_brier_ratio'] and
                metrics[prefix + 'center_dice_delta'] >=
                thresholds['sct_min_center_dice_delta']):
            return False
    return True


def reaggregate(report, source_path):
    checkpoints = report['checkpoints']
    if tuple(item['step'] for item in checkpoints) != LOCKED_STEPS:
        raise ValueError('expected checkpoints 18k/24k/30k in order')
    thresholds = report['thresholds']
    adu_by_patient = {patient: [] for patient in LOCKED_PATIENTS}
    sct_by_patient = {patient: [] for patient in LOCKED_PATIENTS}
    checkpoint_audit = []
    quality_pairs = 0

    for checkpoint in checkpoints:
        if tuple(checkpoint['adu']['by_patient']) != LOCKED_PATIENTS:
            raise ValueError('unexpected ADU patient set or order')
        if tuple(checkpoint['sct']['by_patient']) != LOCKED_PATIENTS:
            raise ValueError('unexpected SCT patient set or order')
        adu_passes = 0
        sct_passes = 0
        for patient in LOCKED_PATIENTS:
            pairs = checkpoint['adu']['by_patient'][patient]['pairs']
            complete = len(pairs) == 4 and all(
                _adu_pair_quality(pair) for pair in pairs)
            if not complete:
                raise ValueError(
                    '{} {} lacks four complete ADU pairs'.format(
                        checkpoint['step'], patient))
            quality_pairs += len(pairs)
            adu_metrics = {
                key: _median([pair[key] for pair in pairs])
                for key in ADU_KEYS
            }
            adu_metrics['quality_complete'] = True
            adu_metrics['pass'] = _adu_pass(adu_metrics, thresholds)
            adu_passes += int(adu_metrics['pass'])
            adu_by_patient[patient].append(adu_metrics)

            sct_metrics = _sct_values(
                checkpoint['sct']['by_patient'][patient])
            if not all(_finite(value) for value in sct_metrics.values()):
                raise ValueError(
                    '{} {} has nonfinite SCT metrics'.format(
                        checkpoint['step'], patient))
            sct_metrics['pass'] = _sct_pass(sct_metrics, thresholds)
            sct_passes += int(sct_metrics['pass'])
            sct_by_patient[patient].append(sct_metrics)
        checkpoint_audit.append({
            'step': checkpoint['step'],
            'adu_patient_passes': adu_passes,
            'sct_patient_passes': sct_passes,
        })

    patient_audit = {}
    adu_named_passes = 0
    sct_named_passes = 0
    for patient in LOCKED_PATIENTS:
        adu_metrics = {
            key: _median([item[key] for item in adu_by_patient[patient]])
            for key in ADU_KEYS
        }
        adu_metrics['quality_complete_checkpoints'] = sum(
            item['quality_complete'] for item in adu_by_patient[patient])
        adu_metrics['pass'] = (
            adu_metrics['quality_complete_checkpoints'] == 3 and
            _adu_pass(adu_metrics, thresholds))
        adu_named_passes += int(adu_metrics['pass'])

        sct_keys = tuple(
            key for key in sct_by_patient[patient][0] if key != 'pass')
        sct_metrics = {
            key: _median([item[key] for item in sct_by_patient[patient]])
            for key in sct_keys
        }
        sct_metrics['pass'] = _sct_pass(sct_metrics, thresholds)
        sct_named_passes += int(sct_metrics['pass'])
        patient_audit[patient] = {'adu': adu_metrics, 'sct': sct_metrics}

    decision = (
        'authorize_exploratory_slice_eq_occ_adu_training'
        if adu_named_passes >= 5 and sct_named_passes < 5
        else 'no_exploratory_authorization')
    return {
        'schema_version': 1,
        'classification': 'post_run_conservative_reaggregation',
        'not_a_final_hash_confirmatory_rerun': True,
        'source_artifact': source_path.name,
        'source_sha256': _sha256(source_path),
        'script_sha256': _sha256(Path(__file__).resolve()),
        'locked_steps': list(LOCKED_STEPS),
        'locked_patients': list(LOCKED_PATIENTS),
        'quality_complete_adu_pairs': quality_pairs,
        'checkpoint_audit': checkpoint_audit,
        'patient_audit': patient_audit,
        'adu_named_patient_passes': adu_named_passes,
        'sct_named_patient_passes': sct_named_passes,
        'decision': decision,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_json', type=Path)
    args = parser.parse_args()
    source = args.source_json.resolve()
    report = json.loads(source.read_text(encoding='utf-8'))
    derived = reaggregate(report, source)
    print(json.dumps(
        derived, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False))


if __name__ == '__main__':
    main()
