"""Run the preregistered H7.4 posterior-commutation fidelity gate.

This entry is read-only: it performs frozen inference and target analysis, has
no parameter updates, and writes only one atomic JSON report.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import analyze_sliceeq_gates as h73
from dataloaders.sliceeq_dataset import (
    SliceStackDataSets, StackRandomGenerator)
from utils.sliceeq import paired_slice_reacquisition
from utils.sliceeq_gates import (
    PearsonAccumulator, acquisition_residual)
from utils.sliceeq_posterior import (
    distribution_residual, occupancy_brier_map,
    profile_weighted_distribution, topology_gate_binary_posterior)


DEFAULT_H7_3_REFERENCE = (
    '../model/SliceEqOcc_PROMISE12_7_labeled/analysis/'
    'h7_3_gates_iter23000.json')
DEFAULT_OUTPUT = (
    '../model/SliceEqOcc_PROMISE12_7_labeled/analysis/'
    'h7_4_posterior_commutation_gate_iter23000.json')

RESIDUAL_EPSILON = 1e-6
MAX_EXACT_SUPPORT_BRIER_RATIO = 0.85
MIN_RESIDUAL_PEARSON = 0.50
MAX_OUTSIDE_RESIDUAL_MASS = 0.15
MAX_FULL_IMAGE_BRIER_RATIO = 1.05
REFERENCE_REPRODUCTION_ATOL = 1e-6


def build_parser():
    parser = argparse.ArgumentParser(
        description='Read-only SliceEq H7.4 posterior-commutation gate')
    parser.add_argument('--root_path', type=str, default=h73.DEFAULT_ROOT)
    parser.add_argument('--checkpoint', type=str, default=h73.DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--h7_3_reference_json', type=str, default=DEFAULT_H7_3_REFERENCE)
    parser.add_argument('--output_json', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_classes', type=int, default=2, choices=[2])
    parser.add_argument('--labeled_slices', type=int, default=191)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_labeled_batches', type=int, default=16)
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--sliceeq_radius', type=int, default=1, choices=[1])
    parser.add_argument('--sliceeq_sigma_min', type=float, default=0.45)
    parser.add_argument('--sliceeq_sigma_max', type=float, default=0.85)
    parser.add_argument('--sliceeq_phase_min', type=float, default=-0.25)
    parser.add_argument('--sliceeq_phase_max', type=float, default=0.25)
    return parser


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


class CandidateAccumulator:
    """Aggregate target fidelity without retaining patient-level tensors."""

    def __init__(self):
        self.pixel_correlation = PearsonAccumulator()
        self.mass_correlation = PearsonAccumulator()
        self.samples = 0
        self.total_pixels = 0
        self.exact_support_pixels = 0
        self.candidate_support_pixels = 0
        self.union_support_pixels = 0
        self.intersection_support_pixels = 0
        self.exact_support_brier_sum = 0.0
        self.full_brier_sum = 0.0
        self.candidate_residual_mass = 0.0
        self.outside_residual_mass = 0.0
        self.exact_foreground_mass = 0.0
        self.candidate_foreground_mass = 0.0
        self.per_sample_absolute_foreground_bias = []

    def update(self, candidate_occupancy, candidate_center,
               exact_occupancy, exact_center_target):
        exact_residual = acquisition_residual(
            exact_occupancy, exact_center_target)
        candidate_residual = distribution_residual(
            candidate_occupancy, candidate_center)
        exact_support = exact_residual > RESIDUAL_EPSILON
        candidate_support = candidate_residual > RESIDUAL_EPSILON
        union = exact_support | candidate_support
        intersection = exact_support & candidate_support

        self.pixel_correlation.update(
            exact_residual[union], candidate_residual[union])
        exact_mass = exact_residual.flatten(1).sum(dim=1)
        candidate_mass = candidate_residual.flatten(1).sum(dim=1)
        active_mass = (exact_mass > RESIDUAL_EPSILON) | \
            (candidate_mass > RESIDUAL_EPSILON)
        self.mass_correlation.update(
            exact_mass[active_mass], candidate_mass[active_mass])

        brier = occupancy_brier_map(candidate_occupancy, exact_occupancy)
        exact_foreground = exact_occupancy[:, 1].flatten(1).sum(dim=1)
        candidate_foreground = candidate_occupancy[:, 1].flatten(1).sum(dim=1)
        foreground_denominator = exact_foreground.clamp_min(1.0)
        absolute_bias = (
            candidate_foreground - exact_foreground).abs() / \
            foreground_denominator

        self.samples += int(candidate_occupancy.shape[0])
        self.total_pixels += int(brier.numel())
        self.exact_support_pixels += int(exact_support.sum().item())
        self.candidate_support_pixels += int(candidate_support.sum().item())
        self.union_support_pixels += int(union.sum().item())
        self.intersection_support_pixels += int(intersection.sum().item())
        self.exact_support_brier_sum += float(
            brier[exact_support].sum().item())
        self.full_brier_sum += float(brier.sum().item())
        self.candidate_residual_mass += float(
            candidate_residual.sum().item())
        self.outside_residual_mass += float(
            candidate_residual[~exact_support].sum().item())
        self.exact_foreground_mass += float(exact_foreground.sum().item())
        self.candidate_foreground_mass += float(
            candidate_foreground.sum().item())
        self.per_sample_absolute_foreground_bias.extend(
            float(value) for value in absolute_bias.detach().cpu().tolist())

    def report(self):
        signed_foreground_bias = _safe_ratio(
            self.candidate_foreground_mass - self.exact_foreground_mass,
            self.exact_foreground_mass)
        return {
            'samples': self.samples,
            'total_pixels': self.total_pixels,
            'exact_support_pixels': self.exact_support_pixels,
            'candidate_support_pixels': self.candidate_support_pixels,
            'union_support_pixels': self.union_support_pixels,
            'intersection_support_pixels': self.intersection_support_pixels,
            'support_precision': _safe_ratio(
                self.intersection_support_pixels,
                self.candidate_support_pixels),
            'support_recall': _safe_ratio(
                self.intersection_support_pixels,
                self.exact_support_pixels),
            'support_iou': _safe_ratio(
                self.intersection_support_pixels,
                self.union_support_pixels),
            'exact_support_occupancy_brier_mse': _safe_ratio(
                self.exact_support_brier_sum,
                self.exact_support_pixels),
            'full_image_occupancy_brier_mse': _safe_ratio(
                self.full_brier_sum, self.total_pixels),
            'union_support_residual_pearson':
                self.pixel_correlation.correlation(),
            'active_sample_residual_mass_pearson':
                self.mass_correlation.correlation(),
            'candidate_residual_mass': self.candidate_residual_mass,
            'outside_exact_support_residual_mass':
                self.outside_residual_mass,
            'outside_exact_support_residual_mass_fraction': _safe_ratio(
                self.outside_residual_mass,
                self.candidate_residual_mass),
            'exact_foreground_occupancy_mass': self.exact_foreground_mass,
            'candidate_foreground_occupancy_mass':
                self.candidate_foreground_mass,
            'signed_relative_foreground_occupancy_bias':
                signed_foreground_bias,
            'per_sample_absolute_relative_foreground_bias_mean': (
                statistics.fmean(self.per_sample_absolute_foreground_bias)
                if self.per_sample_absolute_foreground_bias else None),
        }


def _one_hot_stack(labels, num_classes, dtype):
    encoded = F.one_hot(labels.long(), num_classes=num_classes)
    return encoded.permute(0, 1, 4, 2, 3).to(dtype=dtype)


def _posterior_and_lcc_stack(model, image_stack):
    batch_size, stack_size, channels, height, width = image_stack.shape
    flat = image_stack.reshape(
        batch_size * stack_size, channels, height, width)
    with torch.no_grad():
        logits = h73._logits(model(flat))
        posterior = torch.softmax(logits, dim=1)
        hard = h73._largest_component(posterior.argmax(dim=1))
    posterior = posterior.reshape(
        batch_size, stack_size, posterior.shape[1], height, width)
    hard = hard.reshape(batch_size, stack_size, height, width)
    return posterior, hard


def _update_strata(accumulators, candidate_occupancy, candidate_center,
                   exact_occupancy, exact_center_target, clamped):
    accumulators['all'].update(
        candidate_occupancy, candidate_center,
        exact_occupancy, exact_center_target)
    clamped_mask = clamped.bool()
    for name, mask in (
            ('clamped', clamped_mask),
            ('non_clamped', ~clamped_mask)):
        if mask.any():
            accumulators[name].update(
                candidate_occupancy[mask], candidate_center[mask],
                exact_occupancy[mask], exact_center_target[mask])


def _new_stratified_accumulators():
    return {
        'all': CandidateAccumulator(),
        'clamped': CandidateAccumulator(),
        'non_clamped': CandidateAccumulator(),
    }


def _run_analysis(args, model, loader, device):
    variants = {
        'hard_lcc': _new_stratified_accumulators(),
        'raw_posterior': _new_stratified_accumulators(),
        'topology_gated_posterior': _new_stratified_accumulators(),
    }
    h73_fidelity = h73.FidelityGateAccumulator()
    generator = h73._profile_generator(device, args.seed + 1)
    seen_samples = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_labeled_batches:
            break
        image_stack = batch['image_stack'].to(device, non_blocking=True)
        label_stack = batch['label_stack'].to(device, non_blocking=True)
        clamped = batch['neighbor_clamped'].to(device)
        center = image_stack.shape[1] // 2
        weights = h73._sample_weights(
            args, image_stack.shape[0], device, generator)

        with torch.no_grad():
            _, _, exact_occupancy = paired_slice_reacquisition(
                image_stack, label_stack, weights, args.num_classes)
            posterior_stack, hard_stack = _posterior_and_lcc_stack(
                model, image_stack)
            _, _, hard_occupancy = paired_slice_reacquisition(
                image_stack, hard_stack, weights, args.num_classes)
            raw_occupancy = profile_weighted_distribution(
                posterior_stack, weights)
            gated_stack = topology_gate_binary_posterior(
                posterior_stack, hard_stack)
            gated_occupancy = profile_weighted_distribution(
                gated_stack, weights)

            exact_center = label_stack[:, center]
            hard_center_distribution = _one_hot_stack(
                hard_stack, args.num_classes,
                exact_occupancy.dtype)[:, center]
            raw_center_distribution = posterior_stack[:, center]
            gated_center_distribution = gated_stack[:, center]

            exact_residual = acquisition_residual(
                exact_occupancy, exact_center)
            hard_residual = distribution_residual(
                hard_occupancy, hard_center_distribution)
            h73_fidelity.update(exact_residual, hard_residual)

            _update_strata(
                variants['hard_lcc'], hard_occupancy,
                hard_center_distribution, exact_occupancy,
                exact_center, clamped)
            _update_strata(
                variants['raw_posterior'], raw_occupancy,
                raw_center_distribution, exact_occupancy,
                exact_center, clamped)
            _update_strata(
                variants['topology_gated_posterior'], gated_occupancy,
                gated_center_distribution, exact_occupancy,
                exact_center, clamped)

        seen_samples += int(image_stack.shape[0])
        print(
            '[H7.4 labeled] batch {}/{} samples={}'.format(
                batch_index + 1,
                min(len(loader), args.max_labeled_batches),
                seen_samples), flush=True)

    reports = {
        variant_name: {
            stratum: accumulator.report()
            for stratum, accumulator in stratified.items()
        }
        for variant_name, stratified in variants.items()
    }
    return reports, h73_fidelity.report()


def _load_reference(path):
    with open(path, 'r', encoding='utf-8') as stream:
        report = json.load(stream)
    if report.get('schema_version') != 1:
        raise ValueError('unsupported H7.3 reference schema')
    if report.get('decisions', {}).get('joint_decision') != 'stop_h7_3':
        raise ValueError('H7.3 reference does not record stop_h7_3')
    return report


def _reference_reproduction(reference, observed, checkpoint_path):
    expected = reference['gate2_frozen_student_proxy_fidelity']
    comparisons = {
        'samples': (expected['samples'], observed['samples']),
        'union_support_pixels': (
            expected['union_support_pixels'],
            observed['union_support_pixels']),
        'union_support_pixel_pearson': (
            expected['union_support_pixel_pearson'],
            observed['union_support_pixel_pearson']),
        'per_sample_residual_mass_pearson': (
            expected['per_sample_residual_mass_pearson'],
            observed['per_sample_residual_mass_pearson']),
        'support_iou': (
            expected['support_iou'], observed['support_iou']),
        'outside_mass_fraction': (
            expected['proxy_residual_mass_outside_exact_fraction'],
            observed['proxy_residual_mass_outside_exact_fraction']),
    }
    mismatches = {}
    for name, (expected_value, observed_value) in comparisons.items():
        if isinstance(expected_value, int):
            matched = int(observed_value) == expected_value
        else:
            matched = math.isclose(
                float(observed_value), float(expected_value),
                abs_tol=REFERENCE_REPRODUCTION_ATOL, rel_tol=0.0)
        if not matched:
            mismatches[name] = {
                'expected': expected_value,
                'observed': observed_value,
            }

    expected_checkpoint_hash = reference['provenance']['file_sha256'][
        reference['provenance']['arguments']['checkpoint']]
    observed_checkpoint_hash = h73._sha256(checkpoint_path)
    if observed_checkpoint_hash != expected_checkpoint_hash:
        mismatches['checkpoint_sha256'] = {
            'expected': expected_checkpoint_hash,
            'observed': observed_checkpoint_hash,
        }
    result = {
        'pass': not mismatches,
        'absolute_tolerance': REFERENCE_REPRODUCTION_ATOL,
        'comparisons': {
            name: {'expected': values[0], 'observed': values[1]}
            for name, values in comparisons.items()
        },
        'checkpoint_sha256': observed_checkpoint_hash,
        'mismatches': mismatches,
    }
    if mismatches:
        raise RuntimeError(
            'H7.3 hard-LCC reference reproduction failed: {}'.format(
                mismatches))
    return result


def _candidate_decision(candidate, hard):
    exact_brier_ratio = _safe_ratio(
        candidate['exact_support_occupancy_brier_mse'],
        hard['exact_support_occupancy_brier_mse'])
    full_brier_ratio = _safe_ratio(
        candidate['full_image_occupancy_brier_mse'],
        hard['full_image_occupancy_brier_mse'])
    pearson = candidate['union_support_residual_pearson']
    outside = candidate[
        'outside_exact_support_residual_mass_fraction']
    passed = all(value is not None for value in (
        exact_brier_ratio, full_brier_ratio, pearson, outside)) and \
        exact_brier_ratio <= MAX_EXACT_SUPPORT_BRIER_RATIO and \
        pearson >= MIN_RESIDUAL_PEARSON and \
        outside <= MAX_OUTSIDE_RESIDUAL_MASS and \
        full_brier_ratio <= MAX_FULL_IMAGE_BRIER_RATIO
    return {
        'pass': passed,
        'observed': {
            'exact_support_brier_ratio_vs_hard': exact_brier_ratio,
            'residual_pearson': pearson,
            'outside_residual_mass_fraction': outside,
            'full_image_brier_ratio_vs_hard': full_brier_ratio,
        },
        'thresholds': {
            'maximum_exact_support_brier_ratio':
                MAX_EXACT_SUPPORT_BRIER_RATIO,
            'minimum_residual_pearson': MIN_RESIDUAL_PEARSON,
            'maximum_outside_residual_mass_fraction':
                MAX_OUTSIDE_RESIDUAL_MASS,
            'maximum_full_image_brier_ratio':
                MAX_FULL_IMAGE_BRIER_RATIO,
        },
    }


def _joint_decision(reports):
    hard = reports['hard_lcc']['all']
    decisions = {
        name: _candidate_decision(reports[name]['all'], hard)
        for name in ('raw_posterior', 'topology_gated_posterior')
    }
    passing = [
        name for name, decision in decisions.items()
        if decision['pass']]
    if passing:
        selected = min(
            passing,
            key=lambda name: reports[name]['all'][
                'exact_support_occupancy_brier_mse'])
        joint = 'authorize_h7_4_training'
    else:
        selected = None
        joint = 'stop_posterior_commutation'
    return {
        'candidates': decisions,
        'selected_candidate': selected,
        'joint_decision': joint,
    }


def _validate_args(args, dataset_length):
    for name, path in (
            ('checkpoint', args.checkpoint),
            ('H7.3 reference JSON', args.h7_3_reference_json)):
        if not os.path.isfile(path):
            raise FileNotFoundError('{} does not exist: {}'.format(name, path))
    if args.labeled_slices <= 0 or args.labeled_slices > dataset_length:
        raise ValueError('labeled_slices is outside the dataset')
    if args.batch_size <= 0 or args.max_labeled_batches <= 0:
        raise ValueError('batch and maximum batch counts must be positive')


def _provenance(args, checkpoint_format, dataset_length, device):
    code_root = Path(__file__).resolve().parent
    workspace_root = code_root.parent
    tracked_files = [
        Path(__file__).resolve(),
        code_root / 'utils' / 'sliceeq_posterior.py',
        code_root / 'analyze_sliceeq_gates.py',
        code_root / 'utils' / 'sliceeq_gates.py',
        code_root / 'utils' / 'sliceeq.py',
        code_root / 'dataloaders' / 'sliceeq_dataset.py',
        workspace_root / 'research' / 'experiments' /
        'h7_slice_profile_reacquisition' /
        'h7_4_posterior_commutation_gate_protocol.md',
        Path(args.root_path) / 'train_slices.list',
        Path(args.checkpoint),
        Path(args.h7_3_reference_json),
    ]
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_kind': 'read_only_no_training',
        'gate2_evidence_source': 'frozen_student_proxy_in_eval_mode',
        'checkpoint_contains_ema_teacher': False,
        'checkpoint_format': checkpoint_format,
        'dataset_length': dataset_length,
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'arguments': vars(args),
        'file_sha256': {
            str(path.resolve()): h73._sha256(path.resolve())
            for path in tracked_files
        },
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = h73._resolve_device(args.device)

    dataset = SliceStackDataSets(
        base_dir=args.root_path,
        transform=StackRandomGenerator(args.patch_size),
        radius=args.sliceeq_radius)
    _validate_args(args, len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(args.labeled_slices))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=False, drop_last=False)
    reference = _load_reference(args.h7_3_reference_json)
    model, checkpoint_format = h73._load_model(args.checkpoint, device)

    print(
        'Running H7.4 posterior-commutation gate on frozen checkpoint: {}'.format(
            args.checkpoint), flush=True)
    reports, observed_reference = _run_analysis(
        args, model, loader, device)
    reproduction = _reference_reproduction(
        reference, observed_reference, args.checkpoint)
    decisions = _joint_decision(reports)
    report = {
        'schema_version': 1,
        'provenance': _provenance(
            args, checkpoint_format, len(dataset), device),
        'locked_definitions': {
            'occupancy_brier_mse':
                'mean over classes of squared occupancy error per pixel',
            'exact_residual':
                '0.5 * L1(A(one_hot(y)), one_hot(y_center))',
            'candidate_residual':
                '0.5 * L1(A(candidate_distribution), candidate_center)',
            'support_epsilon': RESIDUAL_EPSILON,
        },
        'h7_3_hard_lcc_reference_reproduction': reproduction,
        'variants': reports,
        'decisions': decisions,
        'interpretation_warning': (
            'The iteration-23000 checkpoint has no EMA state. All posterior '
            'targets in this gate use the frozen student in eval mode as a '
            'proxy; a pass is conditional training authorization, not a '
            'claimed performance improvement.'),
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary_path, output_path)
    print(json.dumps(decisions, ensure_ascii=False, indent=2), flush=True)
    print('Wrote H7.4 gate report: {}'.format(output_path), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
