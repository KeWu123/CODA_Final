"""Run the preregistered SliceEq H7.10 operator-reliability gate.

This entry performs frozen, train-mode proxy inference on labeled training
stacks. It does not optimize parameters, write checkpoints, or inspect the
official validation/test sets.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_sliceeq_gates as h73
from dataloaders.sliceeq_dataset import (
    SliceStackDataSets, StackRandomGenerator, parse_slice_name)
from networks.unet import UNet
from utils.sliceeq import (
    paired_slice_reacquisition, sample_slice_profiles)
from utils.sliceeq_reliability import (
    INDEPENDENT_DROPOUT,
    STACK_SHARED_DROPOUT,
    binary_dice_per_sample,
    jensen_shannon_map,
    occupancy_brier_map,
    reliability_from_js,
    restore_buffers,
    snapshot_buffers,
    soft_dice_error_per_sample,
    spearman_correlation,
    temporary_stack_dropout,
    top_fraction_error_ratio,
)


DEFAULT_ROOT = h73.DEFAULT_ROOT
DEFAULT_CHECKPOINT_DIR = (
    '../model/SliceEqOcc_PROMISE12_7_labeled/self_train/unet')
DEFAULT_CHECKPOINT_STEPS = [18000, 24000, 30000]
DEFAULT_OUTPUT = (
    '../model/SliceEqOcc_PROMISE12_7_labeled/analysis/'
    'h7_10_operator_reliability_gate.json')
DEFAULT_BATCH_SCHEDULE_SEEDS = [1337, 7331]

EXPECTED_TRAIN_CASES = 35
EXPECTED_TRAIN_SLICES = 940
LOCKED_LABELED_SLICES = 191
LOCKED_LABELNUM = 7
LOCKED_BATCH_SIZE = 12
LOCKED_MC_DRAWS = 8
STACK_SIZE = 3
RESIDUAL_EPSILON = 1e-6
JS_EPSILON = 1e-7

SCT_MIN_RESIDUAL_VARIANCE_REDUCTION = 0.15
SCT_MIN_RESIDUAL_BRIER_REDUCTION = 0.05
SCT_MAX_FULL_BRIER_RATIO = 1.01
SCT_MIN_CENTER_DICE_DELTA = -0.002
ADU_MIN_SPEARMAN = 0.25
ADU_MIN_TOP20_ERROR_RATIO = 1.50
ADU_MIN_WEIGHTED_BRIER_REDUCTION = 0.05
ADU_MIN_FRACTIONAL_WEIGHT = 0.90
MIN_PATIENT_PASSES = 5
MIN_CHECKPOINT_PASSES = 2
ADU_SELECTION_MARGIN = 0.02
CONVEXITY_ATOL = 1e-6


def build_parser():
    parser = argparse.ArgumentParser(
        description='Read-only SliceEq H7.10 SCT/ADU reliability gate')
    parser.add_argument('--root_path', type=str, default=DEFAULT_ROOT)
    parser.add_argument(
        '--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--checkpoints', type=str, nargs=3, default=None)
    parser.add_argument(
        '--checkpoint_steps', type=int, nargs=3,
        default=DEFAULT_CHECKPOINT_STEPS)
    parser.add_argument('--output_json', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_schedule_seeds', type=int, nargs=2,
                        default=DEFAULT_BATCH_SCHEDULE_SEEDS)
    parser.add_argument('--num_classes', type=int, default=2, choices=[2])
    parser.add_argument(
        '--labeled_slices', type=int, default=LOCKED_LABELED_SLICES)
    parser.add_argument('--labelnum', type=int, default=LOCKED_LABELNUM)
    parser.add_argument('--batch_size', type=int, default=LOCKED_BATCH_SIZE)
    parser.add_argument('--mc_draws', type=int, default=LOCKED_MC_DRAWS)
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--sliceeq_radius', type=int, default=1, choices=[1])
    parser.add_argument('--sliceeq_sigma_min', type=float, default=0.45)
    parser.add_argument('--sliceeq_sigma_max', type=float, default=0.85)
    parser.add_argument('--sliceeq_phase_min', type=float, default=-0.25)
    parser.add_argument('--sliceeq_phase_max', type=float, default=0.25)
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_nonempty_lines(path):
    with open(path, 'r', encoding='utf-8-sig') as stream:
        return [line.strip() for line in stream if line.strip()]


def _ordered_unique(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _labeled_only_preflight(root_path, labeled_slices, labelnum):
    """Validate the labeled prefix without touching val/test contracts."""
    train_path = os.path.join(root_path, 'train.list')
    slices_path = os.path.join(root_path, 'train_slices.list')
    for path in (train_path, slices_path):
        if not os.path.isfile(path):
            raise FileNotFoundError('required training list is absent: {}'.format(
                path))
    train_cases = _read_nonempty_lines(train_path)
    train_slices = _read_nonempty_lines(slices_path)
    if len(train_cases) != EXPECTED_TRAIN_CASES:
        raise ValueError(
            'expected {} train cases, found {}'.format(
                EXPECTED_TRAIN_CASES, len(train_cases)))
    if len(train_slices) != EXPECTED_TRAIN_SLICES:
        raise ValueError(
            'expected {} train slices, found {}'.format(
                EXPECTED_TRAIN_SLICES, len(train_slices)))
    selected = train_slices[:labeled_slices]
    later = train_slices[labeled_slices:]
    selected_cases = [
        parse_slice_name(sample_name)[0] for sample_name in selected]
    selected_case_order = _ordered_unique(selected_cases)
    if selected_case_order != train_cases[:labelnum]:
        raise ValueError(
            'the labeled slice prefix does not match the first train cases')
    if len(set(selected_case_order)) != labelnum:
        raise ValueError('the labeled prefix does not contain labelnum cases')
    leaked = sorted(
        set(selected_case_order) &
        {parse_slice_name(name)[0] for name in later})
    if leaked:
        raise ValueError(
            'labeled patient slices cross the prefix boundary: {}'.format(
                leaked))

    patient_slice_counts = {
        patient: selected_cases.count(patient)
        for patient in selected_case_order
    }
    case_bounds = {}
    for patient in selected_case_order:
        indices = [
            parse_slice_name(name)[1]
            for name in selected
            if parse_slice_name(name)[0] == patient
        ]
        case_bounds[patient] = [min(indices), max(indices)]
    return {
        'train_cases_path': str(Path(train_path).resolve()),
        'train_slices_path': str(Path(slices_path).resolve()),
        'total_train_cases': len(train_cases),
        'total_train_slices': len(train_slices),
        'labeled_slices': labeled_slices,
        'labeled_patients': selected_case_order,
        'patient_slice_counts': patient_slice_counts,
        'case_bounds': case_bounds,
        'allowlist': selected,
    }


class LabeledOnlySliceStackDataSets(SliceStackDataSets):
    """Reject any H5 label read outside the preflighted labeled prefix."""

    def __init__(self, base_dir, transform, radius, num, allowlist):
        self._label_allowlist = set(allowlist)
        self.label_h5_reads = 0
        super().__init__(
            base_dir=base_dir, transform=transform, radius=radius, num=num)
        if set(self.sample_list) != self._label_allowlist:
            raise ValueError('dataset prefix differs from the label allowlist')
        for neighbors, _ in self.neighbor_table:
            if not set(neighbors).issubset(self._label_allowlist):
                raise ValueError('a labeled stack neighbor leaves the allowlist')

    def _read_slice(self, sample_name):
        if sample_name not in self._label_allowlist:
            raise RuntimeError(
                'blocked non-labeled H5 access: {}'.format(sample_name))
        path = os.path.join(
            self._base_dir, 'data', 'slices', sample_name + '.h5')
        with h5py.File(path, 'r') as stream:
            image = stream['image'][:]
            label = stream['label'][:]
        self.label_h5_reads += 1
        if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
            raise ValueError(
                'SliceEq expects matched 2D image/label arrays: {}'.format(
                    path))
        return image, label


def _set_host_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _preload_labeled_samples(args, preflight):
    _set_host_seed(args.seed)
    dataset = LabeledOnlySliceStackDataSets(
        base_dir=args.root_path,
        transform=StackRandomGenerator(args.patch_size),
        radius=args.sliceeq_radius,
        num=args.labeled_slices,
        allowlist=preflight['allowlist'])
    samples = []
    for index in range(len(dataset)):
        sample = dataset[index]
        patient, slice_index = parse_slice_name(sample['case'])
        samples.append({
            'image_stack': sample['image_stack'].contiguous(),
            'label_stack': sample['label_stack'].contiguous(),
            'sample_name': sample['case'],
            'patient': patient,
            'slice_index': slice_index,
            'neighbor_clamped': bool(
                sample['neighbor_clamped'].item() > 0.5),
        })
    if len(samples) != args.labeled_slices:
        raise RuntimeError('not all labeled samples were preloaded')
    return samples, dataset.label_h5_reads


def _profile_weights(args, device):
    offsets = tuple(
        range(-args.sliceeq_radius, args.sliceeq_radius + 1))
    generator_device = 'cuda' if device.type == 'cuda' else 'cpu'
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.seed + 1)
    weights, sigma, phase = sample_slice_profiles(
        args.labeled_slices,
        offsets,
        (args.sliceeq_sigma_min, args.sliceeq_sigma_max),
        (args.sliceeq_phase_min, args.sliceeq_phase_max),
        device=device,
        generator=generator)
    return weights.cpu(), {
        'sigma_mean': float(sigma.mean().item()),
        'absolute_phase_mean': float(phase.abs().mean().item()),
        'center_weight_mean': float(
            weights[:, weights.shape[1] // 2].mean().item()),
    }


def _resolve_checkpoints(args):
    if args.checkpoints is None:
        paths = [
            os.path.join(args.checkpoint_dir, 'iter_{}.pth'.format(step))
            for step in args.checkpoint_steps
        ]
    else:
        paths = list(args.checkpoints)
    if len(paths) != 3 or len(args.checkpoint_steps) != 3:
        raise ValueError('H7.10 requires exactly three checkpoints')
    resolved = [str(Path(path).expanduser().resolve()) for path in paths]
    if len(set(resolved)) != 3:
        raise ValueError('checkpoint paths must be unique')
    for path in resolved:
        if not os.path.isfile(path):
            raise FileNotFoundError('checkpoint does not exist: {}'.format(path))
    for step, path in zip(args.checkpoint_steps, resolved):
        stem = Path(path).stem
        expected = 'iter_{}'.format(step)
        if stem != expected and not stem.startswith(expected + '_dice_'):
            raise ValueError(
                'checkpoint basename does not match locked step {}: {}'.format(
                    step, path))
    return list(zip(args.checkpoint_steps, resolved))


def _load_train_mode_proxy(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and 'net' in payload and isinstance(
            payload['net'], dict):
        state_dict = payload['net']
        checkpoint_format = 'training_bundle.net'
    elif isinstance(payload, dict):
        state_dict = payload
        checkpoint_format = 'plain_state_dict'
    else:
        raise TypeError('checkpoint must contain a PyTorch state_dict')
    model = UNet(in_chns=1, class_num=2).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.train()
    return model, checkpoint_format


def _dropout_inventory(model):
    probabilities = [
        float(module.p)
        for module in model.modules()
        if isinstance(module, nn.Dropout)
    ]
    positive = [value for value in probabilities if value > 0.0]
    expected = [0.05, 0.1, 0.2, 0.3, 0.5]
    if len(probabilities) != 9 or not np.allclose(positive, expected):
        raise RuntimeError(
            'unexpected U-Net dropout inventory: {}'.format(probabilities))
    return {
        'module_count': len(probabilities),
        'probabilities': probabilities,
        'positive_probabilities': positive,
    }


def _model_hash(model, include_parameters=True, include_buffers=True):
    digest = hashlib.sha256()
    tensors = []
    if include_parameters:
        tensors.extend(
            ('parameter.' + name, tensor)
            for name, tensor in model.named_parameters())
    if include_buffers:
        tensors.extend(
            ('buffer.' + name, tensor)
            for name, tensor in model.named_buffers())
    for name, tensor in tensors:
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(detached.dtype).encode('ascii'))
        digest.update(str(tuple(detached.shape)).encode('ascii'))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fork_devices(device):
    if device.type != 'cuda':
        return []
    # _seed_torch deliberately seeds all CUDA generators. Preserve every
    # visible device so a read-only analysis cannot perturb another process
    # stage that later uses a non-target GPU in the same interpreter.
    return list(range(torch.cuda.device_count()))


def _seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_lcc_stack(model, image_stack, buffer_snapshot, seed,
                       dropout_mode=None):
    batch_size, stack_size, channels, height, width = image_stack.shape
    flat = image_stack.reshape(
        batch_size * stack_size, channels, height, width)
    restore_buffers(model, buffer_snapshot)
    try:
        with torch.no_grad(), torch.random.fork_rng(
                devices=_fork_devices(image_stack.device)):
            _seed_torch(seed)
            if dropout_mode is None:
                logits = h73._logits(model(flat))
            else:
                with temporary_stack_dropout(
                        model, dropout_mode, stack_size) as patched_count:
                    if patched_count != 9:
                        raise RuntimeError(
                            'not all U-Net Dropout modules were patched')
                    logits = h73._logits(model(flat))
            prediction = h73._largest_component(logits.argmax(dim=1))
    finally:
        restore_buffers(model, buffer_snapshot)
    return prediction.reshape(batch_size, stack_size, height, width)


def _one_hot_center(center_mask, num_classes, dtype):
    encoded = F.one_hot(
        center_mask.long(), num_classes=num_classes)
    return encoded.permute(0, 3, 1, 2).to(dtype=dtype)


def _mode_metrics(occupancies, center_masks, exact_occupancy,
                  exact_center, exact_support):
    """Return per-sample single-draw fidelity and MC mechanism metrics."""
    draws, batch_size, classes, height, width = occupancies.shape
    expanded_exact = exact_occupancy.unsqueeze(0).expand_as(occupancies)
    occupancy_error = (
        occupancies - expanded_exact).square().mean(dim=2)
    full_brier = occupancy_error.mean(dim=(0, 2, 3))

    support_count = exact_support.flatten(1).sum(dim=1)
    support_denominator = (
        support_count.to(dtype=occupancy_error.dtype) * draws)
    residual_brier_sum = (
        occupancy_error * exact_support.unsqueeze(0)).sum(
            dim=(0, 2, 3))
    residual_brier = residual_brier_sum / support_denominator.clamp_min(1.0)
    residual_brier = torch.where(
        support_count > 0,
        residual_brier,
        torch.full_like(residual_brier, float('nan')))

    center_distributions = _one_hot_center(
        center_masks.reshape(draws * batch_size, height, width),
        classes, occupancies.dtype)
    center_distributions = center_distributions.reshape(
        draws, batch_size, classes, height, width)
    acquisition_delta = occupancies - center_distributions
    residual_variance_map = acquisition_delta.var(
        dim=0, unbiased=False).mean(dim=1)
    residual_variance = (
        residual_variance_map * exact_support).flatten(1).sum(dim=1) / \
        support_count.to(dtype=residual_variance_map.dtype).clamp_min(1.0)
    residual_variance = torch.where(
        support_count > 0,
        residual_variance,
        torch.full_like(residual_variance, float('nan')))
    raw_occupancy_variance = occupancies.var(
        dim=0, unbiased=False).mean(dim=1).flatten(1).mean(dim=1)

    center_dice = binary_dice_per_sample(
        center_masks.reshape(draws * batch_size, height, width),
        exact_center.unsqueeze(0).expand(
            draws, batch_size, height, width).reshape(
                draws * batch_size, height, width))
    center_dice = center_dice.reshape(draws, batch_size).mean(dim=0)

    soft_dice = soft_dice_error_per_sample(
        occupancies.reshape(
            draws * batch_size, classes, height, width),
        expanded_exact.reshape(
            draws * batch_size, classes, height, width))
    soft_dice = soft_dice.reshape(draws, batch_size).mean(dim=0)

    residual_mask = exact_support.unsqueeze(0).unsqueeze(2).to(
        dtype=occupancies.dtype)
    masked_candidate = occupancies * residual_mask
    masked_exact = expanded_exact * residual_mask
    residual_intersection = (
        masked_candidate * masked_exact).flatten(3).sum(dim=3)
    residual_denominator = (
        masked_candidate.square().flatten(3).sum(dim=3) +
        masked_exact.square().flatten(3).sum(dim=3))
    residual_soft_dice = 1.0 - (
        (2.0 * residual_intersection + 1e-10) /
        (residual_denominator + 1e-10)).mean(dim=2)
    residual_soft_dice = residual_soft_dice.mean(dim=0)
    residual_soft_dice = torch.where(
        support_count > 0,
        residual_soft_dice,
        torch.full_like(residual_soft_dice, float('nan')))

    exact_mass = exact_occupancy[:, 1].flatten(1).sum(dim=1)
    candidate_mass = occupancies[:, :, 1].flatten(2).sum(dim=2)
    mass_error = (
        candidate_mass - exact_mass.unsqueeze(0)).abs() / \
        exact_mass.unsqueeze(0).clamp_min(1.0)
    mass_error = mass_error.mean(dim=0)

    return {
        'full_brier': full_brier,
        'residual_brier': residual_brier,
        'residual_brier_numerator': residual_brier_sum,
        'residual_brier_denominator': support_denominator,
        'residual_variance': residual_variance,
        'residual_variance_numerator': (
            residual_variance_map * exact_support).flatten(1).sum(dim=1),
        'residual_variance_denominator': support_count.to(
            dtype=residual_variance_map.dtype),
        'raw_occupancy_variance': raw_occupancy_variance,
        'center_dice': center_dice,
        'soft_dice_error': soft_dice,
        'residual_soft_dice_error': residual_soft_dice,
        'absolute_relative_foreground_mass_error': mass_error,
    }


class SCTStore:
    def __init__(self):
        self.values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))

    def update(self, patient, stratum, prefix, metrics, sample_index):
        for name, values in metrics.items():
            value = float(values[sample_index].item())
            if math.isfinite(value):
                self.values[patient][stratum][
                    '{}_{}'.format(prefix, name)].append(value)

    @staticmethod
    def _mean(values):
        return statistics.fmean(values) if values else None

    def patient_report(self, patient, stratum='all'):
        metrics = {
            name: self._mean(values)
            for name, values in self.values[patient][stratum].items()
        }
        for prefix in ('independent', 'shared'):
            for metric in ('residual_brier', 'residual_variance'):
                numerator_key = '{}_{}_numerator'.format(prefix, metric)
                denominator_key = '{}_{}_denominator'.format(prefix, metric)
                numerators = self.values[patient][stratum][numerator_key]
                denominators = self.values[patient][stratum][denominator_key]
                denominator = sum(denominators)
                metrics['{}_{}'.format(prefix, metric)] = (
                    sum(numerators) / denominator
                    if denominator > 0.0 else None)
                metrics.pop(numerator_key, None)
                metrics.pop(denominator_key, None)
        metrics['active_slice_observations'] = len(
            self.values[patient][stratum][
                'independent_residual_variance_denominator'])
        metrics['exact_fractional_support_pixel_observations'] = int(sum(
            self.values[patient][stratum][
                'independent_residual_variance_denominator']))
        return _sct_decision(metrics)


class ADUStore:
    def __init__(self, pair_count):
        self.pair_count = pair_count
        self.data = defaultdict(
            lambda: [
                {
                    'scores': [],
                    'errors': [],
                    'weights': [],
                    'fractional_weights': [],
                    'identity_max_error': 0.0,
                }
                for _ in range(pair_count)
            ])

    def update(self, patient, pair_index, scores, errors, weights,
               fractional_weights, identity_max_error):
        target = self.data[patient][pair_index]
        target['scores'].append(
            scores.detach().cpu().numpy().astype(np.float32, copy=False))
        target['errors'].append(
            errors.detach().cpu().numpy().astype(np.float32, copy=False))
        target['weights'].append(
            weights.detach().cpu().numpy().astype(np.float32, copy=False))
        if fractional_weights.numel() > 0:
            target['fractional_weights'].append(
                fractional_weights.detach().cpu().numpy().astype(
                    np.float32, copy=False))
        target['identity_max_error'] = max(
            target['identity_max_error'], float(identity_max_error))

    def patient_report(self, patient):
        pair_reports = [
            _adu_pair_report(data)
            for data in self.data[patient]
        ]
        return _adu_patient_decision(pair_reports)


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _relative_reduction(candidate, reference):
    ratio = _safe_ratio(candidate, reference)
    return None if ratio is None else 1.0 - ratio


def _all_finite(values):
    return all(
        value is not None and math.isfinite(float(value))
        for value in values)


def _sct_decision(metrics):
    residual_variance_reduction = _relative_reduction(
        metrics.get('shared_residual_variance'),
        metrics.get('independent_residual_variance'))
    residual_brier_reduction = _relative_reduction(
        metrics.get('shared_residual_brier'),
        metrics.get('independent_residual_brier'))
    full_brier_ratio = _safe_ratio(
        metrics.get('shared_full_brier'),
        metrics.get('independent_full_brier'))
    center_dice_delta = None
    if _all_finite([
            metrics.get('shared_center_dice'),
            metrics.get('independent_center_dice')]):
        center_dice_delta = (
            metrics['shared_center_dice'] -
            metrics['independent_center_dice'])
    observed = {
        'residual_variance_reduction':
            residual_variance_reduction,
        'residual_brier_reduction': residual_brier_reduction,
        'full_brier_ratio': full_brier_ratio,
        'center_dice_delta': center_dice_delta,
    }
    passed = _all_finite(observed.values()) and \
        residual_variance_reduction >= \
        SCT_MIN_RESIDUAL_VARIANCE_REDUCTION and \
        residual_brier_reduction >= SCT_MIN_RESIDUAL_BRIER_REDUCTION and \
        full_brier_ratio <= SCT_MAX_FULL_BRIER_RATIO and \
        center_dice_delta >= SCT_MIN_CENTER_DICE_DELTA
    return {
        'metrics': metrics,
        'observed': observed,
        'pass_without_non_clamped_guard': bool(passed),
    }


def _concatenate(parts):
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)


def _adu_pair_report(data):
    scores = _concatenate(data['scores'])
    errors = _concatenate(data['errors'])
    weights = _concatenate(data['weights'])
    fractional_weights = _concatenate(data['fractional_weights'])
    if scores.size == 0:
        return {
            'pass': False,
            'nondegenerate': False,
            'reason': 'empty_foreground_union',
        }
    rho = spearman_correlation(scores, errors)
    top_ratio = top_fraction_error_ratio(scores, errors, fraction=0.20)
    unweighted = float(errors.mean())
    weight_sum = float(weights.sum())
    weighted = None if weight_sum <= 0.0 else float(
        np.dot(weights.astype(np.float64), errors.astype(np.float64)) /
        weight_sum)
    weighted_reduction = _relative_reduction(weighted, unweighted)
    fractional_retention = (
        float(fractional_weights.mean())
        if fractional_weights.size else None)
    mean_weight = float(weights.mean())
    denominator = float(
        weights.size * np.dot(
            weights.astype(np.float64), weights.astype(np.float64)))
    ess_fraction = None if denominator <= 0.0 else \
        weight_sum * weight_sum / denominator
    positive_js_fraction = float((scores > JS_EPSILON).mean())
    nondegenerate = (
        scores.size >= 2 and float(scores.std()) > 0.0 and
        positive_js_fraction > 0.0)
    core = [rho, top_ratio, weighted_reduction, fractional_retention]
    passed = _all_finite(core) and nondegenerate and \
        rho >= ADU_MIN_SPEARMAN and \
        top_ratio >= ADU_MIN_TOP20_ERROR_RATIO and \
        weighted_reduction >= ADU_MIN_WEIGHTED_BRIER_REDUCTION and \
        fractional_retention >= ADU_MIN_FRACTIONAL_WEIGHT and \
        data['identity_max_error'] <= CONVEXITY_ATOL
    return {
        'pass': bool(passed),
        'nondegenerate': bool(nondegenerate),
        'eligible_pixels': int(scores.size),
        'spearman_js_error': rho,
        'top20_error_ratio': top_ratio,
        'unweighted_brier': unweighted,
        'weighted_brier': weighted,
        'weighted_brier_reduction': weighted_reduction,
        'mean_reliability_weight': mean_weight,
        'effective_sample_fraction': ess_fraction,
        'fractional_support_mean_weight': fractional_retention,
        'positive_js_fraction': positive_js_fraction,
        'js_standard_deviation': float(scores.std()),
        'convexity_identity_max_abs_error':
            data['identity_max_error'],
    }


def _median_or_none(values):
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _adu_patient_decision(pair_reports):
    fields = (
        'spearman_js_error',
        'top20_error_ratio',
        'weighted_brier_reduction',
        'fractional_support_mean_weight',
        'effective_sample_fraction',
        'positive_js_fraction',
    )
    medians = {
        field: _median_or_none([
            report.get(field) for report in pair_reports])
        for field in fields
    }
    all_pairs_complete = (
        len(pair_reports) == LOCKED_MC_DRAWS // 2 and
        all(
            report.get('eligible_pixels', 0) > 0 and
            report.get('nondegenerate', False) and
            _all_finite([
                report.get('spearman_js_error'),
                report.get('top20_error_ratio'),
                report.get('weighted_brier_reduction'),
                report.get('fractional_support_mean_weight'),
            ]) and
            report.get(
                'convexity_identity_max_abs_error', math.inf) <=
            CONVEXITY_ATOL
            for report in pair_reports))
    core = [
        medians['spearman_js_error'],
        medians['top20_error_ratio'],
        medians['weighted_brier_reduction'],
        medians['fractional_support_mean_weight'],
    ]
    passed = all_pairs_complete and _all_finite(core) and \
        medians['spearman_js_error'] >= ADU_MIN_SPEARMAN and \
        medians['top20_error_ratio'] >= ADU_MIN_TOP20_ERROR_RATIO and \
        medians['weighted_brier_reduction'] >= \
        ADU_MIN_WEIGHTED_BRIER_REDUCTION and \
        medians['fractional_support_mean_weight'] >= \
        ADU_MIN_FRACTIONAL_WEIGHT and \
        all(report.get('convexity_identity_max_abs_error', math.inf) <=
            CONVEXITY_ATOL for report in pair_reports)
    return {
        'pass': bool(passed),
        'all_pairs_complete': bool(all_pairs_complete),
        'pair_medians': medians,
        'pairs': pair_reports,
    }


def _summary(values):
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))]
    if not finite:
        return {'mean': None, 'median': None, 'q1': None, 'q3': None}
    ordered = np.asarray(finite, dtype=np.float64)
    return {
        'mean': float(ordered.mean()),
        'median': float(np.median(ordered)),
        'q1': float(np.quantile(ordered, 0.25)),
        'q3': float(np.quantile(ordered, 0.75)),
    }


def _sct_balanced_summary(patient_reports):
    observed_fields = (
        'residual_variance_reduction',
        'residual_brier_reduction',
        'full_brier_ratio',
        'center_dice_delta',
    )
    return {
        field: _summary([
            report['observed'][field]
            for report in patient_reports.values()])
        for field in observed_fields
    }


def _checkpoint_sct_report(store, patients):
    patient_reports = {}
    for patient in patients:
        report = store.patient_report(patient, 'all')
        non_clamped = store.patient_report(patient, 'non_clamped')
        non_clamped_gain = non_clamped['observed'][
            'residual_brier_reduction']
        passed = report['pass_without_non_clamped_guard'] and \
            non_clamped_gain is not None and non_clamped_gain > 0.0
        report['non_clamped'] = non_clamped
        report['pass'] = bool(passed)
        patient_reports[patient] = report
    patient_passes = sum(
        report['pass'] for report in patient_reports.values())
    balanced = _sct_balanced_summary(patient_reports)
    core_medians = [
        balanced['residual_variance_reduction']['median'],
        balanced['residual_brier_reduction']['median'],
        balanced['full_brier_ratio']['median'],
        balanced['center_dice_delta']['median'],
    ]
    checkpoint_pass = patient_passes >= MIN_PATIENT_PASSES and \
        _all_finite(core_medians) and \
        core_medians[0] >= \
        SCT_MIN_RESIDUAL_VARIANCE_REDUCTION and \
        core_medians[1] >= \
        SCT_MIN_RESIDUAL_BRIER_REDUCTION and \
        core_medians[2] <= \
        SCT_MAX_FULL_BRIER_RATIO and \
        core_medians[3] >= \
        SCT_MIN_CENTER_DICE_DELTA
    axial_strata = {}
    for stratum in (
            'first_index_third', 'middle_index_third', 'last_index_third'):
        stratum_reports = {
            patient: store.patient_report(patient, stratum)
            for patient in patients
        }
        axial_strata[stratum] = {
            'patient_balanced_summary':
                _sct_balanced_summary(stratum_reports),
            'by_patient': stratum_reports,
        }
    return {
        'pass': bool(checkpoint_pass),
        'patient_passes': patient_passes,
        'required_patient_passes': MIN_PATIENT_PASSES,
        'patient_balanced_summary': balanced,
        'by_patient': patient_reports,
        'axial_index_thirds_descriptive_only': axial_strata,
    }


def _checkpoint_adu_report(store, patients):
    patient_reports = {
        patient: store.patient_report(patient)
        for patient in patients
    }
    patient_passes = sum(
        report['pass'] for report in patient_reports.values())
    fields = (
        'spearman_js_error',
        'top20_error_ratio',
        'weighted_brier_reduction',
        'fractional_support_mean_weight',
        'effective_sample_fraction',
        'positive_js_fraction',
    )
    balanced = {
        field: _summary([
            report['pair_medians'][field]
            for report in patient_reports.values()])
        for field in fields
    }
    core_medians = [
        balanced['spearman_js_error']['median'],
        balanced['top20_error_ratio']['median'],
        balanced['weighted_brier_reduction']['median'],
        balanced['fractional_support_mean_weight']['median'],
    ]
    checkpoint_pass = patient_passes >= MIN_PATIENT_PASSES and \
        _all_finite(core_medians) and \
        core_medians[0] >= ADU_MIN_SPEARMAN and \
        core_medians[1] >= \
        ADU_MIN_TOP20_ERROR_RATIO and \
        core_medians[2] >= \
        ADU_MIN_WEIGHTED_BRIER_REDUCTION and \
        core_medians[3] >= \
        ADU_MIN_FRACTIONAL_WEIGHT
    return {
        'pass': bool(checkpoint_pass),
        'patient_passes': patient_passes,
        'required_patient_passes': MIN_PATIENT_PASSES,
        'patient_balanced_summary': balanced,
        'by_patient': patient_reports,
    }


def _axial_stratum(patient, slice_index, case_bounds):
    first_index, last_index = case_bounds[patient]
    span = max(last_index - first_index + 1, 1)
    relative = (slice_index - first_index + 0.5) / span
    if relative < 1.0 / 3.0:
        return 'first_index_third'
    if relative < 2.0 / 3.0:
        return 'middle_index_third'
    return 'last_index_third'


def _stable_forward_seed(base_seed, schedule_index, batch_index, draw_index,
                         family_offset):
    return int(
        base_seed + family_offset +
        schedule_index * 100000 + batch_index * 100 + draw_index)


def _stack_batch(samples, weights, indices, batch_size, device):
    valid_count = len(indices)
    padded_indices = list(indices)
    while len(padded_indices) < batch_size:
        padded_indices.append(indices[0])
    images = torch.stack([
        samples[index]['image_stack'] for index in padded_indices]).to(device)
    labels = torch.stack([
        samples[index]['label_stack'] for index in padded_indices]).to(device)
    batch_weights = weights[padded_indices].to(device)
    return images, labels, batch_weights, valid_count, padded_indices


def _update_sct_store(store, metrics_independent, metrics_shared,
                      samples, batch_indices, valid_count, case_bounds):
    for sample_index in range(valid_count):
        sample = samples[batch_indices[sample_index]]
        strata = [
            'all',
            'clamped' if sample['neighbor_clamped'] else 'non_clamped',
            _axial_stratum(
                sample['patient'], sample['slice_index'], case_bounds),
        ]
        for stratum in strata:
            store.update(
                sample['patient'], stratum, 'independent',
                metrics_independent, sample_index)
            store.update(
                sample['patient'], stratum, 'shared',
                metrics_shared, sample_index)


def _update_adu_store(store, ordinary_occupancies, exact_occupancy,
                      exact_support, samples, batch_indices, valid_count):
    draws = ordinary_occupancies.shape[0]
    exact_foreground = exact_occupancy[:, 1]
    predicted_foreground = ordinary_occupancies[:, :, 1].amax(dim=0)
    eligible = (
        (exact_foreground > RESIDUAL_EPSILON) |
        (predicted_foreground > RESIDUAL_EPSILON))
    fractional = (
        (exact_foreground > RESIDUAL_EPSILON) &
        (exact_foreground < 1.0 - RESIDUAL_EPSILON))

    for pair_index in range(draws // 2):
        first = ordinary_occupancies[2 * pair_index]
        second = ordinary_occupancies[2 * pair_index + 1]
        mean = 0.5 * (first + second)
        js = jensen_shannon_map(first, second)
        reliability = reliability_from_js(js, first.shape[1])
        error = occupancy_brier_map(mean, exact_occupancy)
        first_error = occupancy_brier_map(first, exact_occupancy)
        second_error = occupancy_brier_map(second, exact_occupancy)
        disagreement = occupancy_brier_map(first, second)
        identity_error = (
            0.5 * (first_error + second_error) - error -
            0.25 * disagreement).abs()
        identity_max_per_sample = identity_error.flatten(1).amax(dim=1)
        identity_max = float(
            identity_max_per_sample[:valid_count].max().item())
        if identity_max > CONVEXITY_ATOL:
            raise RuntimeError(
                'Brier convexity identity failed: {}'.format(identity_max))

        for sample_index in range(valid_count):
            mask = eligible[sample_index]
            sample = samples[batch_indices[sample_index]]
            store.update(
                sample['patient'],
                pair_index,
                js[sample_index][mask],
                error[sample_index][mask],
                reliability[sample_index][mask],
                reliability[sample_index][fractional[sample_index]],
                float(identity_max_per_sample[sample_index].item()))


def _run_checkpoint(args, checkpoint_step, checkpoint_path, samples,
                    profile_weights, preflight, device):
    model, checkpoint_format = _load_train_mode_proxy(
        checkpoint_path, device)
    dropout_inventory = _dropout_inventory(model)
    initial_hash = _model_hash(model)
    initial_buffers = snapshot_buffers(model)
    sct_store = SCTStore()
    adu_store = ADUStore(args.mc_draws // 2)
    padded_observations = 0
    batch_manifests = []

    for schedule_index, schedule_seed in enumerate(
            args.batch_schedule_seeds):
        permutation = np.random.RandomState(schedule_seed).permutation(
            len(samples)).tolist()
        batches = [
            permutation[start:start + args.batch_size]
            for start in range(0, len(permutation), args.batch_size)
        ]
        batch_manifests.append({
            'seed': schedule_seed,
            'batch_count': len(batches),
            'permutation_sha256': hashlib.sha256(
                np.asarray(permutation, dtype=np.int32).tobytes()).hexdigest(),
            'final_batch_size': len(batches[-1]),
        })

        for batch_index, indices in enumerate(batches):
            image_stack, label_stack, weights, valid_count, padded_indices = \
                _stack_batch(
                    samples, profile_weights, indices,
                    args.batch_size, device)
            padded_observations += args.batch_size - valid_count
            center = image_stack.shape[1] // 2
            with torch.no_grad():
                _, _, exact_occupancy = paired_slice_reacquisition(
                    image_stack, label_stack, weights, args.num_classes)
                exact_center = label_stack[:, center]
                exact_center_distribution = _one_hot_center(
                    exact_center, args.num_classes, exact_occupancy.dtype)
                exact_delta = exact_occupancy - exact_center_distribution
                exact_support = (
                    0.5 * exact_delta.abs().sum(dim=1) >
                    RESIDUAL_EPSILON)

            independent_occupancies = []
            independent_centers = []
            shared_occupancies = []
            shared_centers = []
            ordinary_occupancies = []

            for draw_index in range(args.mc_draws):
                sct_seed = _stable_forward_seed(
                    args.seed, schedule_index, batch_index, draw_index,
                    family_offset=1000000)
                independent_stack = _predict_lcc_stack(
                    model, image_stack, initial_buffers, sct_seed,
                    dropout_mode=INDEPENDENT_DROPOUT)
                shared_stack = _predict_lcc_stack(
                    model, image_stack, initial_buffers, sct_seed,
                    dropout_mode=STACK_SHARED_DROPOUT)
                with torch.no_grad():
                    _, _, independent_occupancy = \
                        paired_slice_reacquisition(
                            image_stack, independent_stack, weights,
                            args.num_classes)
                    _, _, shared_occupancy = paired_slice_reacquisition(
                        image_stack, shared_stack, weights, args.num_classes)
                independent_occupancies.append(independent_occupancy)
                independent_centers.append(
                    independent_stack[:, center].to(torch.uint8))
                shared_occupancies.append(shared_occupancy)
                shared_centers.append(
                    shared_stack[:, center].to(torch.uint8))

                ordinary_seed = _stable_forward_seed(
                    args.seed, schedule_index, batch_index, draw_index,
                    family_offset=2000000)
                ordinary_stack = _predict_lcc_stack(
                    model, image_stack, initial_buffers, ordinary_seed,
                    dropout_mode=None)
                with torch.no_grad():
                    _, _, ordinary_occupancy = paired_slice_reacquisition(
                        image_stack, ordinary_stack, weights,
                        args.num_classes)
                ordinary_occupancies.append(ordinary_occupancy)

            independent_occupancies = torch.stack(
                independent_occupancies)
            independent_centers = torch.stack(independent_centers)
            shared_occupancies = torch.stack(shared_occupancies)
            shared_centers = torch.stack(shared_centers)
            ordinary_occupancies = torch.stack(ordinary_occupancies)

            metrics_independent = _mode_metrics(
                independent_occupancies,
                independent_centers,
                exact_occupancy,
                exact_center,
                exact_support)
            metrics_shared = _mode_metrics(
                shared_occupancies,
                shared_centers,
                exact_occupancy,
                exact_center,
                exact_support)
            _update_sct_store(
                sct_store,
                metrics_independent,
                metrics_shared,
                samples,
                padded_indices,
                valid_count,
                preflight['case_bounds'])
            _update_adu_store(
                adu_store,
                ordinary_occupancies,
                exact_occupancy,
                exact_support,
                samples,
                padded_indices,
                valid_count)

            print(
                '[H7.10 step {} schedule {}/{} batch {}/{}] '
                'valid={} padded={}'.format(
                    checkpoint_step,
                    schedule_index + 1,
                    len(args.batch_schedule_seeds),
                    batch_index + 1,
                    len(batches),
                    valid_count,
                    args.batch_size - valid_count),
                flush=True)

    restore_buffers(model, initial_buffers)
    final_hash = _model_hash(model)
    if final_hash != initial_hash:
        raise RuntimeError('model parameters or buffers changed during gate')
    patients = preflight['labeled_patients']
    sct_report = _checkpoint_sct_report(sct_store, patients)
    adu_report = _checkpoint_adu_report(adu_store, patients)
    return {
        'step': checkpoint_step,
        'path': checkpoint_path,
        'sha256': _sha256(checkpoint_path),
        'checkpoint_format': checkpoint_format,
        'checkpoint_role': 'student_as_proxy_teacher',
        'checkpoint_contains_ema_teacher': False,
        'dropout_inventory': dropout_inventory,
        'model_state_sha256_before': initial_hash,
        'model_state_sha256_after': final_hash,
        'batch_schedules': batch_manifests,
        'evaluated_unique_samples': len(samples),
        'evaluated_sample_observations':
            len(samples) * len(args.batch_schedule_seeds),
        'padded_forward_observations': padded_observations,
        'sct': sct_report,
        'adu': adu_report,
    }


def _cross_checkpoint_patient_consistency(checkpoint_reports):
    patient_names = sorted(
        checkpoint_reports[0]['sct']['by_patient'])
    sct_by_patient = {}
    adu_by_patient = {}
    for patient in patient_names:
        sct_observed = {
            field: _median_or_none([
                report['sct']['by_patient'][patient]['observed'][field]
                for report in checkpoint_reports])
            for field in (
                'residual_variance_reduction',
                'residual_brier_reduction',
                'full_brier_ratio',
                'center_dice_delta',
            )
        }
        non_clamped_gain = _median_or_none([
            report['sct']['by_patient'][patient]['non_clamped'][
                'observed']['residual_brier_reduction']
            for report in checkpoint_reports])
        sct_core = list(sct_observed.values()) + [non_clamped_gain]
        sct_pass = _all_finite(sct_core) and \
            sct_observed['residual_variance_reduction'] >= \
            SCT_MIN_RESIDUAL_VARIANCE_REDUCTION and \
            sct_observed['residual_brier_reduction'] >= \
            SCT_MIN_RESIDUAL_BRIER_REDUCTION and \
            sct_observed['full_brier_ratio'] <= \
            SCT_MAX_FULL_BRIER_RATIO and \
            sct_observed['center_dice_delta'] >= \
            SCT_MIN_CENTER_DICE_DELTA and non_clamped_gain > 0.0
        sct_by_patient[patient] = {
            'pass': bool(sct_pass),
            'checkpoint_median_observed': sct_observed,
            'checkpoint_median_non_clamped_residual_brier_reduction':
                non_clamped_gain,
        }

        complete_reports = [
            report for report in checkpoint_reports
            if report['adu']['by_patient'][patient].get(
                'all_pairs_complete', False)
        ]
        adu_observed = {
            field: _median_or_none([
                report['adu']['by_patient'][patient][
                    'pair_medians'][field]
                for report in complete_reports])
            for field in (
                'spearman_js_error',
                'top20_error_ratio',
                'weighted_brier_reduction',
                'fractional_support_mean_weight',
            )
        }
        adu_core = list(adu_observed.values())
        complete_checkpoint_count = sum(
            bool(report['adu']['by_patient'][patient].get(
                'all_pairs_complete', False))
            for report in checkpoint_reports)
        adu_pass = _all_finite(adu_core) and \
            complete_checkpoint_count >= MIN_CHECKPOINT_PASSES and \
            adu_observed['spearman_js_error'] >= ADU_MIN_SPEARMAN and \
            adu_observed['top20_error_ratio'] >= \
            ADU_MIN_TOP20_ERROR_RATIO and \
            adu_observed['weighted_brier_reduction'] >= \
            ADU_MIN_WEIGHTED_BRIER_REDUCTION and \
            adu_observed['fractional_support_mean_weight'] >= \
            ADU_MIN_FRACTIONAL_WEIGHT
        adu_by_patient[patient] = {
            'pass': bool(adu_pass),
            'quality_complete_checkpoints': complete_checkpoint_count,
            'required_quality_complete_checkpoints': MIN_CHECKPOINT_PASSES,
            'checkpoint_median_observed': adu_observed,
        }
    return {
        'sct': {
            'patient_passes': sum(
                item['pass'] for item in sct_by_patient.values()),
            'required_patient_passes': MIN_PATIENT_PASSES,
            'by_patient': sct_by_patient,
        },
        'adu': {
            'patient_passes': sum(
                item['pass'] for item in adu_by_patient.values()),
            'required_patient_passes': MIN_PATIENT_PASSES,
            'by_patient': adu_by_patient,
        },
    }


def _across_checkpoint_decision(checkpoint_reports):
    sct_passes = sum(
        report['sct']['pass'] for report in checkpoint_reports)
    adu_passes = sum(
        report['adu']['pass'] for report in checkpoint_reports)
    consistency = _cross_checkpoint_patient_consistency(checkpoint_reports)
    sct_patient_passes = consistency['sct']['patient_passes']
    adu_patient_passes = consistency['adu']['patient_passes']
    sct_pass = sct_passes >= MIN_CHECKPOINT_PASSES and \
        sct_patient_passes >= MIN_PATIENT_PASSES
    adu_pass = adu_passes >= MIN_CHECKPOINT_PASSES and \
        adu_patient_passes >= MIN_PATIENT_PASSES
    sct_gain = _median_or_none([
        report['sct']['patient_balanced_summary'][
            'residual_brier_reduction']['median']
        for report in checkpoint_reports])
    adu_gain = _median_or_none([
        report['adu']['patient_balanced_summary'][
            'weighted_brier_reduction']['median']
        for report in checkpoint_reports])

    if sct_pass and adu_pass:
        if _all_finite([sct_gain, adu_gain]) and \
                adu_gain - sct_gain >= ADU_SELECTION_MARGIN:
            selected = 'adu'
            joint = 'authorize_slice_eq_occ_adu_training'
        else:
            selected = 'sct'
            joint = 'authorize_slice_eq_occ_sct_training'
    elif sct_pass:
        selected = 'sct'
        joint = 'authorize_slice_eq_occ_sct_training'
    elif adu_pass:
        selected = 'adu'
        joint = 'authorize_slice_eq_occ_adu_training'
    else:
        selected = None
        joint = 'stop_h7_10_small_method_extensions'
    return {
        'sct': {
            'pass': sct_pass,
            'checkpoint_passes': sct_passes,
            'required_checkpoint_passes': MIN_CHECKPOINT_PASSES,
            'cross_checkpoint_patient_passes': sct_patient_passes,
            'required_cross_checkpoint_patient_passes': MIN_PATIENT_PASSES,
            'median_effective_gain': sct_gain,
        },
        'adu': {
            'pass': adu_pass,
            'checkpoint_passes': adu_passes,
            'required_checkpoint_passes': MIN_CHECKPOINT_PASSES,
            'cross_checkpoint_patient_passes': adu_patient_passes,
            'required_cross_checkpoint_patient_passes': MIN_PATIENT_PASSES,
            'median_effective_gain': adu_gain,
        },
        'selected_candidate': selected,
        'joint_decision': joint,
        'cross_checkpoint_patient_consistency': consistency,
    }


def _validate_args(args):
    if list(args.checkpoint_steps) != DEFAULT_CHECKPOINT_STEPS:
        raise ValueError('confirmatory H7.10 requires 18k/24k/30k checkpoints')
    if args.seed != 1337:
        raise ValueError('confirmatory H7.10 requires seed 1337')
    if list(args.batch_schedule_seeds) != DEFAULT_BATCH_SCHEDULE_SEEDS:
        raise ValueError(
            'confirmatory H7.10 requires schedule seeds 1337 and 7331')
    if args.labeled_slices != LOCKED_LABELED_SLICES:
        raise ValueError('confirmatory H7.10 requires 191 labeled slices')
    if args.labelnum != LOCKED_LABELNUM:
        raise ValueError('confirmatory H7.10 requires seven labeled patients')
    if args.batch_size != LOCKED_BATCH_SIZE:
        raise ValueError('confirmatory H7.10 requires batch size 12')
    if args.mc_draws != LOCKED_MC_DRAWS or args.mc_draws % 2:
        raise ValueError('confirmatory H7.10 requires eight MC draws')
    if list(args.patch_size) != [256, 256]:
        raise ValueError('confirmatory H7.10 requires a 256x256 patch')
    if args.sliceeq_radius != 1:
        raise ValueError('confirmatory H7.10 requires radius one')
    locked_profile = [0.45, 0.85, -0.25, 0.25]
    supplied_profile = [
        args.sliceeq_sigma_min,
        args.sliceeq_sigma_max,
        args.sliceeq_phase_min,
        args.sliceeq_phase_max,
    ]
    if not np.allclose(supplied_profile, locked_profile, atol=0.0, rtol=0.0):
        raise ValueError('confirmatory H7.10 requires the parent profile range')
    if args.sliceeq_sigma_min <= 0.0 or \
            args.sliceeq_sigma_max < args.sliceeq_sigma_min:
        raise ValueError('invalid SliceEq sigma range')
    if args.sliceeq_phase_max < args.sliceeq_phase_min:
        raise ValueError('invalid SliceEq phase range')


def _provenance(args, checkpoints, preflight, device):
    code_root = Path(__file__).resolve().parent
    workspace_root = code_root.parent
    tracked = [
        Path(__file__).resolve(),
        code_root / 'utils' / 'sliceeq_reliability.py',
        code_root / 'analyze_sliceeq_gates.py',
        code_root / 'utils' / 'sliceeq.py',
        code_root / 'utils' / 'sliceeq_occ.py',
        code_root / 'dataloaders' / 'sliceeq_dataset.py',
        code_root / 'networks' / 'unet.py',
        code_root / 'train_sliceeq_occ.py',
        workspace_root / 'research' / 'experiments' /
        'h7_slice_profile_reacquisition' /
        'h7_10_operator_reliability_gate_protocol.md',
        Path(preflight['train_cases_path']),
        Path(preflight['train_slices_path']),
    ] + [Path(path) for _, path in checkpoints]
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_kind': 'read_only_zero_training',
        'checkpoint_role': 'student_as_proxy_teacher',
        'checkpoint_contains_ema_teacher': False,
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'arguments': vars(args),
        'rng_contract': (
            'fixed preloaded transforms/profiles; two fixed case-mixed '
            'batch schedules; paired custom dropout consumes equal masks; '
            'every forward is forked and all model buffers are restored'),
        'file_sha256': {
            str(path.resolve()): _sha256(path.resolve())
            for path in tracked
        },
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    _validate_args(args)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = h73._resolve_device(args.device)
    checkpoints = _resolve_checkpoints(args)
    preflight = _labeled_only_preflight(
        args.root_path, args.labeled_slices, args.labelnum)
    samples, label_h5_reads = _preload_labeled_samples(args, preflight)
    profile_weights, profile_summary = _profile_weights(args, device)

    print(
        'Running H7.10 on {} labeled slices, {} patients, {} checkpoints'.format(
            len(samples), len(preflight['labeled_patients']),
            len(checkpoints)), flush=True)
    checkpoint_reports = [
        _run_checkpoint(
            args, step, path, samples, profile_weights, preflight, device)
        for step, path in checkpoints
    ]
    decisions = _across_checkpoint_decision(checkpoint_reports)
    report = {
        'schema_version': 1,
        'hypothesis': 'H7.10',
        'provenance': _provenance(
            args, checkpoints, preflight, device),
        'data_contract': {
            key: value for key, value in preflight.items()
            if key != 'allowlist'
        },
        'data_access_audit': {
            'labeled_dataset_items': len(samples),
            'labeled_h5_label_reads': label_h5_reads,
            'unlabeled_h5_label_reads': 0,
            'validation_label_reads': 0,
            'test_label_reads': 0,
            'lists_read': ['train.list', 'train_slices.list'],
        },
        'profile_summary': profile_summary,
        'locked_definitions': {
            'sct': (
                'paired custom Bernoulli dropout; shared mode broadcasts '
                'the center mask over each contiguous three-slice stack'),
            'sct_reference': (
                'standard-distribution custom independent dropout, not '
                'claimed bit-identical to CUDA F.dropout'),
            'adu': (
                'four disjoint pairs from eight builtin train-mode dropout '
                'draws after hard argmax, per-slice 2D LCC, and one profile'),
            'exact_fractional_support':
                '0.5*L1(q_exact, one_hot(y_center)) > 1e-6',
            'adu_primary_support': (
                'union of exact foreground occupancy and predicted '
                'foreground occupancy across the eight ordinary draws'),
            'patient_balance':
                'pixels/draws/slices aggregate within patient first',
        },
        'thresholds': {
            'sct_min_residual_variance_reduction':
                SCT_MIN_RESIDUAL_VARIANCE_REDUCTION,
            'sct_min_residual_brier_reduction':
                SCT_MIN_RESIDUAL_BRIER_REDUCTION,
            'sct_max_full_brier_ratio': SCT_MAX_FULL_BRIER_RATIO,
            'sct_min_center_dice_delta': SCT_MIN_CENTER_DICE_DELTA,
            'adu_min_spearman': ADU_MIN_SPEARMAN,
            'adu_min_top20_error_ratio': ADU_MIN_TOP20_ERROR_RATIO,
            'adu_min_weighted_brier_reduction':
                ADU_MIN_WEIGHTED_BRIER_REDUCTION,
            'adu_min_fractional_weight': ADU_MIN_FRACTIONAL_WEIGHT,
            'minimum_patient_passes': MIN_PATIENT_PASSES,
            'minimum_checkpoint_passes': MIN_CHECKPOINT_PASSES,
        },
        'checkpoints': checkpoint_reports,
        'decisions': decisions,
        'interpretation_warning': (
            'Historical SliceEqOcc checkpoints contain only the student '
            'state_dict. This train-mode frozen-student proxy gate is a '
            'necessary method screen, not direct reconstruction of the '
            'training-time EMA teacher and not evidence of test improvement.'),
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(
            report, stream, ensure_ascii=False, indent=2,
            sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary_path, output_path)
    print(json.dumps(decisions, ensure_ascii=False, indent=2), flush=True)
    print('Wrote H7.10 gate report: {}'.format(output_path), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
