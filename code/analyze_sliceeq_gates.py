"""Run the three preregistered SliceEq H7.3 mechanism gates.

This is a read-only model analysis. It never trains a model and never writes a
checkpoint. The only output is a provenance-rich JSON report.
"""

import argparse
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
from skimage.measure import label as connected_components
import torch
from torch.utils.data import DataLoader, Subset

from dataloaders.sliceeq_dataset import (
    SliceStackDataSets, StackRandomGenerator)
from networks.unet import UNet
from utils.sliceeq import paired_slice_reacquisition, sample_slice_profiles
from utils.sliceeq_gates import (
    PearsonAccumulator, acquisition_residual, gradient_pixel_norm,
    normalized_weighted_soft_cross_entropy, per_sample_gradient_cosine)
from utils.sliceeq_occ import soft_cross_entropy, soft_segmentation_loss


DEFAULT_ROOT = (
    '/home/aiteam/zhengtaoma/Baseline/data/'
    'PROMISE12_h5_training_source')
DEFAULT_CHECKPOINT = (
    '/home/aiteam/zhengtaoma/CoDA/model/'
    'SliceEqOcc_PROMISE12_7_labeled/self_train/unet/'
    'iter_23000_dice_0.8152.pth')
DEFAULT_OUTPUT = (
    '../model/SliceEqOcc_PROMISE12_7_labeled/analysis/'
    'h7_3_gates_iter23000.json')

RESIDUAL_EPSILON = 1e-6
GATE1_MAX_GRADIENT_SHARE = 0.20
GATE2_MIN_PIXEL_CORRELATION = 0.30
GATE2_MIN_MASS_CORRELATION = 0.30
GATE2_MAX_OUTSIDE_MASS = 0.50
GATE3_MAX_GRADIENT_COSINE = 0.98


class ImageOnlySliceStackDataSets(SliceStackDataSets):
    """Preserve stack geometry/augmentation without reading hidden U labels."""

    def _read_slice(self, sample_name):
        path = os.path.join(
            self._base_dir, 'data', 'slices', sample_name + '.h5')
        with h5py.File(path, 'r') as stream:
            image = stream['image'][:]
        if image.ndim != 2:
            raise ValueError(
                'SliceEq expects a 2D image array: {}'.format(path))
        placeholder = np.zeros_like(image, dtype=np.uint8)
        return image, placeholder


def build_parser():
    parser = argparse.ArgumentParser(
        description='Read-only SliceEq H7.3 three-gate analysis')
    parser.add_argument('--root_path', type=str, default=DEFAULT_ROOT)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output_json', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_classes', type=int, default=2, choices=[2])
    parser.add_argument('--labeled_slices', type=int, default=191)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_labeled_batches', type=int, default=16)
    parser.add_argument('--max_unlabeled_batches', type=int, default=16)
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


def _logits(network_output):
    return network_output[0] if isinstance(network_output, tuple) \
        else network_output


def _largest_component(prediction):
    """Match the baseline's binary per-slice foreground LCC operation."""
    prediction_np = prediction.detach().cpu().numpy()
    output = np.zeros_like(prediction_np, dtype=np.int64)
    for sample_index in range(prediction_np.shape[0]):
        foreground = prediction_np[sample_index] == 1
        components = connected_components(foreground)
        component_count = int(components.max())
        if component_count == 0:
            continue
        counts = np.bincount(components.reshape(-1))[1:]
        largest_index = int(np.argmax(counts)) + 1
        output[sample_index] = components == largest_index
    return torch.from_numpy(output).to(
        device=prediction.device, dtype=torch.long)


def _predict_lcc_stack(model, image_stack):
    batch_size, stack_size, channels, height, width = image_stack.shape
    flat_images = image_stack.reshape(
        batch_size * stack_size, channels, height, width)
    with torch.no_grad():
        logits = _logits(model(flat_images))
        prediction = logits.argmax(dim=1)
        prediction = _largest_component(prediction)
    return prediction.reshape(batch_size, stack_size, height, width)


def _load_model(checkpoint_path, device):
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
    model.eval()
    return model, checkpoint_format


class GradientGateAccumulator:
    def __init__(self):
        self.samples = 0
        self.batches = 0
        self.total_pixels = 0
        self.support_pixels = 0
        self.residual_sum = 0.0
        self.residual_square_sum = 0.0
        self.full_gradient_sum = 0.0
        self.full_support_gradient_sum = 0.0
        self.ce_gradient_sum = 0.0
        self.ce_support_gradient_sum = 0.0
        self.cosines = []
        self.unit_distances = []
        self.active_gradient_samples = 0

    def update(self, logits, occupancy, center_target):
        residual = acquisition_residual(occupancy, center_target)
        support = residual > RESIDUAL_EPSILON
        active_samples = support.flatten(1).any(dim=1)

        full_loss, _, _ = soft_segmentation_loss(logits, occupancy)
        ce_loss = soft_cross_entropy(logits, occupancy)
        full_gradient = torch.autograd.grad(
            full_loss, logits, retain_graph=True)[0]
        ce_gradient = torch.autograd.grad(
            ce_loss, logits, retain_graph=True)[0]

        residual_loss = normalized_weighted_soft_cross_entropy(
            logits, occupancy, residual)
        binary_loss = normalized_weighted_soft_cross_entropy(
            logits, occupancy, support.to(dtype=logits.dtype))
        residual_gradient = torch.autograd.grad(
            residual_loss, logits, retain_graph=True)[0]
        binary_gradient = torch.autograd.grad(binary_loss, logits)[0]

        full_norm = gradient_pixel_norm(full_gradient)
        ce_norm = gradient_pixel_norm(ce_gradient)
        cosine, distance = per_sample_gradient_cosine(
            residual_gradient, binary_gradient, active_samples)

        detached_residual = residual.detach().double()
        support_values = detached_residual[support]
        self.samples += int(logits.shape[0])
        self.batches += 1
        self.total_pixels += int(support.numel())
        self.support_pixels += int(support.sum().item())
        self.full_gradient_sum += float(full_norm.sum().item())
        self.full_support_gradient_sum += float(full_norm[support].sum().item())
        self.ce_gradient_sum += float(ce_norm.sum().item())
        self.ce_support_gradient_sum += float(ce_norm[support].sum().item())
        if support_values.numel() > 0:
            self.residual_sum += float(support_values.sum().item())
            self.residual_square_sum += float(
                support_values.square().sum().item())
        self.cosines.extend(float(value) for value in cosine)
        self.unit_distances.extend(float(value) for value in distance)
        self.active_gradient_samples += len(cosine)

    def report(self):
        support_fraction = _safe_ratio(
            self.support_pixels, self.total_pixels)
        full_share = _safe_ratio(
            self.full_support_gradient_sum, self.full_gradient_sum)
        ce_share = _safe_ratio(
            self.ce_support_gradient_sum, self.ce_gradient_sum)
        if self.support_pixels > 0:
            mean_residual = self.residual_sum / self.support_pixels
            variance = self.residual_square_sum / self.support_pixels - \
                mean_residual * mean_residual
            residual_cv = math.sqrt(max(variance, 0.0)) / max(
                mean_residual, 1e-12)
        else:
            mean_residual = None
            residual_cv = None
        return {
            'samples': self.samples,
            'batches': self.batches,
            'total_pixels': self.total_pixels,
            'residual_support_pixels': self.support_pixels,
            'residual_support_fraction': support_fraction,
            'residual_support_mean_weight': mean_residual,
            'residual_support_weight_cv': residual_cv,
            'full_ce_dice_gradient_share': full_share,
            'ce_only_gradient_share': ce_share,
            'active_gradient_samples': self.active_gradient_samples,
            'fractional_vs_binary_gradient_cosine_mean': _mean(self.cosines),
            'fractional_vs_binary_gradient_cosine_median': _median(
                self.cosines),
            'fractional_vs_binary_unit_gradient_l2_mean': _mean(
                self.unit_distances),
        }


class FidelityGateAccumulator:
    def __init__(self):
        self.pixel_correlation = PearsonAccumulator()
        self.mass_correlation = PearsonAccumulator()
        self.samples = 0
        self.union_pixels = 0
        self.exact_support_pixels = 0
        self.proxy_support_pixels = 0
        self.intersection_pixels = 0
        self.proxy_mass = 0.0
        self.proxy_mass_outside_exact = 0.0

    def update(self, exact_residual, proxy_residual):
        exact_support = exact_residual > RESIDUAL_EPSILON
        proxy_support = proxy_residual > RESIDUAL_EPSILON
        union = exact_support | proxy_support
        intersection = exact_support & proxy_support
        self.pixel_correlation.update(
            exact_residual[union], proxy_residual[union])
        exact_mass = exact_residual.flatten(1).sum(dim=1)
        proxy_mass = proxy_residual.flatten(1).sum(dim=1)
        active_mass = (exact_mass > RESIDUAL_EPSILON) | \
            (proxy_mass > RESIDUAL_EPSILON)
        self.mass_correlation.update(
            exact_mass[active_mass], proxy_mass[active_mass])
        self.samples += int(exact_residual.shape[0])
        self.union_pixels += int(union.sum().item())
        self.exact_support_pixels += int(exact_support.sum().item())
        self.proxy_support_pixels += int(proxy_support.sum().item())
        self.intersection_pixels += int(intersection.sum().item())
        self.proxy_mass += float(proxy_residual.sum().item())
        self.proxy_mass_outside_exact += float(
            proxy_residual[~exact_support].sum().item())

    def report(self):
        return {
            'samples': self.samples,
            'union_support_pixels': self.union_pixels,
            'exact_support_pixels': self.exact_support_pixels,
            'proxy_support_pixels': self.proxy_support_pixels,
            'intersection_support_pixels': self.intersection_pixels,
            'union_support_pixel_pearson':
                self.pixel_correlation.correlation(),
            'per_sample_residual_mass_pearson':
                self.mass_correlation.correlation(),
            'support_precision': _safe_ratio(
                self.intersection_pixels, self.proxy_support_pixels),
            'support_recall': _safe_ratio(
                self.intersection_pixels, self.exact_support_pixels),
            'support_iou': _safe_ratio(
                self.intersection_pixels, self.union_pixels),
            'proxy_residual_mass': self.proxy_mass,
            'proxy_residual_mass_outside_exact':
                self.proxy_mass_outside_exact,
            'proxy_residual_mass_outside_exact_fraction': _safe_ratio(
                self.proxy_mass_outside_exact, self.proxy_mass),
        }


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values):
    return statistics.fmean(values) if values else None


def _median(values):
    return statistics.median(values) if values else None


def _resolve_device(device_text):
    if device_text == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but is unavailable; pass --device cpu only '
            'for a deliberately slow diagnostic run')
    return torch.device(device_text)


def _validate_args(args, dataset_length):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            'SliceEq gate checkpoint does not exist: {}'.format(
                args.checkpoint))
    train_list = os.path.join(args.root_path, 'train_slices.list')
    if not os.path.isfile(train_list):
        raise FileNotFoundError(
            'PROMISE12 train_slices.list does not exist: {}'.format(
                train_list))
    if args.labeled_slices <= 0 or args.labeled_slices >= dataset_length:
        raise ValueError('labeled_slices must split the dataset into L and U')
    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if args.max_labeled_batches <= 0 or args.max_unlabeled_batches <= 0:
        raise ValueError('max batch counts must be positive')
    if args.sliceeq_sigma_min <= 0 or \
            args.sliceeq_sigma_max < args.sliceeq_sigma_min:
        raise ValueError('invalid SliceEq sigma range')
    if args.sliceeq_phase_max < args.sliceeq_phase_min:
        raise ValueError('invalid SliceEq phase range')


def _make_loader(dataset, indices, batch_size):
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)


def _profile_generator(device, seed):
    generator_device = 'cuda' if device.type == 'cuda' else 'cpu'
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def _sample_weights(args, count, device, generator):
    offsets = tuple(range(-args.sliceeq_radius, args.sliceeq_radius + 1))
    weights, _, _ = sample_slice_profiles(
        count, offsets,
        (args.sliceeq_sigma_min, args.sliceeq_sigma_max),
        (args.sliceeq_phase_min, args.sliceeq_phase_max),
        device=device, generator=generator)
    return weights


def _infer_reacquired_logits(model, reacquired_image):
    with torch.no_grad():
        logits = _logits(model(reacquired_image))
    return logits.detach().requires_grad_(True)


def _run_labeled(args, model, loader, device, gate1_and_3, gate2):
    profile_generator = _profile_generator(device, args.seed + 1)
    clamped_samples = 0.0
    seen_samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_labeled_batches:
            break
        image_stack = batch['image_stack'].to(device, non_blocking=True)
        label_stack = batch['label_stack'].to(device, non_blocking=True)
        center = image_stack.shape[1] // 2
        weights = _sample_weights(
            args, image_stack.shape[0], device, profile_generator)

        with torch.no_grad():
            reacquired_image, _, exact_occupancy = \
                paired_slice_reacquisition(
                    image_stack, label_stack, weights, args.num_classes)
            proxy_stack = _predict_lcc_stack(model, image_stack)
            _, _, proxy_occupancy = paired_slice_reacquisition(
                image_stack, proxy_stack, weights, args.num_classes)
            exact_residual = acquisition_residual(
                exact_occupancy, label_stack[:, center])
            proxy_residual = acquisition_residual(
                proxy_occupancy, proxy_stack[:, center])
            gate2.update(exact_residual, proxy_residual)

        logits = _infer_reacquired_logits(model, reacquired_image)
        gate1_and_3.update(
            logits, exact_occupancy, label_stack[:, center])
        clamped_samples += float(batch['neighbor_clamped'].sum().item())
        seen_samples += int(image_stack.shape[0])
        print(
            '[labeled] batch {}/{} samples={}'.format(
                batch_index + 1, min(len(loader), args.max_labeled_batches),
                seen_samples), flush=True)
    return _safe_ratio(clamped_samples, seen_samples)


def _run_unlabeled(args, model, loader, device, gate1_and_3):
    profile_generator = _profile_generator(device, args.seed)
    clamped_samples = 0.0
    seen_samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_unlabeled_batches:
            break
        image_stack = batch['image_stack'].to(device, non_blocking=True)
        center = image_stack.shape[1] // 2
        weights = _sample_weights(
            args, image_stack.shape[0], device, profile_generator)
        with torch.no_grad():
            proxy_stack = _predict_lcc_stack(model, image_stack)
            reacquired_image, _, proxy_occupancy = \
                paired_slice_reacquisition(
                    image_stack, proxy_stack, weights, args.num_classes)
        logits = _infer_reacquired_logits(model, reacquired_image)
        gate1_and_3.update(
            logits, proxy_occupancy, proxy_stack[:, center])
        clamped_samples += float(batch['neighbor_clamped'].sum().item())
        seen_samples += int(image_stack.shape[0])
        print(
            '[unlabeled] batch {}/{} samples={}'.format(
                batch_index + 1, min(len(loader), args.max_unlabeled_batches),
                seen_samples), flush=True)
    return _safe_ratio(clamped_samples, seen_samples)


def _gate_decisions(labeled_report, fidelity_report):
    gate1_value = labeled_report['full_ce_dice_gradient_share']
    gate2_pixel = fidelity_report['union_support_pixel_pearson']
    gate2_mass = fidelity_report['per_sample_residual_mass_pearson']
    gate2_outside = fidelity_report[
        'proxy_residual_mass_outside_exact_fraction']
    gate3_value = labeled_report[
        'fractional_vs_binary_gradient_cosine_mean']

    gate1_pass = gate1_value is not None and \
        gate1_value < GATE1_MAX_GRADIENT_SHARE
    gate2_pass = gate2_pixel is not None and gate2_mass is not None and \
        gate2_outside is not None and \
        gate2_pixel >= GATE2_MIN_PIXEL_CORRELATION and \
        gate2_mass >= GATE2_MIN_MASS_CORRELATION and \
        gate2_outside <= GATE2_MAX_OUTSIDE_MASS
    gate3_pass = gate3_value is not None and \
        gate3_value < GATE3_MAX_GRADIENT_COSINE
    joint_pass = gate1_pass and gate2_pass and gate3_pass
    return {
        'gate1_gradient_dilution': {
            'pass': gate1_pass,
            'observed': gate1_value,
            'operator': '<',
            'threshold': GATE1_MAX_GRADIENT_SHARE,
        },
        'gate2_frozen_student_proxy_fidelity': {
            'pass': gate2_pass,
            'provisional_not_ema_teacher': True,
            'observed': {
                'union_support_pixel_pearson': gate2_pixel,
                'per_sample_residual_mass_pearson': gate2_mass,
                'outside_mass_fraction': gate2_outside,
            },
            'thresholds': {
                'minimum_pixel_pearson': GATE2_MIN_PIXEL_CORRELATION,
                'minimum_mass_pearson': GATE2_MIN_MASS_CORRELATION,
                'maximum_outside_mass_fraction': GATE2_MAX_OUTSIDE_MASS,
            },
        },
        'gate3_acquisition_specificity': {
            'pass': gate3_pass,
            'observed': gate3_value,
            'operator': '<',
            'threshold': GATE3_MAX_GRADIENT_COSINE,
        },
        'joint_decision': (
            'provisional_proceed' if joint_pass else 'stop_h7_3'),
    }


def _provenance(args, checkpoint_format, dataset_length, device):
    code_root = Path(__file__).resolve().parent
    workspace_root = code_root.parent
    tracked_files = [
        Path(__file__).resolve(),
        code_root / 'utils' / 'sliceeq_gates.py',
        code_root / 'utils' / 'sliceeq.py',
        code_root / 'utils' / 'sliceeq_occ.py',
        code_root / 'dataloaders' / 'sliceeq_dataset.py',
        code_root / 'train_sliceeq_occ.py',
        workspace_root / 'research' / 'experiments' /
        'h7_slice_profile_reacquisition' / 'h7_3_gate_protocol.md',
        Path(args.root_path) / 'train_slices.list',
        Path(args.checkpoint),
    ]
    hashes = {}
    for path in tracked_files:
        resolved = path.resolve()
        hashes[str(resolved)] = _sha256(resolved)
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_kind': 'read_only_no_training',
        'checkpoint_format': checkpoint_format,
        'checkpoint_contains_ema_teacher': False,
        'gate2_evidence_source': 'frozen_student_proxy_in_eval_mode',
        'dataset_length': dataset_length,
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'file_sha256': hashes,
        'arguments': vars(args),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)

    labeled_dataset = SliceStackDataSets(
        base_dir=args.root_path,
        transform=StackRandomGenerator(args.patch_size),
        radius=args.sliceeq_radius)
    unlabeled_dataset = ImageOnlySliceStackDataSets(
        base_dir=args.root_path,
        transform=StackRandomGenerator(args.patch_size),
        radius=args.sliceeq_radius)
    _validate_args(args, len(labeled_dataset))
    if len(unlabeled_dataset) != len(labeled_dataset):
        raise RuntimeError('labeled and image-only dataset indices differ')
    labeled_indices = list(range(args.labeled_slices))
    unlabeled_indices = list(range(
        args.labeled_slices, len(unlabeled_dataset)))
    labeled_loader = _make_loader(
        labeled_dataset, labeled_indices, args.batch_size)
    unlabeled_loader = _make_loader(
        unlabeled_dataset, unlabeled_indices, args.batch_size)

    model, checkpoint_format = _load_model(args.checkpoint, device)
    labeled_gradient_gate = GradientGateAccumulator()
    unlabeled_gradient_gate = GradientGateAccumulator()
    fidelity_gate = FidelityGateAccumulator()

    print('Running Gate 1/2/3 on frozen checkpoint: {}'.format(
        args.checkpoint), flush=True)
    labeled_clamped = _run_labeled(
        args, model, labeled_loader, device,
        labeled_gradient_gate, fidelity_gate)
    unlabeled_clamped = _run_unlabeled(
        args, model, unlabeled_loader, device, unlabeled_gradient_gate)

    labeled_report = labeled_gradient_gate.report()
    unlabeled_report = unlabeled_gradient_gate.report()
    fidelity_report = fidelity_gate.report()
    decisions = _gate_decisions(labeled_report, fidelity_report)
    report = {
        'schema_version': 1,
        'provenance': _provenance(
            args, checkpoint_format, len(labeled_dataset), device),
        'locked_definitions': {
            'residual': '0.5 * L1(occupancy, center_one_hot)',
            'support_epsilon': RESIDUAL_EPSILON,
            'gate1_primary': 'exact_gt_labeled_full_ce_dice_gradient_share',
            'gate2_primary': 'frozen_student_proxy_on_labeled_stacks',
            'gate3_primary': 'exact_gt_labeled_fractional_vs_binary_soft_ce',
        },
        'gate1_and_gate3_exact_gt_labeled': labeled_report,
        'gate1_and_gate3_unlabeled_proxy_descriptive': unlabeled_report,
        'gate2_frozen_student_proxy_fidelity': fidelity_report,
        'neighbor_clamped_sample_fraction': {
            'labeled': labeled_clamped,
            'unlabeled': unlabeled_clamped,
        },
        'decisions': decisions,
        'interpretation_warning': (
            'The saved iteration-23000 checkpoint has no EMA state. Gate 2 '
            'uses the frozen student in eval mode and is provisional evidence, '
            'not a direct teacher measurement.'),
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary_path, output_path)
    print(json.dumps(decisions, ensure_ascii=False, indent=2), flush=True)
    print('Wrote gate report: {}'.format(output_path), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
