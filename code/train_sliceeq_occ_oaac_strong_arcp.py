"""SliceEqOcc-OAAC-Strong with H5-only axial-response profile calibration.

The final Strong parent remains unchanged.  This successor changes only the
three profile weights immediately before the existing paired image/occupancy
operator.  It does not estimate scanner PSF or slice thickness.
"""

import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn

from utils.sliceeq_arcp import (
    calibrate_profile_weights, compute_patient_balanced_reference,
    reference_tensor, save_reference_report)


EXPECTED_STRONG_TRAIN_SHA256 = (
    'b3219557df39aee680b5bc055b9ff6dc88f5acfa03e8ed1fa5042aacade5bf60')
ARCP_REFERENCE_FILENAME = 'arcp_reference.json'
_ORIGINAL_ARGV = list(sys.argv)


def _inject_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


_inject_default('--exp', 'SliceEqOccOAACStrongARCP_PROMISE12')

import train_sliceeq_occ_oaac_strong as strong  # noqa: E402


args = strong.args


class _ARCPContext:
    def __init__(self):
        self.reference = None
        self.paired_calls = 0
        self.diagnostic_calls = 0
        self.pending = []
        self.activity_logging_enabled = False

    def reset(self, reference, activity_logging_enabled=True):
        if reference.shape != (2, 2):
            raise ValueError('ARCP runtime reference must be 2x2')
        self.reference = reference.detach()
        self.paired_calls = 0
        self.diagnostic_calls = 0
        self.pending = []
        self.activity_logging_enabled = activity_logging_enabled

    def assert_complete(self, expected_paired_calls=None):
        if self.paired_calls != self.diagnostic_calls or self.pending:
            raise RuntimeError(
                'ARCP parent call sequence incomplete: paired={}, '
                'diagnostics={}, pending={}'.format(
                    self.paired_calls, self.diagnostic_calls,
                    len(self.pending)))
        if self.paired_calls % 2 != 0:
            raise RuntimeError('ARCP parent call sequence is not L/U paired')
        if expected_paired_calls is not None and \
                self.paired_calls != expected_paired_calls:
            raise RuntimeError(
                'ARCP expected {} paired calls; observed {}'.format(
                    expected_paired_calls, self.paired_calls))


_context = _ARCPContext()


def _arcp_oaac_paired_reacquisition(
        image_stack, hard_target_stack, weights, num_classes):
    if _context.reference is None:
        raise RuntimeError('ARCP reference was not initialized')
    if len(_context.pending) >= 2:
        raise RuntimeError('ARCP received acquisition before diagnostics')
    calibrated, metadata = calibrate_profile_weights(
        image_stack, weights, _context.reference)
    _context.pending.append((calibrated, metadata))
    _context.paired_calls += 1
    return strong._oaac_paired_reacquisition(
        image_stack, hard_target_stack, calibrated, num_classes)


def _arcp_oaac_reacquisition_diagnostics(
        *diagnostic_args, **diagnostic_kwargs):
    if not _context.pending:
        raise RuntimeError('ARCP diagnostics have no calibrated profile')
    if len(diagnostic_args) < 7:
        raise RuntimeError('ARCP diagnostics signature changed')
    calibrated, metadata = _context.pending.pop(0)
    replaced = list(diagnostic_args)
    replaced[4] = calibrated
    diagnostics = strong._oaac_reacquisition_diagnostics(
        *replaced, **diagnostic_kwargs)
    diagnostics.update(metadata)
    _context.diagnostic_calls += 1
    if _context.activity_logging_enabled and \
            _context.diagnostic_calls % 2 == 0:
        completed_steps = _context.diagnostic_calls // 2
        if completed_steps == 1 or completed_steps % 200 == 0:
            logging.info(
                'ARCP iteration %d: alpha(mean/std/absdev)=%.6f/%.6f/%.6f; '
                'active=%.6f; bounds(low/high)=%.6f/%.6f; '
                'center(parent/calibrated)=%.6f/%.6f; '
                'effect(before/after/reference)=%.6f/%.6f/%.6f',
                1000 + completed_steps,
                metadata['arcp_alpha_mean'].item(),
                metadata['arcp_alpha_std'].item(),
                metadata['arcp_abs_alpha_minus_one_mean'].item(),
                metadata['arcp_active_sample_fraction'].item(),
                metadata['arcp_lower_bound_fraction'].item(),
                metadata['arcp_upper_bound_fraction'].item(),
                metadata['arcp_parent_center_weight_mean'].item(),
                metadata['arcp_calibrated_center_weight_mean'].item(),
                metadata['arcp_effect_before_mean'].item(),
                metadata['arcp_effect_after_mean'].item(),
                metadata['arcp_effect_reference_mean'].item())
    return diagnostics


def _configure_parent(flags):
    strong._configure_parent(flags)
    strong.parent.paired_slice_reacquisition = \
        _arcp_oaac_paired_reacquisition
    strong.parent.reacquisition_diagnostics = \
        _arcp_oaac_reacquisition_diagnostics


def _validate_args(flags):
    strong.parent._validate_args(flags)
    locked_recipe = {
        'root_path': (
            '/home/aiteam/zhengtaoma/Baseline/data/'
            'PROMISE12_h5_training_source'),
        'exp': 'SliceEqOccOAACStrongARCP_PROMISE12',
        'model': 'unet',
        'max_iterations': 30000,
        'batch_size': 24,
        'deterministic': 1,
        'base_lr': 0.01,
        'patch_size': [256, 256],
        'seed': 1337,
        'num_classes': 2,
        'labeled_bs': 12,
        'labelnum': 7,
        'ema_decay': 0.99,
        'consistency_type': 'mse',
        'consistency': 0.1,
        'consistency_rampup': 200.0,
        'sliceeq_radius': 1,
        'sliceeq_sigma_min': 0.45,
        'sliceeq_sigma_max': 0.85,
        'sliceeq_phase_min': -0.25,
        'sliceeq_phase_max': 0.25,
    }
    for name, expected in locked_recipe.items():
        actual = getattr(flags, name)
        if name == 'patch_size':
            actual = list(actual)
        if actual != expected:
            raise ValueError(
                'H7.18 ARCP locks --{}={!r}; received {!r}'.format(
                    name, expected, actual))


def _run_arcp_smoke(device, reference):
    base = torch.arange(
        2 * 3 * 1 * 7 * 8, dtype=torch.float32,
        device=device).reshape(2, 3, 1, 7, 8) / 100.0
    base[1, 0] = base[1, 1]  # Explicit duplicated endpoint/support sample.
    weights = torch.tensor(
        [[0.20, 0.60, 0.20], [0.10, 0.70, 0.20]],
        dtype=base.dtype, device=device)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()
    calibrated, metadata = calibrate_profile_weights(
        base, weights, reference)
    repeated, repeated_metadata = calibrate_profile_weights(
        base, weights, reference)
    if not torch.equal(calibrated, repeated):
        raise RuntimeError('ARCP smoke is not deterministic')
    if not torch.allclose(
            calibrated.sum(dim=1), torch.ones(2, device=device),
            atol=1e-6, rtol=1e-6):
        raise RuntimeError('ARCP smoke weights do not sum to one')
    if (calibrated < 0.0).any():
        raise RuntimeError('ARCP smoke produced negative weights')
    if not torch.equal(calibrated[1], weights[1]):
        raise RuntimeError('ARCP smoke changed duplicated endpoint support')
    for name, value in metadata.items():
        if not torch.isfinite(value):
            raise RuntimeError(
                'ARCP smoke produced non-finite {}'.format(name))
        if not torch.equal(value, repeated_metadata[name]):
            raise RuntimeError(
                'ARCP smoke metadata is not deterministic: {}'.format(name))
    if not torch.equal(cpu_before, torch.random.get_rng_state()) or \
            not torch.equal(cuda_before, torch.cuda.get_rng_state(device)):
        raise RuntimeError('ARCP smoke consumed a parent-visible RNG stream')
    logging.info(
        'ARCP CUDA smoke passed: deterministic convex weights, duplicate '
        'support identity, finite diagnostics and parent RNG unchanged')


if __name__ == '__main__':
    _validate_args(args)
    strong_hash = strong.parent.locked._sha256(strong.__file__)
    if strong_hash != EXPECTED_STRONG_TRAIN_SHA256:
        raise RuntimeError(
            'ARCP Strong source changed: expected {}; got {}'.format(
                EXPECTED_STRONG_TRAIN_SHA256, strong_hash))
    parent_hash = strong.parent.locked._sha256(strong.parent.__file__)
    if parent_hash != strong.EXPECTED_PARENT_TRAIN_SHA256:
        raise RuntimeError(
            'ARCP base source changed: expected {}; got {}'.format(
                strong.EXPECTED_PARENT_TRAIN_SHA256, parent_hash))
    pretrained_checkpoint = strong.parent.locked._resolve_pretrained_checkpoint(
        args.pretrained_checkpoint)
    checkpoint_hash = strong.parent.locked._sha256(pretrained_checkpoint)
    if checkpoint_hash != strong.EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(
            'ARCP requires shared Pre10000 SHA-256 {}; got {}'.format(
                strong.EXPECTED_PRETRAINED_SHA256, checkpoint_hash))
    dataset_report = strong.validate_promise12_root(
        args.root_path, strict_split=True, check_hdf5=True)
    print('PROMISE12 preflight: {}'.format(dataset_report))

    cudnn.benchmark = False
    cudnn.deterministic = True
    strong.parent.locked._reset_stage_rng(args.seed)

    snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info(
        '========== START SliceEqOccOAACStrong-ARCP SELF-TRAINING ==========')
    logging.info(str(args))

    reference_report = compute_patient_balanced_reference(
        args.root_path, output_size=tuple(args.patch_size))
    reference_path = os.path.join(snapshot_path, ARCP_REFERENCE_FILENAME)
    save_reference_report(reference_report, reference_path)
    reference = reference_tensor(
        reference_report, device=torch.device('cuda'), dtype=torch.float32)
    logging.info(
        'ARCP image-only reference: patients=%d, interior_stacks=%d, '
        'matrix=%s, train-list-sha256=%s',
        reference_report['patient_count'],
        reference_report['eligible_interior_stacks'],
        reference_report['reference_matrix'],
        reference_report['train_slices_sha256'])
    logging.info('ARCP reference artifact: %s', reference_path)
    logging.info(
        'ARCP changes only three profile weights; Strong OAAC, batch36, '
        'EMA/LR/ramp/loss/validation/inference and 1000-step archives remain')

    strong._run_cuda_smoke(torch.device('cuda'))
    _run_arcp_smoke(torch.device('cuda'), reference)
    appearance_generator = torch.Generator(device='cuda')
    appearance_generator.manual_seed(strong.APPEARANCE_SEED)
    strong._context.reset(appearance_generator)
    _context.reset(reference)
    _configure_parent(args)
    result = strong.parent.self_train(
        args, pretrained_checkpoint, snapshot_path)
    expected = 2 * (args.max_iterations - 1000)
    strong._context.assert_complete(expected_paired_calls=expected)
    _context.assert_complete(expected_paired_calls=expected)
    logging.info(
        'ARCP/OAAC call contracts complete: paired=%d, diagnostics=%d',
        _context.paired_calls, _context.diagnostic_calls)
    print(result)

