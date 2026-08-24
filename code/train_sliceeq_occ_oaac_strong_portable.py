"""Portable H7.15 Strong-all OAAC on the frozen SliceEqOcc recipe.

Only the post-warmup unlabeled re-acquired student image receives the monotonic
coordinate-preserving appearance transform. OAAC ranges are jointly scaled by
1.25. Validation and best-model rules remain the original SliceEqOcc rules;
only ordinary periodic raw-student archiving changes from 3000 to 1000 steps.
"""

import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn

from utils.promise12_preflight import validate_promise12_root
from utils.sliceeq_oaac_strong import ordered_appearance_transform


EXPECTED_PRETRAINED_SHA256 = (
    '49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9')
EXPECTED_PARENT_TRAIN_SHA256 = (
    '001d130576df8f669e7ea4c1fec01a362f10194c91b1c615e07b9bd4762fcc4d')
APPEARANCE_SEED = 1339


def _import_parent():
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        import train_sliceeq_occ_h7_15_base as parent
    finally:
        sys.argv = original_argv
    return parent


parent = _import_parent()
parent.parser.set_defaults(exp='SliceEqOccOAACStrong_PROMISE12')
args = parent.parser.parse_args()

_parent_paired_reacquisition = parent.paired_slice_reacquisition
_parent_reacquisition_diagnostics = parent.reacquisition_diagnostics


class _OAACContext:
    def __init__(self):
        self.generator = None
        self.paired_calls = 0
        self.diagnostic_calls = 0
        self.pending_metadata = None
        self.activity_logging_enabled = False

    def reset(self, generator, activity_logging_enabled=True):
        self.generator = generator
        self.paired_calls = 0
        self.diagnostic_calls = 0
        self.pending_metadata = None
        self.activity_logging_enabled = activity_logging_enabled

    def assert_complete(self, expected_paired_calls=None):
        if self.paired_calls != self.diagnostic_calls:
            raise RuntimeError(
                'OAAC parent call sequence ended mid-step: paired={}, '
                'diagnostics={}'.format(
                    self.paired_calls, self.diagnostic_calls))
        if self.paired_calls % 2 != 0 or self.pending_metadata is not None:
            raise RuntimeError('OAAC parent call sequence ended incomplete')
        if expected_paired_calls is not None and \
                self.paired_calls != expected_paired_calls:
            raise RuntimeError(
                'OAAC expected {} paired calls; observed {}'.format(
                    expected_paired_calls, self.paired_calls))


_context = _OAACContext()


def _oaac_paired_reacquisition(image_stack, hard_target_stack, weights,
                               num_classes):
    if _context.generator is None:
        raise RuntimeError('OAAC appearance generator was not initialized')
    next_call = _context.paired_calls + 1
    if next_call % 2 == 1:
        if _context.paired_calls != _context.diagnostic_calls or \
                _context.pending_metadata is not None:
            raise RuntimeError(
                'OAAC expected the labeled re-acquisition at step start')
    elif _context.paired_calls != _context.diagnostic_calls + 1 or \
            _context.pending_metadata is not None:
        raise RuntimeError(
            'OAAC expected the unlabeled re-acquisition after labeled')

    clean_image, hard_target, occupancy = _parent_paired_reacquisition(
        image_stack, hard_target_stack, weights, num_classes)
    _context.paired_calls += 1
    # Parent order is exactly labeled then unlabeled once per post-warmup step.
    if _context.paired_calls % 2 == 1:
        return clean_image, hard_target, occupancy
    strong_image, metadata = ordered_appearance_transform(
        clean_image, generator=_context.generator)
    _context.pending_metadata = metadata
    return strong_image, hard_target, occupancy


def _oaac_reacquisition_diagnostics(*diagnostic_args, **diagnostic_kwargs):
    next_call = _context.diagnostic_calls + 1
    if next_call % 2 == 1:
        if _context.paired_calls != _context.diagnostic_calls + 2 or \
                _context.pending_metadata is None:
            raise RuntimeError(
                'OAAC expected labeled diagnostics after both acquisitions')
    elif _context.paired_calls != _context.diagnostic_calls + 1 or \
            _context.pending_metadata is None:
        raise RuntimeError(
            'OAAC expected unlabeled diagnostics after labeled diagnostics')

    diagnostics = _parent_reacquisition_diagnostics(
        *diagnostic_args, **diagnostic_kwargs)
    _context.diagnostic_calls += 1
    if _context.diagnostic_calls % 2 == 0:
        if _context.pending_metadata is None:
            raise RuntimeError('OAAC unlabeled diagnostics have no metadata')
        metadata = _context.pending_metadata
        diagnostics.update(metadata)
        completed_steps = _context.diagnostic_calls // 2
        if _context.activity_logging_enabled and (
                completed_steps == 1 or completed_steps % 200 == 0):
            logging.info(
                'OAAC appearance iteration %d: '
                'abs(log_gamma/log_contrast/brightness)=%.6f/%.6f/%.6f; '
                'normalized_abs_change=%.6f; active_samples=%.6f; '
                'outside(low/high)=%.6f/%.6f',
                1000 + completed_steps,
                metadata['appearance_abs_log_gamma_mean'].item(),
                metadata['appearance_abs_log_contrast_mean'].item(),
                metadata[
                    'appearance_abs_brightness_fraction_mean'].item(),
                metadata['appearance_normalized_absolute_change'].item(),
                metadata['appearance_active_sample_fraction'].item(),
                metadata['appearance_below_source_min_fraction'].item(),
                metadata['appearance_above_source_max_fraction'].item())
        _context.pending_metadata = None
    return diagnostics


def _validate_args(flags):
    parent._validate_args(flags)
    # Absolute storage locations differ across workstations. Dataset identity
    # is still enforced by the strict PROMISE12 preflight in the main block.
    locked_recipe = {
        'exp': 'SliceEqOccOAACStrong_PROMISE12',
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
                'H7.15 Strong-all locks --{}={!r}; received {!r}'.format(
                    name, expected, actual))


def _configure_parent(flags):
    parent.args = flags
    parent.locked.args = flags
    parent.base.args = flags
    if hasattr(parent.locked, 'base'):
        parent.locked.base.args = flags
    parent.paired_slice_reacquisition = _oaac_paired_reacquisition
    parent.reacquisition_diagnostics = _oaac_reacquisition_diagnostics


def _run_cuda_smoke(device):
    """Fail closed before training if the OAAC branch violates its contract."""
    height, width = 6, 7
    base_image = torch.linspace(
        -0.75, 1.25, height * width, device=device).reshape(
            1, 1, 1, height, width)
    slice_offsets = torch.tensor(
        [-0.10, 0.0, 0.15], device=device).reshape(1, 3, 1, 1, 1)
    image_stack = (base_image + slice_offsets).repeat(2, 1, 1, 1, 1)
    target_stack = torch.zeros(
        (2, 3, height, width), dtype=torch.long, device=device)
    target_stack[:, 0, 2:4, 2:5] = 1
    target_stack[:, 1, 2:5, 2:6] = 1
    target_stack[:, 2, 1:5, 2:6] = 1
    weights = torch.tensor(
        [[0.20, 0.60, 0.20], [0.30, 0.50, 0.20]],
        dtype=image_stack.dtype, device=device)

    expected_l = _parent_paired_reacquisition(
        image_stack, target_stack, weights, args.num_classes)
    expected_u = _parent_paired_reacquisition(
        image_stack, target_stack, weights, args.num_classes)
    fractional = torch.logical_and(expected_u[2] > 0.0, expected_u[2] < 1.0)
    if not fractional.any():
        raise RuntimeError('OAAC smoke target lacks fractional occupancy')
    smoke_generator = torch.Generator(device=device)
    smoke_generator.manual_seed(APPEARANCE_SEED)
    _context.reset(smoke_generator, activity_logging_enabled=False)

    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone()
    appearance_rng_before_l = smoke_generator.get_state().clone()
    observed_l = _oaac_paired_reacquisition(
        image_stack, target_stack, weights, args.num_classes)
    if not all(torch.equal(left, right) for left, right in zip(
            observed_l, expected_l)):
        raise RuntimeError('OAAC smoke changed the labeled branch')
    if not torch.equal(appearance_rng_before_l, smoke_generator.get_state()):
        raise RuntimeError('OAAC labeled branch consumed appearance RNG')

    observed_u = _oaac_paired_reacquisition(
        image_stack, target_stack, weights, args.num_classes)
    if not torch.equal(observed_u[1], expected_u[1]) or \
            not torch.equal(observed_u[2], expected_u[2]):
        raise RuntimeError('OAAC smoke changed an unlabeled target')
    if torch.equal(observed_u[0], expected_u[0]):
        raise RuntimeError('OAAC smoke produced an identity U image')
    if not torch.isfinite(observed_u[0]).all():
        raise RuntimeError('OAAC smoke produced a non-finite U image')
    if not torch.all(
            observed_u[0].flatten(1)[:, 1:] >
            observed_u[0].flatten(1)[:, :-1]):
        raise RuntimeError('OAAC smoke violated monotonic intensity order')

    sigma = torch.ones(2, dtype=image_stack.dtype, device=device)
    phase = torch.zeros(2, dtype=image_stack.dtype, device=device)
    _oaac_reacquisition_diagnostics(
        image_stack, target_stack, observed_l[0], observed_l[1],
        weights, sigma, phase)
    _oaac_reacquisition_diagnostics(
        image_stack, target_stack, observed_u[0], observed_u[1],
        weights, sigma, phase)
    _context.assert_complete(expected_paired_calls=2)

    first_generator = torch.Generator(device=device)
    second_generator = torch.Generator(device=device)
    other_generator = torch.Generator(device=device)
    first_generator.manual_seed(APPEARANCE_SEED)
    second_generator.manual_seed(APPEARANCE_SEED)
    other_generator.manual_seed(APPEARANCE_SEED + 1)
    first, _ = ordered_appearance_transform(
        expected_u[0], generator=first_generator)
    repeated, _ = ordered_appearance_transform(
        expected_u[0], generator=second_generator)
    different, _ = ordered_appearance_transform(
        expected_u[0], generator=other_generator)
    if not torch.equal(first, repeated) or torch.equal(first, different):
        raise RuntimeError('OAAC smoke failed independent-RNG reproducibility')
    if not torch.equal(cpu_rng_before, torch.random.get_rng_state()) or \
            not torch.equal(cuda_rng_before, torch.cuda.get_rng_state(device)):
        raise RuntimeError('OAAC smoke consumed a parent-visible RNG stream')
    logging.info(
        'OAAC CUDA smoke passed: L/targets exact, U active/monotonic/finite, '
        'independent RNG reproducible and parent-visible RNG unchanged')


if __name__ == '__main__':
    _validate_args(args)
    parent_hash = parent.locked._sha256(parent.__file__)
    if parent_hash != EXPECTED_PARENT_TRAIN_SHA256:
        raise RuntimeError(
            'H7.15 parent train source changed: expected {}; got {}'.format(
                EXPECTED_PARENT_TRAIN_SHA256, parent_hash))
    pretrained_checkpoint = parent.locked._resolve_pretrained_checkpoint(
        args.pretrained_checkpoint)
    checkpoint_hash = parent.locked._sha256(pretrained_checkpoint)
    if checkpoint_hash != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(
            'H7.15 requires shared Pre10000 SHA-256 {}; got {}'.format(
                EXPECTED_PRETRAINED_SHA256, checkpoint_hash))
    dataset_report = validate_promise12_root(
        args.root_path, strict_split=True, check_hdf5=True)
    print('PROMISE12 preflight: {}'.format(dataset_report))

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    parent.locked._reset_stage_rng(args.seed)
    _configure_parent(args)

    snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info(
        '============= START SliceEqOccOAACStrong SELF-TRAINING =============')
    logging.info(str(args))
    logging.info(
        'OAAC Strong-all post-acquisition U appearance: '
        'log-gamma=[-0.25,0.25], log-contrast=[-0.1875,0.1875], '
        'brightness-span=[-0.125,0.125], probability=1.0, seed=%d',
        APPEARANCE_SEED)
    logging.info(
        'H7.15 periodic raw-student checkpoints: every 1000 iterations; '
        'validation and best-model selection unchanged')
    logging.info(
        'OAAC keeps L branches, occupancy targets, student batch36, teacher, '
        'loss, validation and inference unchanged')
    _run_cuda_smoke(torch.device('cuda'))
    appearance_generator = torch.Generator(device='cuda')
    appearance_generator.manual_seed(APPEARANCE_SEED)
    _context.reset(appearance_generator)
    training_result = parent.self_train(
        args, pretrained_checkpoint, snapshot_path)
    _context.assert_complete(
        expected_paired_calls=2 * (args.max_iterations - 1000))
    logging.info(
        'OAAC parent call contract complete: paired=%d, diagnostics=%d',
        _context.paired_calls, _context.diagnostic_calls)
    print(training_result)
