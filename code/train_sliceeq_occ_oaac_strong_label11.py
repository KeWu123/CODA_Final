"""Run the frozen SliceEqOcc-OAAC-Strong method with 11 labeled cases.

The method, optimizer, EMA policy, profile, OAAC parameters, validation, and
1000-iteration periodic archive rule are unchanged from the final 7-label
entry. Only the labeled budget changes from 7/191 to 11/306, and a matching
explicit 11-label ``net`` + ``opt`` pretraining checkpoint is required.
"""

import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn


LABELNUM = 11
LABELED_SLICES = 306
_ORIGINAL_ARGV = list(sys.argv)


def _inject_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


_inject_default('--exp', 'SliceEqOccOAACStrong_PROMISE12')
_inject_default('--labelnum', LABELNUM)

import train_sliceeq_occ_oaac_strong as strong  # noqa: E402


args = strong.args


def _read_list(root_path, filename):
    path = os.path.join(root_path, filename)
    with open(path, 'r', encoding='utf-8-sig') as stream:
        return [line.strip() for line in stream if line.strip()]


def _case_name(slice_name):
    if '_slice_' not in slice_name:
        raise ValueError('Unexpected PROMISE12 slice name: {}'.format(
            slice_name))
    return slice_name.split('_slice_', 1)[0]


def _validate_label11_boundary(root_path):
    train_cases = _read_list(root_path, 'train.list')
    train_slices = _read_list(root_path, 'train_slices.list')
    expected_cases = set(train_cases[:LABELNUM])
    selected_slices = train_slices[:LABELED_SLICES]
    selected_cases = {_case_name(item) for item in selected_slices}
    if len(train_cases) < LABELNUM or len(selected_slices) != LABELED_SLICES:
        raise RuntimeError('PROMISE12 does not contain the locked 11-label budget')
    if selected_cases != expected_cases:
        raise RuntimeError(
            'The first {} slices do not exactly match the first {} cases'.format(
                LABELED_SLICES, LABELNUM))
    if any(_case_name(item) in expected_cases
           for item in train_slices[LABELED_SLICES:]):
        raise RuntimeError(
            'An 11-label case crosses the locked boundary {}'.format(
                LABELED_SLICES))
    return {
        'labelnum': LABELNUM,
        'labeled_slices': LABELED_SLICES,
        'labeled_cases': train_cases[:LABELNUM],
    }


def _validate_args(flags):
    strong.parent._validate_args(flags)
    locked = {
        'root_path': (
            '/home/aiteam/zhengtaoma/Baseline/data/'
            'PROMISE12_h5_training_source'),
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
        'labelnum': LABELNUM,
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
    for name, expected in locked.items():
        actual = getattr(flags, name)
        if name == 'patch_size':
            actual = list(actual)
        if actual != expected:
            raise ValueError(
                'Label-11 Strong locks --{}={!r}; received {!r}'.format(
                    name, expected, actual))
    if '--pretrained_checkpoint' not in _ORIGINAL_ARGV:
        raise ValueError(
            'Label-11 training requires an explicit --pretrained_checkpoint '
            'produced from the same first 11 labeled cases; the 7-label '
            'default checkpoint is forbidden')


if __name__ == '__main__':
    _validate_args(args)
    parent_hash = strong.parent.locked._sha256(strong.parent.__file__)
    if parent_hash != strong.EXPECTED_PARENT_TRAIN_SHA256:
        raise RuntimeError(
            'Strong parent source changed: expected {}; got {}'.format(
                strong.EXPECTED_PARENT_TRAIN_SHA256, parent_hash))

    pretrained_checkpoint = strong.parent.locked._resolve_pretrained_checkpoint(
        args.pretrained_checkpoint)
    checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'net' not in checkpoint or \
            'opt' not in checkpoint:
        raise RuntimeError(
            'Label-11 pretrain checkpoint must contain both `net` and `opt`')
    checkpoint_hash = strong.parent.locked._sha256(pretrained_checkpoint)

    dataset_report = strong.validate_promise12_root(
        args.root_path, strict_split=True, check_hdf5=True)
    label_report = _validate_label11_boundary(args.root_path)
    if strong.parent.base.patients_to_slices(
            args.root_path, LABELNUM) != LABELED_SLICES:
        raise RuntimeError('patients_to_slices no longer maps 11 cases to 306')
    print('PROMISE12 preflight: {}'.format(dataset_report))
    print('PROMISE12 label-11 boundary: {}'.format(label_report))

    cudnn.benchmark = False
    cudnn.deterministic = True
    strong.parent.locked._reset_stage_rng(args.seed)
    strong._configure_parent(args)

    snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info(
        '========== START SliceEqOccOAACStrong LABEL-11 SELF-TRAINING ==========')
    logging.info(str(args))
    logging.info('Label budget: first 11 train cases / 306 slices')
    logging.info('Explicit label-11 pretrain: %s', pretrained_checkpoint)
    logging.info('Explicit label-11 pretrain SHA-256: %s', checkpoint_hash)
    logging.info(
        'Method frozen: OAAC scale1.25, batch36, EMA/LR/ramp unchanged; '
        'periodic raw-student checkpoints every 1000 iterations')

    strong._run_cuda_smoke(torch.device('cuda'))
    appearance_generator = torch.Generator(device='cuda')
    appearance_generator.manual_seed(strong.APPEARANCE_SEED)
    strong._context.reset(appearance_generator)
    result = strong.parent.self_train(
        args, pretrained_checkpoint, snapshot_path)
    strong._context.assert_complete(
        expected_paired_calls=2 * (args.max_iterations - 1000))
    logging.info(
        'OAAC parent call contract complete: paired=%d, diagnostics=%d',
        strong._context.paired_calls, strong._context.diagnostic_calls)
    print(result)
