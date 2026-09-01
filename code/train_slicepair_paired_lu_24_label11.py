"""Run the paired-L/U 24-view control with 11 labeled PROMISE12 cases.

The method and optimization recipe are identical to ``paired_lu_24``:
6 native labeled, 6 paired re-acquired labeled, and 12 paired re-acquired
unlabeled student views after the inherited 1k identity warm-up. Only the
labeled training pool changes from 7/191 to 11/306. A matching label-11
Pre10000 checkpoint containing both ``net`` and ``opt`` is mandatory.
"""

import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn


LABELNUM = 11
LABELED_SLICES = 306
EXPERIMENT = (
    'SlicePairPairedLU24_35_5_10_Pre10000_Self30000_'
    'label11_seed1337')
_ORIGINAL_ARGV = list(sys.argv)


def _inject_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


_inject_default('--exp', EXPERIMENT)
_inject_default('--labelnum', LABELNUM)
_inject_default('--occ_ablation', 'paired_lu_24')
_inject_default('--appearance_mode', 'oaac_strong')

import train_sliceeq_occ_ablation as parent  # noqa: E402


args = parent.args


def _read_list(root_path, filename):
    path = os.path.join(root_path, filename)
    with open(path, 'r', encoding='utf-8-sig') as stream:
        return [line.strip() for line in stream if line.strip()]


def _case_name(slice_name):
    if '_slice_' not in slice_name:
        raise ValueError(
            'Unexpected PROMISE12 slice name: {}'.format(slice_name))
    return slice_name.split('_slice_', 1)[0]


def _validate_label11_boundary(root_path):
    """Verify that the first 306 slices are exactly the first 11 cases."""
    train_cases = _read_list(root_path, 'train.list')
    train_slices = _read_list(root_path, 'train_slices.list')
    if len(train_cases) < LABELNUM or len(train_slices) < LABELED_SLICES:
        raise RuntimeError(
            'PROMISE12 does not contain the locked 11-label budget')

    expected_cases = set(train_cases[:LABELNUM])
    selected_slices = train_slices[:LABELED_SLICES]
    selected_cases = {_case_name(item) for item in selected_slices}
    if selected_cases != expected_cases:
        raise RuntimeError(
            'The first {} slices do not exactly match the first {} cases'
            .format(LABELED_SLICES, LABELNUM))
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
    parent._validate_args(flags)
    locked_recipe = {
        'exp': EXPERIMENT,
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
        'occ_ablation': 'paired_lu_24',
        'appearance_mode': 'oaac_strong',
    }
    for name, expected in locked_recipe.items():
        actual = getattr(flags, name)
        if name == 'patch_size':
            actual = list(actual)
        if actual != expected:
            raise ValueError(
                'Label-11 paired-L/U-24 locks --{}={!r}; received {!r}'
                .format(name, expected, actual))
    if '--pretrained_checkpoint' not in _ORIGINAL_ARGV:
        raise ValueError(
            'Label-11 training requires an explicit matching '
            '--pretrained_checkpoint; the label-7 default is forbidden')


def _validate_pretrained_checkpoint(path):
    resolved = parent.locked._resolve_pretrained_checkpoint(path)
    normalized = resolved.replace('\\', '/').lower()
    if not any(marker in normalized for marker in (
            'label11', '11_labeled', '11label')):
        raise RuntimeError(
            'Refusing a checkpoint path without a label-11 marker: {}'.format(
                resolved))
    checkpoint = torch.load(resolved, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'net' not in checkpoint or \
            'opt' not in checkpoint:
        raise RuntimeError(
            'Label-11 Pre10000 checkpoint must contain both `net` and `opt`')
    return resolved


if __name__ == '__main__':
    _validate_args(args)
    pretrained_checkpoint = _validate_pretrained_checkpoint(
        args.pretrained_checkpoint)
    dataset_report = parent.validate_promise12_root(
        args.root_path, strict_split=True, check_hdf5=True)
    label_report = _validate_label11_boundary(args.root_path)
    if parent.base.patients_to_slices(
            args.root_path, LABELNUM) != LABELED_SLICES:
        raise RuntimeError(
            'patients_to_slices no longer maps 11 cases to 306 slices')
    print('PROMISE12 preflight: {}'.format(dataset_report))
    print('PROMISE12 label-11 boundary: {}'.format(label_report))

    cudnn.benchmark = False
    cudnn.deterministic = True
    parent.locked._reset_stage_rng(args.seed)

    snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info(
        '======= START SlicePair paired-L/U-24 LABEL-11 SELF-TRAINING =======')
    logging.info(str(args))
    logging.info('Label budget: first 11 train cases / 306 slices')
    logging.info('Explicit label-11 pretrain: %s', pretrained_checkpoint)
    logging.info('Explicit label-11 pretrain SHA-256: %s',
                 parent.locked._sha256(pretrained_checkpoint))
    logging.info(
        'Student after warm-up: 6 native-L + 6 paired-L + 12 paired-U; '
        'OAAC-Strong, EMA, LR, ramp and validation selector unchanged')

    print(parent.self_train(args, pretrained_checkpoint, snapshot_path))
