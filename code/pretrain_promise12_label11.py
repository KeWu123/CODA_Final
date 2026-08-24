"""Isolated 11-label PROMISE12 supervised pretraining entry.

This reproduces the locked 10k baseline pretraining recipe and writes the
required ``net`` + ``opt`` bundle for the 11-label SliceEqOcc-OAAC-Strong run.
It does not start self-training and does not modify existing 7-label entries.
"""

import argparse
import hashlib
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils.promise12_preflight import validate_promise12_root


LABELNUM = 11
LABELED_SLICES = 306


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', type=str,
        default=(
            '/home/aiteam/zhengtaoma/Baseline/data/'
            'PROMISE12_h5_training_source'))
    parser.add_argument(
        '--exp', type=str, default='SharedPretrain_PROMISE12')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--pre_iterations', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--labeled_bs', type=int, default=12)
    parser.add_argument('--labelnum', type=int, default=LABELNUM)
    return parser


FLAGS = _build_parser().parse_args()


def _import_training_base():
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        import train_coda as training_base
    finally:
        sys.argv = original_argv
    return training_base


def _read_list(root_path, filename):
    path = os.path.join(root_path, filename)
    with open(path, 'r', encoding='utf-8-sig') as stream:
        return [line.strip() for line in stream if line.strip()]


def _case_name(slice_name):
    if '_slice_' not in slice_name:
        raise ValueError('Unexpected PROMISE12 slice name: {}'.format(
            slice_name))
    return slice_name.split('_slice_', 1)[0]


def validate_label11_boundary(root_path):
    """Verify that the first 306 slices are exactly the first 11 cases."""
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
            'An 11-label case crosses the locked slice boundary {}'.format(
                LABELED_SLICES))
    return {
        'labelnum': LABELNUM,
        'labeled_slices': LABELED_SLICES,
        'labeled_cases': train_cases[:LABELNUM],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(flags):
    locked = {
        'exp': 'SharedPretrain_PROMISE12',
        'model': 'unet',
        'pre_iterations': 10000,
        'batch_size': 24,
        'deterministic': 1,
        'base_lr': 0.01,
        'patch_size': [256, 256],
        'seed': 1337,
        'num_classes': 2,
        'labeled_bs': 12,
        'labelnum': LABELNUM,
    }
    for name, expected in locked.items():
        actual = getattr(flags, name)
        if name == 'patch_size':
            actual = list(actual)
        if actual != expected:
            raise ValueError(
                '11-label pretraining locks --{}={!r}; received {!r}'.format(
                    name, expected, actual))


if __name__ == '__main__':
    _validate_args(FLAGS)
    base = _import_training_base()
    dataset_report = validate_promise12_root(
        FLAGS.root_path, strict_split=True, check_hdf5=True)
    label_report = validate_label11_boundary(FLAGS.root_path)
    if base.patients_to_slices(FLAGS.root_path, LABELNUM) != LABELED_SLICES:
        raise RuntimeError('patients_to_slices no longer maps 11 cases to 306')
    print('PROMISE12 preflight: {}'.format(dataset_report))
    print('PROMISE12 label-11 boundary: {}'.format(label_report))

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)

    snapshot_path = '../model/{}_{}_labeled/pre_train/{}'.format(
        FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info('============= START PROMISE12 LABEL-11 PRETRAIN =============')
    logging.info(str(FLAGS))
    logging.info('Locked labeled boundary: first 11 cases / 306 slices')
    base.pre_train(FLAGS, snapshot_path)

    checkpoint = os.path.join(snapshot_path, 'unet_best_model.pth')
    if not os.path.isfile(checkpoint):
        raise RuntimeError('Pretraining did not produce {}'.format(checkpoint))
    logging.info('Label-11 pretrain checkpoint: %s', checkpoint)
    logging.info('Label-11 pretrain SHA-256: %s', _sha256(checkpoint))
    print(checkpoint)
