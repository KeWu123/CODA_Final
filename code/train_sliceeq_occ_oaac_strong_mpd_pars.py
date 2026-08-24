"""Direct H7.20 PARS successor of frozen SliceEqOcc-OAAC-Strong-MPD.

The only runtime intervention is the two-stream slice-index distribution.
PARS chooses patients uniformly and index thirds from one exact-L-designed,
globally frozen acquisition-opportunity law.  It never uses model difficulty,
uncertainty, U labels, validation, test results, or online feedback.
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils.sliceeq_mpd import FrozenProfileSampler, build_direct_design_artifact
from utils.sliceeq_pars import (
    FrozenPARSBatchSamplerFactory, PatientAxialAcquisitionBatchSampler,
    SAMPLER_SEED, build_direct_pars_artifact)


EXPECTED_MPD_TRAIN_SHA256 = (
    '9255640e81259309350f10f80e8a1319ae2a3deaf5a1164292c5baa9292c8f5f')
EXPECTED_MPD_UTILITY_SHA256 = (
    '9b215cefe8f22172c2811af63365c2cb21e40daba4f01f4ceae0d5cabf44c817')
MPD_ARTIFACT_FILENAME = 'mpd_profile_design.json'
PARS_ARTIFACT_FILENAME = 'pars_sampling_design.json'


def _inject_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


_inject_default('--exp', 'SliceEqOccOAACStrongMPDPARS_PROMISE12')

import train_sliceeq_occ_oaac_strong_mpd as mpd  # noqa: E402


args = mpd.args


def _validate_args(flags):
    mpd.strong.parent._validate_args(flags)
    locked_recipe = {
        'root_path': (
            '/home/aiteam/zhengtaoma/Baseline/data/'
            'PROMISE12_h5_training_source'),
        'exp': 'SliceEqOccOAACStrongMPDPARS_PROMISE12',
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
                'H7.20 PARS locks --{}={!r}; received {!r}'.format(
                    name, expected, actual))


def _numpy_state_equal(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and \
        left[2:] == right[2:]


def _run_sampler_smoke(manifest, probabilities):
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(torch.device('cuda')).clone()
    sampler = PatientAxialAcquisitionBatchSampler(
        manifest['labeled_indices'], manifest['unlabeled_indices'],
        batch_size=24, secondary_batch_size=12,
        manifest=manifest, axial_probabilities=probabilities,
        seed=SAMPLER_SEED)
    first_epoch = list(iter(sampler))
    repeated = PatientAxialAcquisitionBatchSampler(
        manifest['labeled_indices'], manifest['unlabeled_indices'],
        batch_size=24, secondary_batch_size=12,
        manifest=manifest, axial_probabilities=probabilities,
        seed=SAMPLER_SEED)
    repeated_epoch = list(iter(repeated))
    different = PatientAxialAcquisitionBatchSampler(
        manifest['labeled_indices'], manifest['unlabeled_indices'],
        batch_size=24, secondary_batch_size=12,
        manifest=manifest, axial_probabilities=probabilities,
        seed=SAMPLER_SEED + 1)
    different_epoch = list(iter(different))
    if first_epoch != repeated_epoch or first_epoch == different_epoch:
        raise RuntimeError('PARS sampler private-RNG reproducibility failed')
    if len(first_epoch) != len(manifest['labeled_indices']) // 12:
        raise RuntimeError('PARS changed the parent iterations per epoch')
    labeled_set = set(manifest['labeled_indices'])
    unlabeled_set = set(manifest['unlabeled_indices'])
    for batch in first_epoch:
        if len(batch) != 24 or not set(batch[:12]).issubset(labeled_set) or \
                not set(batch[12:]).issubset(unlabeled_set):
            raise RuntimeError('PARS violated the 12L+12U batch contract')
    if not _numpy_state_equal(numpy_before, np.random.get_state()) or \
            not torch.equal(torch_before, torch.random.get_rng_state()) or \
            not torch.equal(cuda_before, torch.cuda.get_rng_state('cuda')):
        raise RuntimeError('PARS consumed a parent-visible RNG stream')
    logging.info(
        'PARS smoke passed: private RNG reproducible, epoch length and '
        '12L+12U order unchanged, parent-visible RNG unchanged')


def _log_epoch_sampling(summary):
    logging.info(
        'PARS sampling epoch %d: L thirds=%s; U thirds=%s; '
        'L patient count range=%d--%d; U patient count range=%d--%d',
        summary['epoch'], summary['primary_third_counts'],
        summary['secondary_third_counts'],
        min(summary['primary_patient_counts'].values()),
        max(summary['primary_patient_counts'].values()),
        min(summary['secondary_patient_counts'].values()),
        max(summary['secondary_patient_counts'].values()))


if __name__ == '__main__':
    _validate_args(args)
    mpd_hash = mpd.strong.parent.locked._sha256(mpd.__file__)
    if mpd_hash != EXPECTED_MPD_TRAIN_SHA256:
        raise RuntimeError(
            'PARS MPD parent changed: expected {}; got {}'.format(
                EXPECTED_MPD_TRAIN_SHA256, mpd_hash))
    mpd_utility_path = os.path.join(
        os.path.dirname(__file__), 'utils', 'sliceeq_mpd.py')
    mpd_utility_hash = mpd.strong.parent.locked._sha256(mpd_utility_path)
    if mpd_utility_hash != EXPECTED_MPD_UTILITY_SHA256:
        raise RuntimeError(
            'PARS MPD utility changed: expected {}; got {}'.format(
                EXPECTED_MPD_UTILITY_SHA256, mpd_utility_hash))
    strong_hash = mpd.strong.parent.locked._sha256(mpd.strong.__file__)
    if strong_hash != mpd.EXPECTED_STRONG_TRAIN_SHA256:
        raise RuntimeError('PARS frozen OAAC-Strong source changed')
    base_hash = mpd.strong.parent.locked._sha256(
        mpd.strong.parent.__file__)
    if base_hash != mpd.strong.EXPECTED_PARENT_TRAIN_SHA256:
        raise RuntimeError('PARS frozen H7.15 base source changed')
    pretrained_checkpoint = \
        mpd.strong.parent.locked._resolve_pretrained_checkpoint(
            args.pretrained_checkpoint)
    checkpoint_hash = mpd.strong.parent.locked._sha256(pretrained_checkpoint)
    if checkpoint_hash != mpd.strong.EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(
            'PARS requires shared Pre10000 SHA-256 {}; got {}'.format(
                mpd.strong.EXPECTED_PRETRAINED_SHA256, checkpoint_hash))
    dataset_report = mpd.strong.validate_promise12_root(
        args.root_path, strict_split=True, check_hdf5=True)
    print('PROMISE12 preflight: {}'.format(dataset_report))

    cudnn.benchmark = False
    cudnn.deterministic = True
    mpd.strong.parent.locked._reset_stage_rng(args.seed)

    snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info(
        '====== START SliceEqOccOAACStrong-MPD-PARS DIRECT SELF-TRAINING ======')
    logging.info(str(args))
    logging.info(
        'H7.20 sole intervention: frozen patient-balanced axial '
        'acquisition-opportunity slice-index distribution; no hard mining, '
        'prediction, loss, uncertainty, U label, validation or test feedback')

    experiment_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'research', 'experiments',
        'h7_slice_profile_reacquisition'))
    mpd_protocol_path = os.path.join(
        experiment_root, 'h7_19_robust_moment_profile_gate_protocol.md')
    pars_protocol_path = os.path.join(
        experiment_root, 'h7_20_patient_axial_acquisition_risk_protocol.md')
    mpd_artifact_path = os.path.join(snapshot_path, MPD_ARTIFACT_FILENAME)
    pars_artifact_path = os.path.join(snapshot_path, PARS_ARTIFACT_FILENAME)

    mpd_design = build_direct_design_artifact(
        args.root_path, mpd_artifact_path,
        protocol_path=mpd_protocol_path)
    pars_design = build_direct_pars_artifact(
        args.root_path, mpd_design, pars_artifact_path,
        protocol_path=pars_protocol_path)
    logging.info(
        'PARS frozen axial law: parent=%s; designed=%s; distribution_sha=%s; '
        'artifact_sha=%s',
        pars_design['parent_probabilities'].tolist(),
        pars_design['probabilities'].tolist(),
        pars_design['sampling_distribution_sha256'],
        pars_design['artifact_sha256'])
    diagnostics = pars_design['report']['sampling_design']
    logging.info(
        'PARS design diagnostics: worst exposure %.8f -> %.8f; '
        'entropy %.8f/%.8f; max density ratio %.6f; checks=%s',
        diagnostics['worst_parent_exposure'],
        diagnostics['worst_designed_exposure'],
        diagnostics['entropy'], diagnostics['parent_entropy'],
        diagnostics['max_density_ratio'], diagnostics['checks'])

    profile_sampler = FrozenProfileSampler(mpd_design)
    mpd.strong._run_cuda_smoke(torch.device('cuda'))
    mpd._run_sampler_smoke(torch.device('cuda'), profile_sampler)
    _run_sampler_smoke(
        pars_design['manifest'], pars_design['probabilities'])

    appearance_generator = torch.Generator(device='cuda')
    appearance_generator.manual_seed(mpd.strong.APPEARANCE_SEED)
    mpd.strong._context.reset(appearance_generator)
    mpd.strong._configure_parent(args)
    mpd.strong.parent.sample_slice_profiles = profile_sampler
    sampler_factory = FrozenPARSBatchSamplerFactory(
        pars_design['manifest'], pars_design['probabilities'],
        seed=SAMPLER_SEED, log_callback=_log_epoch_sampling,
        parent_sampler_class=mpd.strong.parent.TwoStreamBatchSampler,
        warmup_batches=1000)
    mpd.strong.parent.TwoStreamBatchSampler = sampler_factory
    logging.info(
        'PARS preserves the exact parent sampler for iterations0--999 and '
        'replaces only post-warmup TwoStreamBatchSampler draws; MPD q, '
        'OAAC Strong1.25, '
        'network, pretrain, batch24/student36, EMA train-mode, LR, '
        'loss/ramp, validation, 1000-iteration archives and inference remain '
        'unchanged')
    result = mpd.strong.parent.self_train(
        args, pretrained_checkpoint, snapshot_path)
    expected = 2 * (args.max_iterations - 1000)
    mpd.strong._context.assert_complete(expected_paired_calls=expected)
    if sampler_factory.calls != 1 or sampler_factory.instance is None:
        raise RuntimeError('PARS parent did not create exactly one sampler')
    logging.info(
        'PARS/MPD/OAAC parent contract complete: sampler_factory_calls=%d; '
        'paired=%d; diagnostics=%d',
        sampler_factory.calls, mpd.strong._context.paired_calls,
        mpd.strong._context.diagnostic_calls)
    print(result)
