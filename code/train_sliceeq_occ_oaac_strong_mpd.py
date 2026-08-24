"""Direct full training for SliceEqOcc-OAAC-Strong with H7.19 MPD.

Before self-training, this isolated successor uses only the first 191 labeled
training slices to solve one global, phase-symmetric profile distribution.  It
then freezes that distribution and replaces only the parent profile sampler.
The user explicitly authorized skipping the previously proposed LOPO gate, so
this run is exploratory by design.  Parent training/validation code is reused.
"""

import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn

from utils.sliceeq_mpd import (
    FrozenProfileSampler, build_direct_design_artifact)


EXPECTED_STRONG_TRAIN_SHA256 = (
    'b3219557df39aee680b5bc055b9ff6dc88f5acfa03e8ed1fa5042aacade5bf60')
DESIGN_ARTIFACT_FILENAME = 'mpd_profile_design.json'


def _inject_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


_inject_default('--exp', 'SliceEqOccOAACStrongMPD_PROMISE12')

import train_sliceeq_occ_oaac_strong as strong  # noqa: E402


args = strong.args


def _validate_args(flags):
    strong.parent._validate_args(flags)
    locked_recipe = {
        'root_path': (
            '/home/aiteam/zhengtaoma/Baseline/data/'
            'PROMISE12_h5_training_source'),
        'exp': 'SliceEqOccOAACStrongMPD_PROMISE12',
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
                'H7.19 MPD locks --{}={!r}; received {!r}'.format(
                    name, expected, actual))


def _run_sampler_smoke(device, sampler):
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()
    first_generator = torch.Generator(device=device)
    repeated_generator = torch.Generator(device=device)
    different_generator = torch.Generator(device=device)
    first_generator.manual_seed(args.seed)
    repeated_generator.manual_seed(args.seed)
    different_generator.manual_seed(args.seed + 1)
    call = dict(
        batch_size=128,
        offsets=(-1.0, 0.0, 1.0),
        sigma_range=(0.45, 0.85),
        phase_range=(-0.25, 0.25),
        device=device)
    first = sampler(generator=first_generator, **call)
    repeated = sampler(generator=repeated_generator, **call)
    different = sampler(generator=different_generator, **call)
    if not all(torch.equal(a, b) for a, b in zip(first, repeated)):
        raise RuntimeError('MPD sampler is not private-RNG reproducible')
    if all(torch.equal(a, b) for a, b in zip(first, different)):
        raise RuntimeError('MPD sampler ignores its private RNG seed')
    if not torch.allclose(
            first[0].sum(dim=1), torch.ones(128, device=device),
            atol=1e-6, rtol=1e-6) or (first[0] < 0.0).any():
        raise RuntimeError('MPD sampler violated convex profile weights')
    if not torch.equal(cpu_before, torch.random.get_rng_state()) or \
            not torch.equal(cuda_before, torch.cuda.get_rng_state(device)):
        raise RuntimeError('MPD sampler consumed a parent-visible RNG stream')
    logging.info(
        'MPD CUDA smoke passed: convex profiles, private RNG reproducible, '
        'parent-visible RNG unchanged')


if __name__ == '__main__':
    _validate_args(args)
    strong_hash = strong.parent.locked._sha256(strong.__file__)
    if strong_hash != EXPECTED_STRONG_TRAIN_SHA256:
        raise RuntimeError(
            'MPD Strong source changed: expected {}; got {}'.format(
                EXPECTED_STRONG_TRAIN_SHA256, strong_hash))
    parent_hash = strong.parent.locked._sha256(strong.parent.__file__)
    if parent_hash != strong.EXPECTED_PARENT_TRAIN_SHA256:
        raise RuntimeError(
            'MPD base source changed: expected {}; got {}'.format(
                strong.EXPECTED_PARENT_TRAIN_SHA256, parent_hash))
    pretrained_checkpoint = strong.parent.locked._resolve_pretrained_checkpoint(
        args.pretrained_checkpoint)
    checkpoint_hash = strong.parent.locked._sha256(pretrained_checkpoint)
    if checkpoint_hash != strong.EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(
            'MPD requires shared Pre10000 SHA-256 {}; got {}'.format(
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
        '====== START SliceEqOccOAACStrong-MPD DIRECT SELF-TRAINING ======')
    logging.info(str(args))
    logging.info(
        'H7.19 execution scope: user-authorized exploratory direct full run; '
        'LOPO/zero-training gate skipped')

    protocol_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'research', 'experiments',
        'h7_slice_profile_reacquisition',
        'h7_19_robust_moment_profile_gate_protocol.md'))
    artifact_path = os.path.join(snapshot_path, DESIGN_ARTIFACT_FILENAME)
    design = build_direct_design_artifact(
        args.root_path, artifact_path, protocol_path=protocol_path)
    logging.info(
        'MPD frozen global profile: distribution_sha256=%s; '
        'artifact_sha256=%s; artifact=%s',
        design['distribution_sha256'], design['artifact_sha256'],
        artifact_path)
    diagnostics = design['report']['full_design']['diagnostics']
    logging.info(
        'MPD design diagnostics: worst_rfi=%.8f; entropy=%.8f/%0.8f; '
        'max_density_ratio=%.6f; mirror_error=%.3e; moments=%s',
        diagnostics['worst_utility'], diagnostics['entropy'],
        diagnostics['parent_entropy'], diagnostics['max_density_ratio'],
        diagnostics['mirror_error'], diagnostics['moments'])
    strata_report = design['report']['patient_strata']
    logging.info(
        'MPD RFI strata: active=%d, structurally_empty=%d (%s); all %d '
        'strata remain in image-residual constraints',
        len(strata_report['active_rfi_strata']),
        len(strata_report['structurally_empty_rfi_strata']),
        strata_report['structurally_empty_rfi_strata'],
        len(strata_report['names']))
    logging.info(
        'MPD design consumed only first191 labeled-training H5 image/label; '
        'no U label, validation, test, prediction or loss entered the design')

    sampler = FrozenProfileSampler(design)
    strong._run_cuda_smoke(torch.device('cuda'))
    _run_sampler_smoke(torch.device('cuda'), sampler)
    appearance_generator = torch.Generator(device='cuda')
    appearance_generator.manual_seed(strong.APPEARANCE_SEED)
    strong._context.reset(appearance_generator)
    strong._configure_parent(args)
    strong.parent.sample_slice_profiles = sampler
    logging.info(
        'MPD replaces only sample_slice_profiles; OAAC Strong scale1.25, '
        'network, batch36, EMA train-mode, LR, loss/ramp, validation and '
        '1000-iteration periodic checkpoint rule remain unchanged')
    result = strong.parent.self_train(
        args, pretrained_checkpoint, snapshot_path)
    expected = 2 * (args.max_iterations - 1000)
    strong._context.assert_complete(expected_paired_calls=expected)
    logging.info(
        'MPD/OAAC parent call contract complete: paired=%d, diagnostics=%d',
        strong._context.paired_calls, strong._context.diagnostic_calls)
    print(result)
