"""Incremental ablations for fractional-occupancy SliceEq.

After the inherited 1k identity warmup, the main chain adds one component at a
time: image-only re-acquisition, aligned fractional occupancy, then an exact-GT
re-acquired labeled anchor. ``paired_lu_24`` estimates the full labeled risk
with six native and six paired labeled views while retaining 12 paired
unlabeled views. ``hard_targets`` remains a mechanism control.
"""

import argparse
from functools import partial
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, TwoStreamBatchSampler
from dataloaders.sliceeq_dataset import (
    SliceStackDataSets, StackRandomGenerator)
from utils import losses, val_2d
from utils.promise12_preflight import validate_promise12_root
from utils.sliceeq import (
    paired_slice_reacquisition, reacquisition_diagnostics,
    sample_slice_profiles)
from utils.sliceeq_occ import (
    occupancy_diagnostics, soft_segmentation_loss)
from utils.sliceeq_oaac_strong import ordered_appearance_transform


DEFAULT_PRETRAINED_CHECKPOINT = (
    '/home/aiteam/zhengtaoma/'
    'UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/'
    'pre_train/unet/unet_best_model.pth')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_path', type=str,
    default='/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SliceEqOccAblation_PROMISE12',
                    help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum self-training iterations')
parser.add_argument('--batch_size', type=int, default=24,
                    help='loader batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7,
                    help='labeled data')
parser.add_argument('--ema_decay', type=float, default=0.99,
                    help='ema_decay')
parser.add_argument('--consistency_type', type=str, default='mse',
                    help='preserved baseline argument')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0,
                    help='consistency_rampup')
parser.add_argument(
    '--pretrained_checkpoint', type=str,
    default=DEFAULT_PRETRAINED_CHECKPOINT,
    help='exact fixed baseline Pre10000 checkpoint containing net and opt')
parser.add_argument('--sliceeq_radius', type=int, default=1, choices=[1],
                    help='fixed neighboring radius; uses three real slices')
parser.add_argument('--sliceeq_sigma_min', type=float, default=0.45,
                    help='minimum Gaussian profile sigma in slice units')
parser.add_argument('--sliceeq_sigma_max', type=float, default=0.85,
                    help='maximum Gaussian profile sigma in slice units')
parser.add_argument('--sliceeq_phase_min', type=float, default=-0.25,
                    help='minimum virtual center shift in slice units')
parser.add_argument('--sliceeq_phase_max', type=float, default=0.25,
                    help='maximum virtual center shift in slice units')
parser.add_argument(
    '--occ_ablation', type=str, default='full',
    choices=['baseline', 'baseline_36', 'image_only', 'image_only_36',
             'aligned_occ', 'paired_lu_24', 'hard_targets', 'occ_l_only',
             'occ_u_only', 'full', 'no_labeled_reacq'],
    help='controlled SliceEqOcc ablation branch')
parser.add_argument(
    '--appearance_mode', type=str, default='none',
    choices=['none', 'oaac_strong'],
    help='fixed post-acquisition appearance policy for matched ablations')
args = parser.parse_args()


APPEARANCE_SEED = 1339
LABELED_ASSIGNMENT_SEED_OFFSET = 3


ABLATION_COMPONENTS = {
    # M0: inherited EMA/LCC hard-pseudo-label self-training.
    'baseline': {
        'labeled_target_mode': 'center',
        'unlabeled_target_mode': 'center',
        'reacquire_unlabeled_image': False,
        'teacher_uses_neighbors': False,
        'use_labeled_reacq': False,
        'reacquire_labeled_image': False,
    },
    # Exact compute/view-matched control with a duplicated labeled view.
    'baseline_36': {
        'labeled_target_mode': 'center',
        'unlabeled_target_mode': 'center',
        'reacquire_unlabeled_image': False,
        'teacher_uses_neighbors': False,
        'use_labeled_reacq': True,
        'reacquire_labeled_image': False,
    },
    # M1: image slab augmentation with the unchanged center-slice target.
    'image_only': {
        'labeled_target_mode': 'center',
        'unlabeled_target_mode': 'center',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': False,
        'use_labeled_reacq': False,
        'reacquire_labeled_image': False,
    },
    # Full-layout image-only SRA control.
    'image_only_36': {
        'labeled_target_mode': 'center',
        'unlabeled_target_mode': 'center',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': False,
        'use_labeled_reacq': True,
        'reacquire_labeled_image': True,
    },
    # M2: acquisition-aligned fractional supervision on unlabeled data.
    'aligned_occ': {
        'labeled_target_mode': 'center',
        'unlabeled_target_mode': 'fractional',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': True,
        'use_labeled_reacq': False,
        'reacquire_labeled_image': False,
    },
    # Backward-compatible name for checkpoints/scripts created before M2 was
    # given an explicit incremental-stage name.
    'no_labeled_reacq': {
        'labeled_target_mode': 'center',
        'unlabeled_target_mode': 'fractional',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': True,
        'use_labeled_reacq': False,
        'reacquire_labeled_image': False,
    },
    # View-budget control: all 12 labeled centers appear exactly once, with a
    # random six receiving native hard supervision and the complementary six
    # receiving paired fractional supervision. The 12 unlabeled centers retain
    # paired fractional supervision, yielding 6 + 6 + 12 = 24 student views.
    'paired_lu_24': {
        'labeled_target_mode': 'fractional',
        'unlabeled_target_mode': 'fractional',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': True,
        'use_labeled_reacq': True,
        'reacquire_labeled_image': True,
        'replace_labeled_half': True,
    },
    # Additional mechanism control, not a stage in the incremental chain.
    'hard_targets': {
        'labeled_target_mode': 'hard',
        'unlabeled_target_mode': 'hard',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': True,
        'use_labeled_reacq': True,
        'reacquire_labeled_image': True,
    },
    # Factorial controls under the full 36-view paired acquisition layout.
    'occ_l_only': {
        'labeled_target_mode': 'fractional',
        'unlabeled_target_mode': 'hard',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': True,
        'use_labeled_reacq': True,
        'reacquire_labeled_image': True,
    },
    'occ_u_only': {
        'labeled_target_mode': 'hard',
        'unlabeled_target_mode': 'fractional',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': True,
        'use_labeled_reacq': True,
        'reacquire_labeled_image': True,
    },
    # M3: M2 plus the exact-GT re-acquired labeled anchor.
    'full': {
        'labeled_target_mode': 'fractional',
        'unlabeled_target_mode': 'fractional',
        'reacquire_unlabeled_image': True,
        'teacher_uses_neighbors': True,
        'use_labeled_reacq': True,
        'reacquire_labeled_image': True,
    },
}

ABLATION_DISPLAY_NAMES = {
    'baseline': 'B0 / MT-24',
    'baseline_36': 'C0 / ViewMatch-36',
    'image_only': 'Legacy / SRA-U-24',
    'image_only_36': 'C1 / SRA-Image-36',
    'aligned_occ': 'Legacy / AFO-U-24',
    'no_labeled_reacq': 'Legacy / AFO-U-24',
    'paired_lu_24': 'A3 / Paired-LU-24',
    'hard_targets': 'C2 / SRA-Hard-36',
    'occ_l_only': 'F10 / AFO-L-only',
    'occ_u_only': 'F01 / AFO-U-only',
    'full': 'C3 / SliceEqOcc',
}


def _ablation_components(name):
    """Return a copy so the locked experiment definition cannot be mutated."""
    components = dict(ABLATION_COMPONENTS[name])
    components.setdefault('replace_labeled_half', False)
    return components


def _import_locked_training_base():
    """Reuse the current network/EMA helpers without consuming v2 CLI args."""
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        import train_sliceeq as training_base
    finally:
        sys.argv = original_argv
    training_base.args = args
    training_base.base.args = args
    return training_base


locked = _import_locked_training_base()
base = locked.base


def seed_data_worker(worker_id, base_seed):
    """Windows-spawn-safe callback matching the current training entries."""
    random.seed(base_seed + worker_id)


def _logits(network_output):
    return network_output[0] if isinstance(network_output, tuple) \
        else network_output


def _hard_segmentation_losses(logits, labels, ce_loss, dice_loss):
    probabilities = torch.softmax(logits, dim=1)
    loss_ce = ce_loss(logits, labels.long())
    loss_dice = dice_loss(probabilities, labels.unsqueeze(1))
    return 0.5 * (loss_dice + loss_ce), loss_ce, loss_dice


def _validate_args(flags):
    locked._validate_args(flags)
    if flags.batch_size != 24 or flags.labeled_bs != 12:
        raise ValueError(
            'SliceEqOcc ablations require the locked loader batch_size=24 '
            'and labeled_bs=12')
    if flags.occ_ablation == 'paired_lu_24' and flags.labeled_bs % 2:
        raise ValueError('paired_lu_24 requires an even labeled_bs')


def _one_hot(labels, num_classes, dtype):
    encoded = F.one_hot(labels.long(), num_classes=num_classes)
    return encoded.permute(0, 3, 1, 2).to(dtype=dtype)


def _prefixed(prefix, values):
    return {prefix + name: value for name, value in values.items()}


def _identity_profile(batch_size, stack_size, center, reference):
    weights = reference.new_zeros((batch_size, stack_size))
    weights[:, center] = 1.0
    zeros = reference.new_zeros(batch_size)
    return weights, zeros, zeros


def self_train(flags, pretrained_checkpoint, snapshot_path):
    base_lr = flags.base_lr
    num_classes = flags.num_classes
    batch_size = flags.batch_size
    max_iterations = flags.max_iterations
    components = _ablation_components(flags.occ_ablation)

    def create_model(ema=False):
        network = base.UNet(in_chns=1, class_num=num_classes).cuda()
        if ema:
            for parameter in network.parameters():
                parameter.detach_()
        return network

    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = base.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    locked._load_pretrained_checkpoint(
        model, ema_model, optimizer, pretrained_checkpoint)
    checkpoint_hash = locked._sha256(pretrained_checkpoint)
    logging.info('Loaded shared pretrain checkpoint: %s',
                 pretrained_checkpoint)
    logging.info('Shared pretrain SHA-256: %s', checkpoint_hash)
    locked._reset_stage_rng(flags.seed)

    train_transform = StackRandomGenerator(flags.patch_size)
    db_train = SliceStackDataSets(
        base_dir=flags.root_path, transform=train_transform,
        radius=flags.sliceeq_radius)
    db_val = BaseDataSets(base_dir=flags.root_path, split='val')

    total_slices = len(db_train)
    labeled_slice = base.patients_to_slices(
        flags.root_path, flags.labelnum)
    print('Total silices is: {}, labeled slices is: {}'.format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size,
        batch_size - flags.labeled_bs)
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4,
        pin_memory=True,
        worker_init_fn=partial(seed_data_worker, base_seed=flags.seed))
    valloader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=1)

    ce_loss = base.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    writer = base.SummaryWriter(snapshot_path + '/log')
    logging.info('%s iterations per epoch', len(trainloader))
    logging.info(
        'SliceEqOcc profile: offsets=(-1,0,1), sigma=[%.3f, %.3f], '
        'phase=[%.3f, %.3f]',
        flags.sliceeq_sigma_min, flags.sliceeq_sigma_max,
        flags.sliceeq_phase_min, flags.sliceeq_phase_max)
    logging.info(
        'SliceEqOcc ablation: %s (%s)', flags.occ_ablation,
        ABLATION_DISPLAY_NAMES[flags.occ_ablation])
    logging.info(
        'SliceEqOcc appearance mode: %s%s', flags.appearance_mode,
        ' (unlabeled post-acquisition only, seed={})'.format(
            APPEARANCE_SEED)
        if flags.appearance_mode == 'oaac_strong' else '')
    if flags.occ_ablation == 'baseline':
        logging.info(
            'SliceEqOcc effective student batch after warmup: 24 '
            '(12 original-L + 12 original-U)')
    elif components['replace_labeled_half']:
        half_labeled = flags.labeled_bs // 2
        logging.info(
            'SliceEqOcc effective student batch after warmup: %d '
            '(%d original-L + %d reacquired-L + %d reacquired-U)',
            batch_size, half_labeled, half_labeled,
            batch_size - flags.labeled_bs)
    elif not components['use_labeled_reacq']:
        logging.info(
            'SliceEqOcc effective student batch after warmup: 24 '
            '(12 original-L + 12 reacquired-U)')
    else:
        logging.info(
            'SliceEqOcc effective student batch after warmup: 36 '
            '(12 original-L + 12 reacquired-L + 12 reacquired-U)')
    logging.info(
        'SliceEqOcc components: target_mode(L/U)=%s/%s, '
        'reacquire_image(L/U)=%s/%s, teacher_uses_neighbors=%s, '
        'use_labeled_reacq=%s, replace_labeled_half=%s',
        components['labeled_target_mode'],
        components['unlabeled_target_mode'],
        components['reacquire_labeled_image'],
        components['reacquire_unlabeled_image'],
        components['teacher_uses_neighbors'],
        components['use_labeled_reacq'],
        components['replace_labeled_half'])
    if components['use_labeled_reacq']:
        logging.info(
            'SliceEqOcc objective: Lsup=0.5*(L_original_hard + '
            'L_reacquired_labeled); total=Lsup+lambda*L_unlabeled')
    else:
        logging.info(
            'SliceEqOcc objective: Lsup=L_original_hard; '
            'total=Lsup+lambda*L_unlabeled')
    logging.info(
        'SliceEqOcc RNG seeds: unlabeled_profile=%d, labeled_profile=%d, '
        'labeled_assignment=%d',
        flags.seed, flags.seed + 1,
        flags.seed + LABELED_ASSIGNMENT_SEED_OFFSET)

    # Keep the unlabeled profile stream independent from the added labeled
    # view, so adding exact-GT supervision does not consume its random draws.
    unlabeled_profile_generator = torch.Generator(device='cuda')
    unlabeled_profile_generator.manual_seed(flags.seed)
    labeled_profile_generator = torch.Generator(device='cuda')
    labeled_profile_generator.manual_seed(flags.seed + 1)
    labeled_assignment_generator = torch.Generator()
    labeled_assignment_generator.manual_seed(
        flags.seed + LABELED_ASSIGNMENT_SEED_OFFSET)
    appearance_generator = None
    if flags.appearance_mode == 'oaac_strong':
        appearance_generator = torch.Generator(device='cuda')
        appearance_generator.manual_seed(APPEARANCE_SEED)
    offsets = tuple(range(
        -flags.sliceeq_radius, flags.sliceeq_radius + 1))
    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            image_stack = sampled_batch['image_stack'].cuda()
            label_stack = sampled_batch['label_stack'].cuda()
            label_batch = sampled_batch['label'].cuda()
            neighbor_clamped = sampled_batch['neighbor_clamped'].cuda()
            center = image_stack.shape[1] // 2
            volume_batch = image_stack[:, center]
            labeled_stack = image_stack[:flags.labeled_bs]
            labeled_target_stack = label_stack[:flags.labeled_bs]
            labeled_images = volume_batch[:flags.labeled_bs]
            labeled_labels = label_batch[:flags.labeled_bs]
            unlabeled_stack = image_stack[flags.labeled_bs:]
            unlabeled_images = volume_batch[flags.labeled_bs:]
            labeled_clamped = neighbor_clamped[:flags.labeled_bs]
            unlabeled_clamped = neighbor_clamped[flags.labeled_bs:]
            consistency_weight = base.get_current_consistency_weight(
                iter_num // 150)

            optimizer.zero_grad()
            if iter_num < 1000:
                # Exact baseline identity path while consistency is disabled.
                outputs = _logits(model(volume_batch))
                with torch.no_grad():
                    ema_output = _logits(ema_model(unlabeled_images))
                    center_pseudo = base.get_masks(ema_output, nms=1)
                    unlabeled_occupancy = _one_hot(
                        center_pseudo, num_classes, volume_batch.dtype)
                    labeled_occupancy = _one_hot(
                        labeled_labels, num_classes, volume_batch.dtype)

                original_supervised_loss, original_ce, original_dice = \
                    _hard_segmentation_losses(
                        outputs[:flags.labeled_bs], labeled_labels,
                        ce_loss, dice_loss)
                reacquired_labeled_loss = volume_batch.new_tensor(0.0)
                reacquired_labeled_ce = volume_batch.new_tensor(0.0)
                reacquired_labeled_dice = volume_batch.new_tensor(0.0)
                supervised_loss = original_supervised_loss
                consistency_loss = volume_batch.new_tensor(0.0)
                consistency_ce = volume_batch.new_tensor(0.0)
                consistency_dice = volume_batch.new_tensor(0.0)
                loss = supervised_loss
                labeled_reacquired_images = labeled_images
                unlabeled_reacquired_images = unlabeled_images
                pseudo_stack = center_pseudo.unsqueeze(1).repeat(
                    1, len(offsets), 1, 1)
                zero = volume_batch.new_tensor(0.0)
                diagnostics = {
                    'labeled_sigma_mean': zero,
                    'labeled_absolute_phase_mean': zero,
                    'labeled_center_weight_mean': volume_batch.new_tensor(1.0),
                    'labeled_image_absolute_change': zero,
                    'labeled_target_changed_fraction': zero,
                    'labeled_normalized_occupancy_entropy': zero,
                    'labeled_fractional_pixel_fraction': zero,
                    'labeled_occupancy_deviation_from_center': zero,
                    'labeled_hard_target_changed_fraction': zero,
                    'labeled_foreground_occupancy_mean': (
                        labeled_labels > 0).float().mean(),
                    'labeled_center_foreground_fraction': (
                        labeled_labels > 0).float().mean(),
                    'unlabeled_sigma_mean': zero,
                    'unlabeled_absolute_phase_mean': zero,
                    'unlabeled_center_weight_mean': volume_batch.new_tensor(1.0),
                    'unlabeled_image_absolute_change': zero,
                    'unlabeled_target_changed_fraction': zero,
                    'unlabeled_normalized_occupancy_entropy': zero,
                    'unlabeled_fractional_pixel_fraction': zero,
                    'unlabeled_occupancy_deviation_from_center': zero,
                    'unlabeled_hard_target_changed_fraction': zero,
                    'unlabeled_foreground_occupancy_mean': (
                        center_pseudo > 0).float().mean(),
                    'unlabeled_center_foreground_fraction': (
                        center_pseudo > 0).float().mean(),
                }
                preview_image = volume_batch[1, 0:1]
                preview_label = label_batch[1].unsqueeze(0)
            else:
                with torch.no_grad():
                    unlabeled_count, stack_size, channels, height, width = \
                        unlabeled_stack.shape
                    if components['teacher_uses_neighbors']:
                        teacher_inputs = unlabeled_stack.reshape(
                            unlabeled_count * stack_size, channels,
                            height, width)
                        ema_output = _logits(ema_model(teacher_inputs))
                        pseudo_stack = base.get_masks(
                            ema_output, nms=1).reshape(
                                unlabeled_count, stack_size, height, width)
                    else:
                        ema_output = _logits(ema_model(unlabeled_images))
                        center_pseudo = base.get_masks(ema_output, nms=1)
                        pseudo_stack = center_pseudo.unsqueeze(1).repeat(
                            1, stack_size, 1, 1)

                    if components['reacquire_unlabeled_image']:
                        unlabeled_weights, unlabeled_sigma, unlabeled_phase = \
                            sample_slice_profiles(
                                unlabeled_count, offsets,
                                (flags.sliceeq_sigma_min,
                                 flags.sliceeq_sigma_max),
                                (flags.sliceeq_phase_min,
                                 flags.sliceeq_phase_max),
                                device=volume_batch.device,
                                generator=unlabeled_profile_generator)
                        unlabeled_reacquired_images, \
                            unlabeled_hard_target, unlabeled_occupancy = \
                            paired_slice_reacquisition(
                                unlabeled_stack, pseudo_stack,
                                unlabeled_weights, num_classes)
                    else:
                        unlabeled_weights, unlabeled_sigma, unlabeled_phase = \
                            _identity_profile(
                                unlabeled_count, stack_size, center,
                                volume_batch)
                        unlabeled_reacquired_images = unlabeled_images
                        unlabeled_hard_target = pseudo_stack[:, center]
                        unlabeled_occupancy = _one_hot(
                            unlabeled_hard_target, num_classes,
                            volume_batch.dtype)

                    native_labeled_indices = None
                    if components['replace_labeled_half']:
                        labeled_permutation = torch.randperm(
                            flags.labeled_bs,
                            generator=labeled_assignment_generator)
                        paired_count = flags.labeled_bs // 2
                        paired_labeled_indices = labeled_permutation[
                            :paired_count].to(device=volume_batch.device)
                        native_labeled_indices = labeled_permutation[
                            paired_count:].to(device=volume_batch.device)
                        labeled_reacq_stack = labeled_stack.index_select(
                            0, paired_labeled_indices)
                        labeled_reacq_target_stack = \
                            labeled_target_stack.index_select(
                                0, paired_labeled_indices)
                    else:
                        paired_labeled_indices = None
                        paired_count = flags.labeled_bs
                        labeled_reacq_stack = labeled_stack
                        labeled_reacq_target_stack = labeled_target_stack

                    labeled_reacq_center_target = \
                        labeled_reacq_target_stack[:, center]

                    if components['use_labeled_reacq'] and \
                            components['reacquire_labeled_image']:
                        labeled_weights, labeled_sigma, labeled_phase = \
                            sample_slice_profiles(
                                paired_count, offsets,
                                (flags.sliceeq_sigma_min,
                                 flags.sliceeq_sigma_max),
                                (flags.sliceeq_phase_min,
                                 flags.sliceeq_phase_max),
                                device=volume_batch.device,
                                generator=labeled_profile_generator)
                        labeled_reacquired_images, labeled_hard_target, \
                            labeled_occupancy = paired_slice_reacquisition(
                                labeled_reacq_stack,
                                labeled_reacq_target_stack,
                                labeled_weights, num_classes)
                    else:
                        labeled_weights, labeled_sigma, labeled_phase = \
                            _identity_profile(
                                flags.labeled_bs, stack_size, center,
                                volume_batch)
                        labeled_reacquired_images = labeled_images
                        labeled_hard_target = labeled_labels
                        labeled_occupancy = _one_hot(
                            labeled_labels, num_classes, volume_batch.dtype)

                    if components['labeled_target_mode'] == 'hard':
                        labeled_occupancy = _one_hot(
                            labeled_hard_target, num_classes,
                            volume_batch.dtype)
                    elif components['labeled_target_mode'] == 'center':
                        labeled_occupancy = _one_hot(
                            labeled_reacq_center_target, num_classes,
                            volume_batch.dtype)

                    if components['unlabeled_target_mode'] == 'hard':
                        unlabeled_occupancy = _one_hot(
                            unlabeled_hard_target, num_classes,
                            volume_batch.dtype)
                    elif components['unlabeled_target_mode'] == 'center':
                        unlabeled_occupancy = _one_hot(
                            pseudo_stack[:, center], num_classes,
                            volume_batch.dtype)

                    if appearance_generator is not None:
                        unlabeled_reacquired_images, appearance_diagnostics = \
                            ordered_appearance_transform(
                                unlabeled_reacquired_images,
                                generator=appearance_generator)
                    else:
                        appearance_diagnostics = {}

                    labeled_profile_diagnostics = reacquisition_diagnostics(
                        labeled_reacq_stack, labeled_reacq_target_stack,
                        labeled_reacquired_images, labeled_hard_target,
                        labeled_weights, labeled_sigma, labeled_phase)
                    unlabeled_profile_diagnostics = reacquisition_diagnostics(
                        unlabeled_stack, pseudo_stack,
                        unlabeled_reacquired_images, unlabeled_hard_target,
                        unlabeled_weights, unlabeled_sigma, unlabeled_phase)
                    diagnostics = {}
                    diagnostics.update(_prefixed(
                        'labeled_', labeled_profile_diagnostics))
                    diagnostics.update(_prefixed(
                        'labeled_', occupancy_diagnostics(
                            labeled_occupancy,
                            labeled_reacq_center_target)))
                    diagnostics.update(_prefixed(
                        'unlabeled_', unlabeled_profile_diagnostics))
                    diagnostics.update(_prefixed(
                        'unlabeled_', occupancy_diagnostics(
                            unlabeled_occupancy,
                            pseudo_stack[:, center])))
                    diagnostics.update(appearance_diagnostics)

                if components['replace_labeled_half']:
                    native_labeled_images = labeled_images.index_select(
                        0, native_labeled_indices)
                    native_labeled_labels = labeled_labels.index_select(
                        0, native_labeled_indices)
                    student_batch = torch.cat(
                        (native_labeled_images, labeled_reacquired_images,
                         unlabeled_reacquired_images), dim=0)
                    outputs = _logits(model(student_batch))
                    native_count = native_labeled_images.shape[0]
                    paired_count = labeled_reacquired_images.shape[0]
                    original_outputs = outputs[:native_count]
                    reacquired_labeled_outputs = outputs[
                        native_count:native_count + paired_count]
                    unlabeled_outputs = outputs[
                        native_count + paired_count:]
                    original_target_labels = native_labeled_labels
                    reacquired_target_occupancy = labeled_occupancy
                    if student_batch.shape[0] != batch_size:
                        raise RuntimeError(
                            'paired_lu_24 must create exactly {} student '
                            'views, got {}'.format(
                                batch_size, student_batch.shape[0]))
                elif not components['use_labeled_reacq']:
                    student_batch = torch.cat(
                        (labeled_images, unlabeled_reacquired_images), dim=0)
                    outputs = _logits(model(student_batch))
                    original_outputs = outputs[:flags.labeled_bs]
                    reacquired_labeled_outputs = None
                    unlabeled_outputs = outputs[flags.labeled_bs:]
                    original_target_labels = labeled_labels
                    reacquired_target_occupancy = None
                else:
                    student_batch = torch.cat(
                        (labeled_images, labeled_reacquired_images,
                         unlabeled_reacquired_images), dim=0)
                    outputs = _logits(model(student_batch))
                    original_outputs = outputs[:flags.labeled_bs]
                    reacquired_labeled_outputs = outputs[
                        flags.labeled_bs:2 * flags.labeled_bs]
                    unlabeled_outputs = outputs[2 * flags.labeled_bs:]
                    original_target_labels = labeled_labels
                    reacquired_target_occupancy = labeled_occupancy

                preview_image = student_batch[1, 0:1]
                preview_label = original_target_labels[1].unsqueeze(0)

                original_supervised_loss, original_ce, original_dice = \
                    _hard_segmentation_losses(
                        original_outputs, original_target_labels,
                        ce_loss, dice_loss)
                if reacquired_labeled_outputs is None:
                    reacquired_labeled_loss = volume_batch.new_tensor(0.0)
                    reacquired_labeled_ce = volume_batch.new_tensor(0.0)
                    reacquired_labeled_dice = volume_batch.new_tensor(0.0)
                    supervised_loss = original_supervised_loss
                else:
                    reacquired_labeled_loss, reacquired_labeled_ce, \
                        reacquired_labeled_dice = soft_segmentation_loss(
                            reacquired_labeled_outputs,
                            reacquired_target_occupancy)
                    supervised_loss = 0.5 * (
                        original_supervised_loss + reacquired_labeled_loss)
                consistency_loss, consistency_ce, consistency_dice = \
                    soft_segmentation_loss(
                        unlabeled_outputs, unlabeled_occupancy)
                loss = supervised_loss + \
                    consistency_weight * consistency_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    'Non-finite SliceEqOcc ablation loss at iteration {}'.format(
                        iter_num))
            loss.backward()
            optimizer.step()
            base.update_model_ema(model, ema_model, flags.ema_decay)

            iter_num += 1
            writer.add_scalar('info/lr', base_lr, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/supervised_loss',
                              supervised_loss, iter_num)
            writer.add_scalar('info/original_supervised_loss',
                              original_supervised_loss, iter_num)
            writer.add_scalar('info/original_supervised_ce',
                              original_ce, iter_num)
            writer.add_scalar('info/original_supervised_dice',
                              original_dice, iter_num)
            writer.add_scalar('info/reacquired_labeled_loss',
                              reacquired_labeled_loss, iter_num)
            writer.add_scalar('info/reacquired_labeled_ce',
                              reacquired_labeled_ce, iter_num)
            writer.add_scalar('info/reacquired_labeled_dice',
                              reacquired_labeled_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_ce',
                              consistency_ce, iter_num)
            writer.add_scalar('info/consistency_dice',
                              consistency_dice, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            for name, value in diagnostics.items():
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        'Non-finite SliceEqOcc diagnostic {}'.format(name))
                writer.add_scalar('sliceeq_occ/' + name, value, iter_num)
            writer.add_scalar(
                'sliceeq_occ/labeled_neighbor_clamped_sample_fraction',
                labeled_clamped.mean(), iter_num)
            writer.add_scalar(
                'sliceeq_occ/unlabeled_neighbor_clamped_sample_fraction',
                unlabeled_clamped.mean(), iter_num)

            if iter_num % 20 == 0:
                writer.add_image(
                    'train/Image', preview_image, iter_num)
                outputs_img = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image(
                    'train/Prediction', outputs_img[1] * 50, iter_num)
                writer.add_image(
                    'train/GroundTruth',
                    preview_label * 50, iter_num)
                writer.add_image(
                    'sliceeq_occ/ReacquiredLabeled',
                    labeled_reacquired_images[0], iter_num)
                writer.add_image(
                    'sliceeq_occ/LabeledForegroundOccupancy',
                    labeled_occupancy[0, 1:2], iter_num)
                writer.add_image(
                    'sliceeq_occ/ReacquiredUnlabeled',
                    unlabeled_reacquired_images[0], iter_num)
                writer.add_image(
                    'sliceeq_occ/UnlabeledForegroundOccupancy',
                    unlabeled_occupancy[0, 1:2], iter_num)
                writer.add_image(
                    'sliceeq_occ/CenterTeacherMask',
                    pseudo_stack[0, center].unsqueeze(0) * 50, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                logging.info(
                    'SliceEqOcc train iteration %d: lambda=%.6f '
                    'loss(original/L-eq/U-eq)=%.6f/%.6f/%.6f; '
                    'L-profile(sigma/abs_phase/center_w)=%.4f/%.4f/%.4f; '
                    'U-profile=%.4f/%.4f/%.4f',
                    iter_num,
                    float(consistency_weight),
                    original_supervised_loss.item(),
                    reacquired_labeled_loss.item(),
                    consistency_loss.item(),
                    diagnostics['labeled_sigma_mean'].item(),
                    diagnostics['labeled_absolute_phase_mean'].item(),
                    diagnostics['labeled_center_weight_mean'].item(),
                    diagnostics['unlabeled_sigma_mean'].item(),
                    diagnostics['unlabeled_absolute_phase_mean'].item(),
                    diagnostics['unlabeled_center_weight_mean'].item())
                logging.info(
                    'SliceEqOcc occupancy iteration %d: '
                    'L(frac/entropy/dev/hard_change)=%.6f/%.6f/%.6f/%.6f; '
                    'U=%.6f/%.6f/%.6f/%.6f; clamp(L/U)=%.4f/%.4f',
                    iter_num,
                    diagnostics['labeled_fractional_pixel_fraction'].item(),
                    diagnostics[
                        'labeled_normalized_occupancy_entropy'].item(),
                    diagnostics[
                        'labeled_occupancy_deviation_from_center'].item(),
                    diagnostics[
                        'labeled_hard_target_changed_fraction'].item(),
                    diagnostics[
                        'unlabeled_fractional_pixel_fraction'].item(),
                    diagnostics[
                        'unlabeled_normalized_occupancy_entropy'].item(),
                    diagnostics[
                        'unlabeled_occupancy_deviation_from_center'].item(),
                    diagnostics[
                        'unlabeled_hard_target_changed_fraction'].item(),
                    labeled_clamped.mean().item(),
                    unlabeled_clamped.mean().item())

                model.eval()
                metric_list = 0.0
                for _, validation_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        validation_batch['image'], validation_batch['label'],
                        model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        'info/val_{}_dice'.format(class_i + 1),
                        metric_list[class_i, 0], iter_num)
                    writer.add_scalar(
                        'info/val_{}_hd95'.format(class_i + 1),
                        metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar(
                    'info/val_mean_dice', performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(
                            iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(
                        snapshot_path,
                        '{}_best_model.pth'.format(flags.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                logging.info(
                    'iteration %d : mean_dice : %f, best_dice : %f',
                    iter_num, performance, best_performance)
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info('save model to %s', save_mode_path)

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return 'Training Finished!'


if __name__ == '__main__':
    _validate_args(args)
    pretrained_checkpoint = locked._resolve_pretrained_checkpoint(
        args.pretrained_checkpoint)
    dataset_report = validate_promise12_root(
        args.root_path, strict_split=True, check_hdf5=True)
    print('PROMISE12 preflight: {}'.format(dataset_report))

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    locked._reset_stage_rng(args.seed)
    self_snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    if not os.path.exists(self_snapshot_path):
        os.makedirs(self_snapshot_path)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(self_snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info(
        '============= START SliceEqOcc ABLATION SELF-TRAINING =============')
    logging.info(str(args))
    self_train(args, pretrained_checkpoint, self_snapshot_path)
