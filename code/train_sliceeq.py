"""SliceEq self-training on the locked PROMISE12 baseline.

SliceEq changes only the post-warmup unlabeled training pair. Real neighboring
slices and their detached EMA/LCC hard masks pass through one sampled
slice-profile operator. The network, optimizer, losses, EMA, ramp, sampler,
validation, and inference graph remain those of the current baseline.
"""

import argparse
from functools import partial
import hashlib
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
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


DEFAULT_PRETRAINED_CHECKPOINT = (
    '/home/aiteam/zhengtaoma/'
    'UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/'
    'pre_train/unet/unet_best_model.pth')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_path', type=str,
    default='/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SliceEq_PROMISE12',
                    help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum self-training iterations')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
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
    help='exact fixed baseline Pre10000 checkpoint containing net and opt; '
         'the default matches the existing Baseline experiment path')
parser.add_argument('--sliceeq_radius', type=int, default=1, choices=[1],
                    help='fixed neighboring radius; final version uses 3 slices')
parser.add_argument('--sliceeq_sigma_min', type=float, default=0.45,
                    help='minimum Gaussian profile sigma in slice units')
parser.add_argument('--sliceeq_sigma_max', type=float, default=0.85,
                    help='maximum Gaussian profile sigma in slice units')
parser.add_argument('--sliceeq_phase_min', type=float, default=-0.25,
                    help='minimum virtual center shift in slice units')
parser.add_argument('--sliceeq_phase_max', type=float, default=0.25,
                    help='maximum virtual center shift in slice units')
args = parser.parse_args()


def _import_locked_training_base():
    """Import train_coda without letting its parser consume SliceEq args."""
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        import train_coda as training_base
    finally:
        sys.argv = original_argv
    training_base.args = args
    return training_base


base = _import_locked_training_base()


def seed_data_worker(worker_id, base_seed):
    """Windows-spawn-safe callback matching the current training entry."""
    random.seed(base_seed + worker_id)


def _logits(network_output):
    return network_output[0] if isinstance(network_output, tuple) \
        else network_output


def _segmentation_losses(logits, labels, ce_loss, dice_loss):
    probabilities = torch.softmax(logits, dim=1)
    loss_ce = ce_loss(logits, labels.long())
    loss_dice = dice_loss(probabilities, labels.unsqueeze(1))
    return 0.5 * (loss_dice + loss_ce), loss_ce, loss_dice


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_pretrained_checkpoint(path):
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            'Configured --pretrained_checkpoint does not exist: {}'.format(
                resolved))
    return resolved


def _load_pretrained_checkpoint(model, ema_model, optimizer, path):
    """Strictly restore the one shared baseline `net+opt` checkpoint."""
    checkpoint = torch.load(path, map_location='cuda')
    if not isinstance(checkpoint, dict) or 'net' not in checkpoint:
        raise RuntimeError(
            'SliceEq requires a baseline checkpoint dictionary with key `net`: '
            '{}'.format(path))
    if 'opt' not in checkpoint:
        raise RuntimeError(
            'SliceEq requires optimizer state `opt` for a fair shared-pretrain '
            'start: {}'.format(path))
    model.load_state_dict(checkpoint['net'], strict=True)
    ema_model.load_state_dict(checkpoint['net'], strict=True)
    optimizer.load_state_dict(checkpoint['opt'])


def _reset_stage_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_args(flags):
    if flags.num_classes != 2:
        raise ValueError('SliceEq final version expects binary segmentation')
    if flags.batch_size <= flags.labeled_bs or flags.labeled_bs < 1:
        raise ValueError('batch_size must exceed a positive labeled_bs')
    if flags.sliceeq_sigma_min <= 0.0 or \
            flags.sliceeq_sigma_max < flags.sliceeq_sigma_min:
        raise ValueError('invalid SliceEq sigma range')
    if flags.sliceeq_phase_max < flags.sliceeq_phase_min:
        raise ValueError('invalid SliceEq phase range')
    if flags.sliceeq_phase_min < -0.5 or flags.sliceeq_phase_max > 0.5:
        raise ValueError('SliceEq phase range must stay within half a slice')
    if flags.max_iterations < 1:
        raise ValueError('max_iterations must be positive')


def self_train(flags, pretrained_checkpoint, snapshot_path):
    base_lr = flags.base_lr
    num_classes = flags.num_classes
    batch_size = flags.batch_size
    max_iterations = flags.max_iterations

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
    _load_pretrained_checkpoint(
        model, ema_model, optimizer, pretrained_checkpoint)
    checkpoint_hash = _sha256(pretrained_checkpoint)
    logging.info('Loaded shared pretrain checkpoint: %s',
                 pretrained_checkpoint)
    logging.info('Shared pretrain SHA-256: %s', checkpoint_hash)

    # Make every future method start self-training from the same random state,
    # independent of how or when the shared checkpoint was produced.
    _reset_stage_rng(flags.seed)

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
        'SliceEq profile: offsets=(-1,0,1), sigma=[%.3f, %.3f], '
        'phase=[%.3f, %.3f]',
        flags.sliceeq_sigma_min, flags.sliceeq_sigma_max,
        flags.sliceeq_phase_min, flags.sliceeq_phase_max)

    profile_generator = torch.Generator(device='cuda')
    profile_generator.manual_seed(flags.seed)
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
            label_batch = sampled_batch['label'].cuda()
            neighbor_clamped = sampled_batch['neighbor_clamped'].cuda()
            center = image_stack.shape[1] // 2
            volume_batch = image_stack[:, center]
            labeled_images = volume_batch[:flags.labeled_bs]
            labeled_labels = label_batch[:flags.labeled_bs]
            unlabeled_stack = image_stack[flags.labeled_bs:]
            unlabeled_images = volume_batch[flags.labeled_bs:]
            unlabeled_clamped = neighbor_clamped[flags.labeled_bs:]
            consistency_weight = base.get_current_consistency_weight(
                iter_num // 150)

            optimizer.zero_grad()
            if iter_num < 1000:
                # Exact baseline identity path while consistency is disabled.
                student_batch = volume_batch
                outputs = _logits(model(student_batch))
                with torch.no_grad():
                    ema_output = _logits(ema_model(unlabeled_images))
                    center_pseudo = base.get_masks(ema_output, nms=1)
                    pseudo_labels = center_pseudo.long()

                supervised_loss, loss_ce, loss_dice = \
                    _segmentation_losses(
                        outputs[:flags.labeled_bs], labeled_labels,
                        ce_loss, dice_loss)
                consistency_loss = volume_batch.new_tensor(0.0)
                loss = supervised_loss
                reacquired_images = unlabeled_images
                pseudo_stack = center_pseudo.unsqueeze(1).repeat(
                    1, len(offsets), 1, 1)
                diagnostics = {
                    'sigma_mean': volume_batch.new_tensor(0.0),
                    'absolute_phase_mean': volume_batch.new_tensor(0.0),
                    'center_weight_mean': volume_batch.new_tensor(1.0),
                    'image_absolute_change': volume_batch.new_tensor(0.0),
                    'target_changed_fraction': volume_batch.new_tensor(0.0),
                    'center_foreground_fraction': (
                        center_pseudo > 0).float().mean(),
                    'reacquired_foreground_fraction': (
                        pseudo_labels > 0).float().mean(),
                }
            else:
                with torch.no_grad():
                    unlabeled_count, stack_size, channels, height, width = \
                        unlabeled_stack.shape
                    teacher_inputs = unlabeled_stack.reshape(
                        unlabeled_count * stack_size, channels,
                        height, width)
                    ema_output = _logits(ema_model(teacher_inputs))
                    pseudo_stack = base.get_masks(
                        ema_output, nms=1).reshape(
                            unlabeled_count, stack_size, height, width)
                    weights, sigma, phase = sample_slice_profiles(
                        unlabeled_count, offsets,
                        (flags.sliceeq_sigma_min,
                         flags.sliceeq_sigma_max),
                        (flags.sliceeq_phase_min,
                         flags.sliceeq_phase_max),
                        device=unlabeled_stack.device,
                        generator=profile_generator)
                    reacquired_images, pseudo_labels, _ = \
                        paired_slice_reacquisition(
                            unlabeled_stack, pseudo_stack, weights,
                            num_classes)
                    diagnostics = reacquisition_diagnostics(
                        unlabeled_stack, pseudo_stack,
                        reacquired_images, pseudo_labels,
                        weights, sigma, phase)

                student_batch = torch.cat(
                    (labeled_images, reacquired_images), dim=0)
                outputs = _logits(model(student_batch))
                supervised_loss, loss_ce, loss_dice = \
                    _segmentation_losses(
                        outputs[:flags.labeled_bs], labeled_labels,
                        ce_loss, dice_loss)
                unlabeled_outputs = outputs[flags.labeled_bs:]
                consistency_loss, _, _ = _segmentation_losses(
                    unlabeled_outputs, pseudo_labels.long(),
                    ce_loss, dice_loss)
                loss = supervised_loss + \
                    consistency_weight * consistency_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    'Non-finite SliceEq loss at iteration {}'.format(iter_num))
            loss.backward()
            optimizer.step()
            base.update_model_ema(model, ema_model, flags.ema_decay)

            iter_num += 1
            writer.add_scalar('info/lr', base_lr, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/supervised_loss',
                              supervised_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            for name, value in diagnostics.items():
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        'Non-finite SliceEq diagnostic {}'.format(name))
                writer.add_scalar('sliceeq/' + name, value, iter_num)
            writer.add_scalar(
                'sliceeq/neighbor_clamped_sample_fraction',
                unlabeled_clamped.mean(), iter_num)

            if iter_num % 20 == 0:
                writer.add_image(
                    'train/Image', volume_batch[1, 0:1], iter_num)
                outputs_img = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image(
                    'train/Prediction', outputs_img[1] * 50, iter_num)
                writer.add_image(
                    'train/GroundTruth',
                    label_batch[1].unsqueeze(0) * 50, iter_num)
                writer.add_image(
                    'sliceeq/OriginalUnlabeled',
                    unlabeled_images[0], iter_num)
                writer.add_image(
                    'sliceeq/ReacquiredUnlabeled',
                    reacquired_images[0], iter_num)
                writer.add_image(
                    'sliceeq/AbsoluteImageChange',
                    (reacquired_images[0] -
                     unlabeled_images[0]).abs(), iter_num)
                writer.add_image(
                    'sliceeq/CenterTeacherMask',
                    pseudo_stack[0, center].unsqueeze(0) * 50, iter_num)
                writer.add_image(
                    'sliceeq/ReacquiredTarget',
                    pseudo_labels[0].unsqueeze(0) * 50, iter_num)
                writer.add_image(
                    'sliceeq/TargetChange',
                    (pseudo_labels[0] !=
                     pseudo_stack[0, center]).float().unsqueeze(0) * 50,
                    iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
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
    pretrained_checkpoint = _resolve_pretrained_checkpoint(
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

    _reset_stage_rng(args.seed)
    self_snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    if not os.path.exists(self_snapshot_path):
        os.makedirs(self_snapshot_path)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(self_snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info('================ START SliceEq SELF-TRAINING ================')
    logging.info(str(args))
    self_train(args, pretrained_checkpoint, self_snapshot_path)
