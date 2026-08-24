"""OBA training entry built on the locked PROMISE12 baseline recipe.

The existing CoDA entry is imported only to reuse its exact U-Net, supervised
pretraining, validation, EMA, LCC, and loss helpers. OBA changes the post-warmup
unlabeled student views: two antithetic views share the Baseline's hard pseudo-label.
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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from utils import losses, val_2d
from utils.oba import SUPPORTED_AUGMENTATIONS, orbit_balanced_augment
from utils.promise12_preflight import validate_promise12_root


parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_path', type=str,
    default='/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='OBA_PROMISE12',
                    help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000,
                    help='maximum epoch number to pre-train')
parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum epoch number to train')
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
                    help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0,
                    help='consistency_rampup')
parser.add_argument(
    '--pretrained_checkpoint', type=str,
    default='/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth',
    help='fixed baseline Pre10000 checkpoint containing net and opt; no search')

parser.add_argument(
    '--oba_augmentations', type=str,
    default='log_gamma,smooth_bias,gaussian_noise',
    help='comma-separated OBA antithetic coordinate families')
parser.add_argument('--oba_gamma_min', type=float, default=0.10,
                    help='minimum absolute log-gamma coordinate')
parser.add_argument('--oba_gamma_max', type=float, default=0.40,
                    help='maximum absolute log-gamma coordinate')
parser.add_argument('--oba_bias_min', type=float, default=0.10,
                    help='minimum smooth logit-bias coordinate')
parser.add_argument('--oba_bias_max', type=float, default=0.35,
                    help='maximum smooth logit-bias coordinate')
parser.add_argument('--oba_noise_min', type=float, default=0.05,
                    help='minimum Gaussian coordinate relative to slice std')
parser.add_argument('--oba_noise_max', type=float, default=0.15,
                    help='maximum Gaussian coordinate relative to slice std')
parser.add_argument('--oba_bias_grid_size', type=int, default=8,
                    help='low-resolution grid size for smooth bias coordinates')
args = parser.parse_args()


def _import_locked_training_base():
    """Import train_coda without letting its parser consume OBA arguments."""
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
    """Windows-spawn-safe worker callback matching the current CoDA entry."""
    random.seed(base_seed + worker_id)


def _parse_families(value):
    return tuple(item.strip() for item in value.split(',') if item.strip())


def _validate_oba_args(flags):
    if flags.num_classes != 2:
        raise ValueError('OBA final version expects binary segmentation')
    if flags.batch_size <= flags.labeled_bs or flags.labeled_bs < 1:
        raise ValueError('batch_size must exceed a positive labeled_bs')
    families = _parse_families(flags.oba_augmentations)
    if not families or len(families) != len(set(families)):
        raise ValueError('OBA augmentation list must be non-empty and unique')
    unknown = set(families) - set(SUPPORTED_AUGMENTATIONS)
    if unknown:
        raise ValueError('unsupported OBA augmentations: {}'.format(
            ', '.join(sorted(unknown))))
    for name, low, high in (
            ('gamma', flags.oba_gamma_min, flags.oba_gamma_max),
            ('bias', flags.oba_bias_min, flags.oba_bias_max),
            ('noise', flags.oba_noise_min, flags.oba_noise_max)):
        if low < 0.0 or high < low:
            raise ValueError('invalid OBA {} range'.format(name))
    if flags.oba_bias_grid_size < 2:
        raise ValueError('oba_bias_grid_size must be at least 2')


def _logits(network_output):
    return network_output[0] if isinstance(network_output, tuple) \
        else network_output


def _segmentation_losses(logits, labels, ce_loss, dice_loss):
    probabilities = torch.softmax(logits, dim=1)
    loss_ce = ce_loss(logits, labels.long())
    loss_dice = dice_loss(probabilities, labels.unsqueeze(1))
    return 0.5 * (loss_dice + loss_ce), loss_ce, loss_dice


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

    base.load_pretrained_checkpoint(
        model, ema_model, optimizer, pretrained_checkpoint)
    logging.info('Loaded shared pretrain checkpoint: %s',
                 pretrained_checkpoint)
    logging.info('Shared pretrain SHA-256: %s',
                 base.checkpoint_sha256(pretrained_checkpoint))
    base.reset_stage_rng(flags.seed)

    db_train = BaseDataSets(
        base_dir=flags.root_path, split='train', num=None,
        transform=transforms.Compose([RandomGenerator(flags.patch_size)]))
    db_val = BaseDataSets(base_dir=flags.root_path, split='val')

    total_slices = len(db_train)
    labeled_slice = base.patients_to_slices(flags.root_path, flags.labelnum)
    print('Total silices is: {}, labeled slices is: {}'.format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size,
        batch_size - flags.labeled_bs)
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
        worker_init_fn=partial(seed_data_worker, base_seed=flags.seed))
    valloader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=1)

    ce_loss = base.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    writer = base.SummaryWriter(snapshot_path + '/log')
    logging.info('%s iterations per epoch', len(trainloader))

    # Keep OBA sampling independent from the model/dropout RNG stream.
    oba_generator = torch.Generator(device='cuda')
    oba_generator.manual_seed(flags.seed)
    families = _parse_families(flags.oba_augmentations)
    effective_batch = flags.labeled_bs + 2 * (
        flags.batch_size - flags.labeled_bs)
    logging.info(
        'OBA post-warmup student batch: %d = %d labeled + 2 x %d unlabeled',
        effective_batch, flags.labeled_bs,
        flags.batch_size - flags.labeled_bs)
    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            labeled_images = volume_batch[:flags.labeled_bs]
            labeled_labels = label_batch[:flags.labeled_bs]
            unlabeled_images = volume_batch[flags.labeled_bs:]
            ema_inputs = unlabeled_images
            consistency_weight = base.get_current_consistency_weight(
                iter_num // 150)

            optimizer.zero_grad()
            if iter_num < 1000:
                # Exact baseline identity path while its consistency loss is zero.
                outputs = _logits(model(volume_batch))
                with torch.no_grad():
                    ema_output = _logits(ema_model(ema_inputs))
                    pseudo_labels = base.get_masks(ema_output, nms=1)
                    ema_output_soft = torch.zeros_like(ema_output).scatter_(
                        1, pseudo_labels.unsqueeze(1).long(), 1.0)
                    del ema_output_soft

                supervised_loss, loss_ce, loss_dice = _segmentation_losses(
                    outputs[:flags.labeled_bs], labeled_labels,
                    ce_loss, dice_loss)
                consistency_loss = volume_batch.new_tensor(0.0)
                loss = supervised_loss
                loss.backward()
                plus_images = unlabeled_images
                minus_images = unlabeled_images
                visual_outputs = outputs.detach()
                oba_metadata = {
                    'severity': volume_batch.new_zeros(unlabeled_images.shape[0]),
                    'plus_mean_absolute_change': volume_batch.new_tensor(0.0),
                    'minus_mean_absolute_change': volume_batch.new_tensor(0.0),
                    'displacement_cosine': volume_batch.new_tensor(0.0),
                    'midpoint_drift': volume_batch.new_tensor(0.0),
                    'pair_span': volume_batch.new_tensor(0.0),
                }
                for family in families:
                    oba_metadata['family_fraction_{}'.format(family)] = \
                        volume_batch.new_tensor(0.0)
                pair_loss_gap = volume_batch.new_tensor(0.0)
                pair_prediction_disagreement = volume_batch.new_tensor(0.0)
                pair_probability_gap = volume_batch.new_tensor(0.0)
            else:
                plus_images, minus_images, oba_metadata = \
                    orbit_balanced_augment(
                        unlabeled_images, augmentations=families,
                        gamma_magnitude=(flags.oba_gamma_min,
                                         flags.oba_gamma_max),
                        bias_magnitude=(flags.oba_bias_min,
                                        flags.oba_bias_max),
                        noise_magnitude=(flags.oba_noise_min,
                                         flags.oba_noise_max),
                        bias_grid_size=flags.oba_bias_grid_size,
                        generator=oba_generator)

                # A single symmetric batch makes both orbit endpoints share one
                # BatchNorm realization and evaluates the labeled anchor once.
                student_batch = torch.cat(
                    (labeled_images, plus_images, minus_images), dim=0)
                outputs = _logits(model(student_batch))
                with torch.no_grad():
                    ema_output = _logits(ema_model(ema_inputs))
                    pseudo_labels = base.get_masks(ema_output, nms=1)
                    # Preserve the baseline's unused one-hot construction.
                    ema_output_soft = torch.zeros_like(ema_output).scatter_(
                        1, pseudo_labels.unsqueeze(1).long(), 1.0)
                    del ema_output_soft

                unlabeled_count = unlabeled_images.shape[0]
                plus_start = flags.labeled_bs
                minus_start = plus_start + unlabeled_count
                supervised_loss, loss_ce, loss_dice = _segmentation_losses(
                    outputs[:plus_start], labeled_labels, ce_loss, dice_loss)
                plus_consistency, _, _ = _segmentation_losses(
                    outputs[plus_start:minus_start], pseudo_labels.long(),
                    ce_loss, dice_loss)
                minus_consistency, _, _ = _segmentation_losses(
                    outputs[minus_start:], pseudo_labels.long(),
                    ce_loss, dice_loss)
                consistency_loss = 0.5 * (
                    plus_consistency + minus_consistency)
                loss = supervised_loss + \
                    consistency_weight * consistency_loss
                loss.backward()
                pair_loss_gap = (
                    plus_consistency.detach() -
                    minus_consistency.detach()).abs()
                with torch.no_grad():
                    plus_probabilities = torch.softmax(
                        outputs[plus_start:minus_start], dim=1)
                    minus_probabilities = torch.softmax(
                        outputs[minus_start:], dim=1)
                    pair_prediction_disagreement = (
                        plus_probabilities.argmax(dim=1) !=
                        minus_probabilities.argmax(dim=1)).float().mean()
                    pair_probability_gap = (
                        plus_probabilities - minus_probabilities).abs().mean()
                visual_outputs = outputs.detach()

            optimizer.step()
            base.update_model_ema(
                model, ema_model, flags.ema_decay)

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
            writer.add_scalar('oba/severity_mean',
                              oba_metadata['severity'].mean(), iter_num)
            writer.add_scalar(
                'oba/plus_mean_absolute_change',
                oba_metadata['plus_mean_absolute_change'], iter_num)
            writer.add_scalar(
                'oba/minus_mean_absolute_change',
                oba_metadata['minus_mean_absolute_change'], iter_num)
            writer.add_scalar('oba/displacement_cosine',
                              oba_metadata['displacement_cosine'], iter_num)
            writer.add_scalar('oba/midpoint_drift',
                              oba_metadata['midpoint_drift'], iter_num)
            writer.add_scalar('oba/pair_span',
                              oba_metadata['pair_span'], iter_num)
            writer.add_scalar('oba/pair_loss_gap',
                              pair_loss_gap, iter_num)
            writer.add_scalar(
                'oba/pair_prediction_disagreement',
                pair_prediction_disagreement, iter_num)
            writer.add_scalar('oba/pair_probability_gap',
                              pair_probability_gap, iter_num)
            for family in families:
                writer.add_scalar(
                    'oba/family_fraction_{}'.format(family),
                    oba_metadata['family_fraction_{}'.format(family)], iter_num)

            if iter_num % 20 == 0:
                writer.add_image('train/Image',
                                 volume_batch[1, 0:1], iter_num)
                outputs_img = torch.argmax(
                    torch.softmax(visual_outputs, dim=1), dim=1,
                    keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs_img[1] * 50, iter_num)
                writer.add_image('train/GroundTruth',
                                 label_batch[1].unsqueeze(0) * 50, iter_num)
                writer.add_image('oba/OriginalUnlabeled',
                                 unlabeled_images[0], iter_num)
                writer.add_image('oba/PlusUnlabeled',
                                 plus_images[0], iter_num)
                writer.add_image('oba/MinusUnlabeled',
                                 minus_images[0], iter_num)
                writer.add_image('oba/PairMean',
                                 0.5 * (plus_images[0] + minus_images[0]),
                                 iter_num)
                writer.add_image('oba/PlusAbsoluteDelta',
                                 (plus_images[0] -
                                  unlabeled_images[0]).abs(), iter_num)
                writer.add_image('oba/MinusAbsoluteDelta',
                                 (minus_images[0] -
                                  unlabeled_images[0]).abs(), iter_num)

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
                        snapshot_path, 'iter_{}_dice_{}.pth'.format(
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
    _validate_oba_args(args)
    pretrained_checkpoint = base.resolve_pretrained_checkpoint(
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

    base.reset_stage_rng(args.seed)
    self_snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    if not os.path.exists(self_snapshot_path):
        os.makedirs(self_snapshot_path)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(self_snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info('================ START OBA SELF-TRAINING ================')
    logging.info(str(args))
    self_train(args, pretrained_checkpoint, self_snapshot_path)
