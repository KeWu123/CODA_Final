"""BMER training entry built on the locked PROMISE12 baseline recipe.

The existing CoDA entry is imported only to reuse its exact U-Net, supervised
pretraining, validation, EMA, LCC, and loss helpers.  BMER changes one tensor in
self-training: the labeled student input.  The unlabeled input and hard pseudo-target
path remain the baseline path.
"""

import argparse
from functools import partial
import logging
import os
import random
import shutil
import sys

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from utils import losses, val_2d
from utils.bmer import (BoundaryProfileBank, build_position_bin_map,
                        lookup_position_bins, resynthesize_labeled_images)
from utils.promise12_preflight import validate_promise12_root


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BMER_PROMISE12', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int,
                    default=10000, help='maximum epoch number to pre-train')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
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
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default='mse', help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# BMER v1 parameters. Baseline defaults above are unchanged from train_coda.py.
parser.add_argument('--bmer_radius', type=int, default=8,
                    help='signed-distance ribbon radius in pixels')
parser.add_argument('--bmer_sectors', type=int, default=16,
                    help='angular sectors approximating boundary tangential position')
parser.add_argument('--bmer_position_bins', type=int, default=3,
                    help='normalized within-volume slice-position bins')
parser.add_argument('--bmer_probability', type=float, default=0.5,
                    help='per-labeled-sample BMER application probability')
parser.add_argument('--bmer_strength_min', type=float, default=0.5,
                    help='minimum profile interpolation strength')
parser.add_argument('--bmer_strength_max', type=float, default=1.0,
                    help='maximum profile interpolation strength')
parser.add_argument('--bmer_min_foreground_pixels', type=int, default=32,
                    help='minimum foreground pixels for a valid profile')
parser.add_argument('--bmer_bank_batch_size', type=int, default=24,
                    help='deterministic unlabeled-bank inference batch size')
args = parser.parse_args()


def _import_locked_training_base():
    """Import train_coda without letting its module parser consume BMER arguments."""
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        import train_coda as training_base
    finally:
        sys.argv = original_argv
    training_base.args = args
    return training_base


base = _import_locked_training_base()


class UnlabeledImageDataset(Dataset):
    """Read only image tensors for deterministic bank construction.

    Hidden H5 labels are intentionally never opened by this dataset.
    """

    def __init__(self, root_path, sample_names, output_size):
        self.root_path = root_path
        self.sample_names = list(sample_names)
        self.output_size = tuple(output_size)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        case = self.sample_names[index]
        path = os.path.join(self.root_path, 'data', 'slices', case + '.h5')
        with h5py.File(path, 'r') as stream:
            image = stream['image'][:]
        height, width = image.shape
        if (height, width) != self.output_size:
            image = zoom(image, (self.output_size[0] / height,
                                 self.output_size[1] / width), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        return {'image': image, 'case': case}


def seed_data_worker(worker_id, base_seed):
    """Windows-spawn-safe worker callback matching the current CoDA entry."""
    random.seed(base_seed + worker_id)


def _validate_bmer_args(flags):
    if flags.num_classes != 2:
        raise ValueError('BMER v1 expects binary segmentation')
    if flags.bmer_radius < 1 or flags.bmer_sectors < 1:
        raise ValueError('BMER radius/sectors must be positive')
    if flags.bmer_position_bins < 1 or flags.bmer_min_foreground_pixels < 1:
        raise ValueError('BMER position bins/min foreground must be positive')
    if flags.bmer_bank_batch_size < 1:
        raise ValueError('BMER bank batch size must be positive')
    if not 0.0 <= flags.bmer_probability <= 1.0:
        raise ValueError('BMER probability must be in [0, 1]')
    if (flags.bmer_strength_min < 0.0 or
            flags.bmer_strength_max < flags.bmer_strength_min):
        raise ValueError('invalid BMER strength range')


def _build_frozen_profile_bank(flags, bank_model, sample_names,
                               position_bin_map, snapshot_path):
    unlabeled_dataset = UnlabeledImageDataset(
        flags.root_path, sample_names, flags.patch_size)
    loader = DataLoader(unlabeled_dataset,
                        batch_size=flags.bmer_bank_batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    bank = BoundaryProfileBank(
        radius=flags.bmer_radius,
        sectors=flags.bmer_sectors,
        position_bins=flags.bmer_position_bins,
        min_foreground_pixels=flags.bmer_min_foreground_pixels)

    bank_model.eval()
    with torch.no_grad():
        for sampled_batch in tqdm(loader, ncols=70, desc='BMER bank'):
            images = sampled_batch['image'].cuda(non_blocking=True)
            logits = bank_model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            pseudo_masks = base.get_masks(logits, nms=1)
            case_names = list(sampled_batch['case'])
            position_bins = lookup_position_bins(case_names, position_bin_map)
            bank.add(images, pseudo_masks, position_bins, case_names)

    bank.freeze()
    bank_path = os.path.join(snapshot_path, 'bmer_profile_bank.pt')
    bank.save(bank_path)
    logging.info('Frozen BMER bank: %s', bank.summary())
    logging.info('Saved BMER bank to %s', bank_path)
    return bank


def self_train(flags, pre_snapshot_path, snapshot_path):
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
    optimizer = base.optim.SGD(model.parameters(), lr=base_lr,
                               momentum=0.9, weight_decay=0.0001)

    pre_trained_model = os.path.join(
        pre_snapshot_path, '{}_best_model.pth'.format(flags.model))
    checkpoint = torch.load(pre_trained_model)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
        ema_model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['opt'])
        logging.info('Loaded pre-trained weights and optimizer from %s',
                     pre_trained_model)
    else:
        model.load_state_dict(checkpoint)
        ema_model.load_state_dict(checkpoint)
        logging.info('Loaded pre-trained weights from %s', pre_trained_model)

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
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    position_bin_map = build_position_bin_map(
        db_train.sample_list, flags.bmer_position_bins)
    bank_model = create_model(ema=True)
    bank_model.load_state_dict(model.state_dict())
    profile_bank = _build_frozen_profile_bank(
        flags, bank_model, db_train.sample_list[labeled_slice:],
        position_bin_map, snapshot_path)
    del bank_model
    torch.cuda.empty_cache()

    ce_loss = base.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    writer = base.SummaryWriter(snapshot_path + '/log')
    logging.info('%s iterations per epoch', len(trainloader))
    for position_bin, count in profile_bank.summary()['counts'].items():
        writer.add_scalar('bmer/bank_count_bin_{}'.format(position_bin), count, 0)
    writer.add_scalar('bmer/bank_total', len(profile_bank), 0)

    bmer_generator = torch.Generator(device='cpu')
    bmer_generator.manual_seed(flags.seed)
    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            unlabeled_volume_batch = volume_batch[flags.labeled_bs:]
            labeled_case_names = list(sampled_batch['case'][:flags.labeled_bs])
            labeled_position_bins = lookup_position_bins(
                labeled_case_names, position_bin_map)

            augmented_labeled, bmer_metadata = resynthesize_labeled_images(
                volume_batch[:flags.labeled_bs],
                label_batch[:flags.labeled_bs], profile_bank,
                labeled_position_bins,
                probability=flags.bmer_probability,
                strength=(flags.bmer_strength_min, flags.bmer_strength_max),
                generator=bmer_generator)
            student_volume_batch = torch.cat(
                (augmented_labeled, unlabeled_volume_batch), dim=0)

            # Locked baseline teacher input: original unlabeled loader tensor.
            ema_inputs = unlabeled_volume_batch
            outputs = model(student_volume_batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                if isinstance(ema_output, tuple):
                    ema_output = ema_output[0]
                pseudo_labels = base.get_masks(ema_output, nms=1)
                # Preserve the baseline's unused one-hot construction as part of the
                # locked computation path.
                ema_output_soft = torch.zeros_like(ema_output).scatter_(
                    1, pseudo_labels.unsqueeze(1).long(), 1.0)
                del ema_output_soft

            loss_ce = ce_loss(outputs[:flags.labeled_bs],
                              label_batch[:flags.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:flags.labeled_bs],
                label_batch[:flags.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = base.get_current_consistency_weight(iter_num // 150)

            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                unl_outputs = outputs[flags.labeled_bs:]
                unl_outputs_soft = outputs_soft[flags.labeled_bs:]
                unl_labels = pseudo_labels.long()
                unl_ce = ce_loss(unl_outputs, unl_labels)
                unl_dice = dice_loss(unl_outputs_soft,
                                     unl_labels.unsqueeze(1))
                consistency_loss = 0.5 * (unl_dice + unl_ce)

            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            base.update_model_ema(model, ema_model, flags.ema_decay)

            lr_ = base_lr
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            for name in ('applied_fraction', 'valid_mask_fraction',
                         'strength_mean', 'changed_fraction',
                         'mean_absolute_change'):
                writer.add_scalar('bmer/{}'.format(name),
                                  bmer_metadata[name], iter_num)

            if iter_num % 20 == 0:
                writer.add_image('train/Image',
                                 volume_batch[1, 0:1], iter_num)
                outputs_img = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs_img[1] * 50, iter_num)
                writer.add_image('train/GroundTruth',
                                 label_batch[1].unsqueeze(0) * 50, iter_num)
                writer.add_image('bmer/OriginalLabeled',
                                 volume_batch[0], iter_num)
                writer.add_image('bmer/AugmentedLabeled',
                                 augmented_labeled[0], iter_num)
                writer.add_image('bmer/AbsoluteDelta',
                                 (augmented_labeled[0] -
                                  volume_batch[0]).abs(), iter_num)

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
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
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
    _validate_bmer_args(args)
    dataset_report = validate_promise12_root(
        args.root_path, strict_split=True, check_hdf5=True)
    print('PROMISE12 preflight: {}'.format(dataset_report))

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = '../model/{}_{}_labeled/pre_train/{}'.format(
        args.exp, args.labelnum, args.model)
    self_snapshot_path = '../model/{}_{}_labeled/self_train/{}'.format(
        args.exp, args.labelnum, args.model)
    for snapshot_path in (pre_snapshot_path, self_snapshot_path):
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    if os.path.exists(self_snapshot_path + '/code'):
        shutil.rmtree(self_snapshot_path + '/code')

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(pre_snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info(str(args))
    base.pre_train(args, pre_snapshot_path)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(self_snapshot_path + '/log.txt'),
                  logging.StreamHandler(sys.stdout)], force=True)
    logging.info('================ START BMER SELF-TRAINING ================')
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
