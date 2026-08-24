"""Inference entry for OBA checkpoints.

OBA has no inference-time component, so this file reuses the current evaluator while
changing the experiment identity and disabling broad checkpoint fallback by default.
"""

import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_path', type=str,
    default='/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
    help='dataset root path')
parser.add_argument('--exp', type=str, default='OBA_PROMISE12',
                    help='experiment name, must match training --exp')
parser.add_argument('--model', type=str, default='unet', help='model name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--labelnum', type=int, default=7,
                    help='labeled data number, must match training')
parser.add_argument('--stage_name', type=str, default='self_train',
                    choices=['pre_train', 'self_train'])
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_result', type=str, default='True',
                    choices=['True', 'False', 'true', 'false', '1', '0',
                             'yes', 'no'])
parser.add_argument('--nms', type=int, default=0)
parser.add_argument('--model_root', type=str, default='../model')
parser.add_argument('--snapshot_path', type=str, default=None)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--auto_find_checkpoint', type=str, default='False',
                    choices=['True', 'False', 'true', 'false', '1', '0',
                             'yes', 'no'])
parser.add_argument('--test_save_path', type=str, default=None)
FLAGS = parser.parse_args()


def _import_evaluator():
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        import test_coda as evaluator
    finally:
        sys.argv = original_argv
    return evaluator


if __name__ == '__main__':
    base = _import_evaluator()
    metric_avg, save_path, perf_path = base.inference(FLAGS)
    print('\n================ Final Average Metrics ================')
    print('Dice:    {:.6f}'.format(metric_avg[0]))
    print('Jaccard: {:.6f}'.format(metric_avg[1]))
    print('HD95:    {:.6f}'.format(metric_avg[2]))
    print('ASD:     {:.6f}'.format(metric_avg[3]))
    print('Prediction save path: {}'.format(save_path))
    print('Performance file: {}'.format(perf_path))
