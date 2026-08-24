import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'code' / 'train_sliceeq_occ.py'
TEST_PATH = ROOT / 'code' / 'test_sliceeq_occ.py'
UTIL_PATH = ROOT / 'code' / 'utils' / 'sliceeq_occ.py'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parser_defaults(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute):
            continue
        if node.func.attr != 'add_argument' or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'default':
                try:
                    defaults[node.args[0].value] = ast.literal_eval(
                        keyword.value)
                except (ValueError, TypeError):
                    pass
    return defaults


class SliceEqOccupancyContractTest(unittest.TestCase):
    def test_v1_sources_are_unchanged(self):
        expected = {
            'code/train_sliceeq.py':
                '64dbb13fd64b873c067425f61892b244725d0bfddceb401257d03d066856c12f',
            'code/test_sliceeq.py':
                '6587c72e0f73a018eb1a5611951bfbabeb276f8b4d4506b9e1e235252e1a2e15',
            'code/utils/sliceeq.py':
                '44a956a92eccdbb2109034a05ac5ec72f190b33f740ad95c1ed8b505bae168f7',
            'code/dataloaders/sliceeq_dataset.py':
                '9cc39cd6ed373e22ec854340c7975868025c2bf74223370c1eeb69e928fc19d5',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_training_defaults_keep_v1_recipe_and_use_new_identity(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            '--root_path':
                '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--exp': 'SliceEqOcc_PROMISE12',
            '--model': 'unet',
            '--max_iterations': 30000,
            '--batch_size': 24,
            '--base_lr': 0.01,
            '--patch_size': [256, 256],
            '--seed': 1337,
            '--num_classes': 2,
            '--labeled_bs': 12,
            '--labelnum': 7,
            '--ema_decay': 0.99,
            '--consistency': 0.1,
            '--consistency_rampup': 200.0,
            '--sliceeq_radius': 1,
            '--sliceeq_sigma_min': 0.45,
            '--sliceeq_sigma_max': 0.85,
            '--sliceeq_phase_min': -0.25,
            '--sliceeq_phase_max': 0.25,
        }
        for name, value in expected.items():
            self.assertEqual(defaults[name], value, name)

    def test_fractional_objective_and_effective_batch_are_explicit(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('labeled_occupancy = paired_slice_reacquisition(',
                      source)
        self.assertIn('unlabeled_occupancy = paired_slice_reacquisition(',
                      source)
        self.assertIn('soft_segmentation_loss(', source)
        self.assertIn(
            'supervised_loss = 0.5 * (', source)
        self.assertIn(
            '(labeled_images, labeled_reacquired_images,', source)
        self.assertIn('12 original-L + 12 reacquired-L + 12 reacquired-U',
                      source)
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)
        self.assertNotIn('couple_pseudo_target', source)

    def test_shared_checkpoint_is_strict_and_no_pretrain_runs(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('DEFAULT_PRETRAINED_CHECKPOINT = (', source)
        self.assertIn(
            "'UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/'",
            source)
        self.assertIn("'pre_train/unet/unet_best_model.pth'", source)
        self.assertIn('locked._load_pretrained_checkpoint(', source)
        self.assertNotIn('pre_train(', source)
        self.assertNotIn('glob.glob', source)

    def test_occupancy_utility_has_no_uniform_smoothing(self):
        source = UTIL_PATH.read_text(encoding='utf-8')
        self.assertIn('def soft_cross_entropy(', source)
        self.assertIn('def soft_dice_loss(', source)
        self.assertIn('def occupancy_diagnostics(', source)
        self.assertNotIn('full_like', source)
        self.assertNotIn('uniform', source.lower())

    def test_test_defaults_disable_prediction_saving_and_search(self):
        defaults = parser_defaults(TEST_PATH)
        self.assertEqual(defaults['--exp'], 'SliceEqOcc_PROMISE12')
        self.assertEqual(defaults['--save_result'], 'False')
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')
        source = TEST_PATH.read_text(encoding='utf-8')
        self.assertIn("'--save_result', '--save_results'", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)


if __name__ == '__main__':
    unittest.main()
