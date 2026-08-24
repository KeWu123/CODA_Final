import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'code' / 'train_oba.py'
TEST_PATH = ROOT / 'code' / 'test_oba.py'


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
                defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    return defaults


class OBAContractTest(unittest.TestCase):
    def test_existing_sources_are_unchanged(self):
        expected = {
            'code/train_baseline.py':
                '54393fcb977a3e4f199420885b6f6acbd8b1d2b320c820979f355c003cd3eec8',
            'code/test_baseline.py':
                '31cc57d26fb3476f55a593445e10bbc4efa96e6dbfde778baa2d65c599491682',
            'code/dataloaders/dataset.py':
                '7a5b3c28ebeaf7aa2f64e5f111f88bd15f6075afc53238b477bd1a8511b2206a',
            'code/train_coda.py':
                'd551a2726c451b07e40a2d2563299747cab080be66da8bf0720ee218066c0799',
            'code/test_coda.py':
                '54697b89555e0e79ef1104047a3a4c8bf6eabedb9271453512cdc361259b338e',
            'code/train_bmer.py':
                'a88c5c8d1973e123baf96608e20f90ec705b8c31416969208ec24bd2cfd9dba9',
            'code/test_bmer.py':
                '02908d7ead2744deeb47e4bf1b11570888a2e3972b0529b6257b8dac3829db28',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_training_defaults_preserve_current_recipe(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            '--root_path':
                '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--exp': 'OBA_PROMISE12',
            '--model': 'unet',
            '--pre_iterations': 10000,
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
            '--pretrained_checkpoint':
                '/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth',
            '--oba_augmentations':
                'log_gamma,smooth_bias,gaussian_noise',
            '--oba_gamma_min': 0.10,
            '--oba_gamma_max': 0.40,
            '--oba_bias_min': 0.10,
            '--oba_bias_max': 0.35,
            '--oba_noise_min': 0.05,
            '--oba_noise_max': 0.15,
            '--oba_bias_grid_size': 8,
        }
        for name, value in expected.items():
            self.assertEqual(defaults[name], value, name)

    def test_test_entry_matches_training_identity_and_is_strict(self):
        defaults = parser_defaults(TEST_PATH)
        self.assertEqual(
            defaults['--root_path'],
            '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source')
        self.assertEqual(defaults['--exp'], 'OBA_PROMISE12')
        self.assertEqual(defaults['--labelnum'], 7)
        self.assertEqual(defaults['--patch_size'], [256, 256])
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')

    def test_oba_keeps_clean_teacher_and_hard_baseline_target(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('ema_inputs = unlabeled_images', source)
        self.assertIn('pseudo_labels = base.get_masks(ema_output, nms=1)',
                      source)
        self.assertIn('pseudo_labels.long()', source)
        self.assertIn('(labeled_images, plus_images, minus_images)', source)
        self.assertIn('consistency_loss = 0.5 * (', source)
        self.assertIn('outputs[plus_start:minus_start]', source)
        self.assertIn('outputs[minus_start:]', source)
        self.assertNotIn('couple_pseudo_target', source)
        self.assertNotIn('soft_cross_entropy', source)
        self.assertNotIn('soft_dice_loss', source)
        self.assertNotIn('branch_order', source)

    def test_one_optimizer_and_ema_update_per_iteration(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)
        self.assertEqual(source.count('student_batch = torch.cat('), 1)
        self.assertIn('if iter_num < 1000:', source)

    def test_shared_pretrain_is_loaded_without_rerunning_pretrain(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('base.load_pretrained_checkpoint(', source)
        self.assertIn('base.checkpoint_sha256(pretrained_checkpoint)', source)
        self.assertIn('base.reset_stage_rng(flags.seed)', source)
        self.assertNotIn('base.pre_train(args,', source)
        self.assertNotIn('glob.glob', source)


if __name__ == '__main__':
    unittest.main()
