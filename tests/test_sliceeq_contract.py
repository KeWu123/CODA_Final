import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'code' / 'train_sliceeq.py'
TEST_PATH = ROOT / 'code' / 'test_sliceeq.py'
DATASET_PATH = ROOT / 'code' / 'dataloaders' / 'sliceeq_dataset.py'


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
                    # Named constants are asserted directly from source below.
                    pass
    return defaults


class SliceEqContractTest(unittest.TestCase):
    def test_locked_sources_are_unchanged(self):
        expected = {
            'code/train_baseline.py':
                '54393fcb977a3e4f199420885b6f6acbd8b1d2b320c820979f355c003cd3eec8',
            'code/test_baseline.py':
                '31cc57d26fb3476f55a593445e10bbc4efa96e6dbfde778baa2d65c599491682',
            'code/dataloaders/dataset.py':
                '7a5b3c28ebeaf7aa2f64e5f111f88bd15f6075afc53238b477bd1a8511b2206a',
            'code/train_coda.py':
                'd551a2726c451b07e40a2d2563299747cab080be66da8bf0720ee218066c0799',
            'code/train_oba.py':
                '74eceef1f8ce75f1e5388940f0ba6d4cbb67b28c2769d0786ce3c8c939362d45',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_training_defaults_preserve_recipe_and_lock_profile(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            '--root_path':
                '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--exp': 'SliceEq_PROMISE12',
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

    def test_pretrain_path_uses_fixed_baseline_default_and_is_strict(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn(
            "DEFAULT_PRETRAINED_CHECKPOINT = (", source)
        self.assertIn(
            "'UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/'",
            source)
        self.assertIn(
            "'pre_train/unet/unet_best_model.pth'", source)
        self.assertIn('default=DEFAULT_PRETRAINED_CHECKPOINT', source)
        self.assertNotIn(
            "'--pretrained_checkpoint', type=str, required=True", source)
        self.assertIn("'net' not in checkpoint", source)
        self.assertIn("'opt' not in checkpoint", source)
        self.assertIn("model.load_state_dict(checkpoint['net'], strict=True)",
                      source)
        self.assertNotIn('glob.glob', source)

    def test_single_intervention_and_one_update(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('if iter_num < 1000:', source)
        self.assertIn('paired_slice_reacquisition(', source)
        self.assertIn('(labeled_images, reacquired_images)', source)
        self.assertIn('pseudo_labels.long()', source)
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)
        self.assertNotIn('soft_cross_entropy', source)
        self.assertNotIn('soft_dice_loss', source)
        self.assertNotIn('couple_pseudo_target', source)

    def test_dataset_is_parallel_and_strictly_adjacent(self):
        source = DATASET_PATH.read_text(encoding='utf-8')
        self.assertIn("SLICE_MARKER = '_slice_'", source)
        self.assertIn('Non-contiguous SliceEq indices', source)
        self.assertIn('resolved = min(max(requested, first_index), last_index)',
                      source)

    def test_test_entry_matches_identity_and_disables_search(self):
        defaults = parser_defaults(TEST_PATH)
        self.assertEqual(defaults['--exp'], 'SliceEq_PROMISE12')
        self.assertEqual(defaults['--labelnum'], 7)
        self.assertEqual(defaults['--patch_size'], [256, 256])
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')
        source = TEST_PATH.read_text(encoding='utf-8')
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)


if __name__ == '__main__':
    unittest.main()
