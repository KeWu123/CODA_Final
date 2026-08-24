import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'code' / 'train_bmer.py'
TEST_PATH = ROOT / 'code' / 'test_bmer.py'


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
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'add_argument' or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'default':
                defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    return defaults


class BMERContractTest(unittest.TestCase):
    def test_existing_training_and_evaluation_sources_are_unchanged(self):
        expected = {
            'code/train_baseline.py': '54393fcb977a3e4f199420885b6f6acbd8b1d2b320c820979f355c003cd3eec8',
            'code/test_baseline.py': '31cc57d26fb3476f55a593445e10bbc4efa96e6dbfde778baa2d65c599491682',
            'code/dataloaders/dataset.py': '7a5b3c28ebeaf7aa2f64e5f111f88bd15f6075afc53238b477bd1a8511b2206a',
            'code/train_coda.py': 'd551a2726c451b07e40a2d2563299747cab080be66da8bf0720ee218066c0799',
            'code/test_coda.py': '54697b89555e0e79ef1104047a3a4c8bf6eabedb9271453512cdc361259b338e',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(sha256(ROOT / relative_path), expected_hash,
                             relative_path)

    def test_training_defaults_preserve_current_recipe(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            '--root_path': '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--exp': 'BMER_PROMISE12',
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
            '--bmer_radius': 8,
            '--bmer_sectors': 16,
            '--bmer_position_bins': 3,
            '--bmer_probability': 0.5,
            '--bmer_strength_min': 0.5,
            '--bmer_strength_max': 1.0,
            '--bmer_min_foreground_pixels': 32,
            '--bmer_bank_batch_size': 24,
        }
        for name, value in expected.items():
            self.assertEqual(defaults[name], value, name)

    def test_test_entry_matches_training_identity(self):
        defaults = parser_defaults(TEST_PATH)
        self.assertEqual(
            defaults['--root_path'],
            '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source')
        self.assertEqual(defaults['--exp'], 'BMER_PROMISE12')
        self.assertEqual(defaults['--labelnum'], 7)
        self.assertEqual(defaults['--patch_size'], [256, 256])

    def test_bmer_changes_only_labeled_student_input(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn("ema_inputs = unlabeled_volume_batch", source)
        self.assertIn("(augmented_labeled, unlabeled_volume_batch)", source)
        self.assertIn("pseudo_labels = base.get_masks(ema_output, nms=1)", source)
        self.assertIn("unl_ce = ce_loss(unl_outputs, unl_labels)", source)
        self.assertIn("unl_dice = dice_loss(unl_outputs_soft,", source)
        self.assertNotIn('couple_pseudo_target', source)
        self.assertNotIn('soft_cross_entropy', source)
        self.assertNotIn('soft_dice_loss', source)

    def test_unlabeled_bank_dataset_never_opens_hidden_label(self):
        tree = ast.parse(TRAIN_PATH.read_text(encoding='utf-8'),
                         filename=str(TRAIN_PATH))
        dataset_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == 'UnlabeledImageDataset')
        class_source = ast.get_source_segment(
            TRAIN_PATH.read_text(encoding='utf-8'), dataset_class)
        self.assertIn("stream['image']", class_source)
        self.assertNotIn("stream['label']", class_source)


if __name__ == '__main__':
    unittest.main()
