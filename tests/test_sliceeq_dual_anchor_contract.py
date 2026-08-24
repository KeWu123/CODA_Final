import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'code' / 'train_sliceeq_occ_da.py'
TEST_PATH = ROOT / 'code' / 'test_sliceeq_occ_da.py'


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


class SliceEqDualAnchorContractTest(unittest.TestCase):
    def test_parent_occ_and_cap_sources_are_unchanged(self):
        expected = {
            'code/train_sliceeq_occ.py':
                'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5',
            'code/test_sliceeq_occ.py':
                '856b235b008fb161b8f1ee4bb54c7ceb00d8a666113fa529828d0f45cf0c5386',
            'code/train_sliceeq_occ_cap.py':
                'e7981f40a024eb01a80d0263e5aa14d37b43359e89388788dfbda774caa5e005',
            'code/test_sliceeq_occ_cap.py':
                '5d5ab243313e02cee7d8e8374b6585b3d09acc6ea6ee2a7e9f22febd900848ae',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_defaults_keep_locked_recipe_and_isolated_identity(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            '--root_path':
                '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--exp': 'SliceEqOccDA_PROMISE12',
            '--max_iterations': 30000,
            '--batch_size': 24,
            '--base_lr': 0.01,
            '--patch_size': [256, 256],
            '--seed': 1337,
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

    def test_dual_anchor_is_the_only_post_warmup_objective_change(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn(
            '(labeled_images, labeled_reacquired_images,\n'
            '                     unlabeled_reacquired_images,\n'
            '                     native_unlabeled_images)', source)
        self.assertIn(
            'native_unlabeled_target = pseudo_stack[:, center]', source)
        self.assertIn(
            'measurement_consistency_dice = soft_segmentation_loss(', source)
        self.assertIn(
            'native_consistency_dice = _hard_segmentation_losses(', source)
        self.assertIn(
            'consistency_loss = 0.5 * (\n'
            '                    measurement_consistency_loss +\n'
            '                    native_consistency_loss)', source)
        self.assertIn(
            'loss = supervised_loss + \\\n'
            '                    consistency_weight * consistency_loss', source)
        self.assertNotIn('antithetic', source.lower())
        self.assertNotIn('confidence', source.lower())
        self.assertNotIn('attention', source.lower())
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)

    def test_teacher_policy_matches_baseline_train_mode(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertNotIn('ema_model.eval()', source)
        self.assertNotIn('ema_model.train()', source)
        self.assertIn('ema_output = _logits(ema_model(teacher_inputs))', source)
        self.assertIn('ema_output = _logits(ema_model(unlabeled_images))', source)

    def test_unlabeled_ground_truth_is_not_used(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn(
            'labeled_target_stack = label_stack[:flags.labeled_bs]', source)
        self.assertIn(
            'labeled_labels = label_batch[:flags.labeled_bs]', source)
        self.assertNotIn('label_stack[flags.labeled_bs:]', source)
        self.assertNotIn('label_batch[flags.labeled_bs:]', source)

    def test_checkpoint_and_inference_are_strict_and_isolated(self):
        train_source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('DEFAULT_PRETRAINED_CHECKPOINT = (', train_source)
        self.assertIn('locked._load_pretrained_checkpoint(', train_source)
        self.assertNotIn('pre_train(', train_source)
        defaults = parser_defaults(TEST_PATH)
        self.assertEqual(defaults['--exp'], 'SliceEqOccDA_PROMISE12')
        self.assertEqual(defaults['--save_result'], 'False')
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')
        test_source = TEST_PATH.read_text(encoding='utf-8')
        self.assertIn(
            'net.load_state_dict(state_dict, strict=True)', test_source)
        self.assertNotIn('strict=False', test_source)


if __name__ == '__main__':
    unittest.main()
