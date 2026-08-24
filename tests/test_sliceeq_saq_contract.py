import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'code' / 'train_sliceeq_saq.py'
TEST_PATH = ROOT / 'code' / 'test_sliceeq_saq.py'
UTIL_PATH = ROOT / 'code' / 'utils' / 'sliceeq_saq.py'


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


class SliceEqSAQContractTest(unittest.TestCase):
    def test_parent_sliceeq_occ_sources_are_unchanged(self):
        expected = {
            'code/train_sliceeq_occ.py':
                'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5',
            'code/test_sliceeq_occ.py':
                '856b235b008fb161b8f1ee4bb54c7ceb00d8a666113fa529828d0f45cf0c5386',
            'code/utils/sliceeq_occ.py':
                'de2fc77c40de4d543e6aa682589aaae652887de9429742d08feacc97e50d6080',
            'code/utils/sliceeq.py':
                '44a956a92eccdbb2109034a05ac5ec72f190b33f740ad95c1ed8b505bae168f7',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_training_defaults_keep_sliceeq_occ_recipe(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            '--root_path':
                '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--exp': 'SliceEqSAQ_PROMISE12',
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

    def test_only_profile_sampler_changes_in_training_path(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('sample_stratified_slice_profiles(', source)
        self.assertNotIn('sample_slice_profiles(', source)
        self.assertIn('quadrature_assignment_diagnostics(', source)
        self.assertIn('soft_segmentation_loss(', source)
        self.assertIn(
            '(labeled_images, labeled_reacquired_images,', source)
        self.assertIn(
            '12 original-L + 12 reacquired-L + 12 reacquired-U', source)
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)
        self.assertNotIn('confidence', source.lower())
        self.assertNotIn('posterior', source.lower())

    def test_sampler_is_balanced_and_not_task_adaptive(self):
        source = UTIL_PATH.read_text(encoding='utf-8')
        self.assertIn('repeat_interleave(repeats)', source)
        self.assertIn('torch.randperm(', source)
        self.assertIn('1.0 / math.sqrt(3.0)', source)
        self.assertIn(
            'batch_size, offsets, sigma_range, phase_range, device,',
            source)
        for forbidden in (
                'image_stack', 'mask_stack', 'target_stack',
                'confidence=', 'loss='):
            self.assertNotIn(forbidden, source.lower())

    def test_shared_checkpoint_and_no_pretrain_run(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('DEFAULT_PRETRAINED_CHECKPOINT = (', source)
        self.assertIn('locked._load_pretrained_checkpoint(', source)
        self.assertNotIn('pre_train(', source)
        self.assertNotIn('glob.glob', source)

    def test_test_defaults_are_isolated_and_safe(self):
        defaults = parser_defaults(TEST_PATH)
        self.assertEqual(defaults['--exp'], 'SliceEqSAQ_PROMISE12')
        self.assertEqual(defaults['--save_result'], 'False')
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')
        source = TEST_PATH.read_text(encoding='utf-8')
        self.assertIn("'--save_result', '--save_results'", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)


if __name__ == '__main__':
    unittest.main()
