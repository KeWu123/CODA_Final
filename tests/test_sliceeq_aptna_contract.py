import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'code' / 'train_sliceeq_occ_aptna.py'
TEST_PATH = ROOT / 'code' / 'test_sliceeq_occ_aptna.py'
PARENT_TRAIN_PATH = ROOT / 'code' / 'train_sliceeq_occ.py'


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


def load_pure_function(path, function_name):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[function_name]


def validation_block(source):
    start = source.index('                model.eval()\n')
    end = source.index('\n\n            if iter_num % 3000 == 0', start)
    return source[start:end]


class SliceEqAPTNAContractTest(unittest.TestCase):
    def test_parent_occ_and_failed_da_sources_are_unchanged(self):
        expected = {
            'code/train_sliceeq_occ.py':
                'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5',
            'code/test_sliceeq_occ.py':
                '856b235b008fb161b8f1ee4bb54c7ceb00d8a666113fa529828d0f45cf0c5386',
            'code/train_sliceeq_occ_da.py':
                '65767d0555aec943ede062aa570e7621fa6f8e94f093a768f5fb861af5eb2672',
            'code/test_sliceeq_occ_da.py':
                'f3331f54b632bffc71f76def129762e9ba9d464bdddbd313b95d91a2bf7512ed',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_defaults_match_locked_occ_recipe(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            '--root_path':
                '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--exp': 'SliceEqOccAPTNA_PROMISE12',
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

    def test_transient_schedule_is_locked_and_parameter_free(self):
        weight = load_pure_function(
            TRAIN_PATH, '_transient_native_weight')
        self.assertAlmostEqual(weight(0.0, 0.5), 0.0)
        self.assertAlmostEqual(weight(0.25, 0.5), 0.0625)
        self.assertAlmostEqual(weight(0.5, 0.5), 0.0)
        self.assertAlmostEqual(weight(0.3794, 0.5), 0.04575564)
        with self.assertRaises(ValueError):
            weight(0.1, 0.0)
        defaults = parser_defaults(TRAIN_PATH)
        self.assertNotIn('--native_weight', defaults)
        self.assertNotIn('--native_cutoff', defaults)
        self.assertNotIn('--native_ramp', defaults)

    def test_full_occ_main_path_is_preserved(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn(
            '(labeled_images, labeled_reacquired_images,\n'
            '                     unlabeled_reacquired_images), dim=0)',
            source)
        self.assertIn(
            'consistency_weight * measurement_consistency_loss + \\\n'
            '                    native_consistency_weight * '
            'native_consistency_loss', source)
        self.assertNotIn(
            '0.5 * (\n                    measurement_consistency_loss',
            source)
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)

    def test_auxiliary_forward_isolates_bn_and_cuda_rng(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn('torch.random.fork_rng(devices=devices)', source)
        self.assertIn('torch.cuda.manual_seed(seed)', source)
        self.assertIn('with _frozen_batch_norm_running_stats(network):',
                      source)
        self.assertIn(
            'isinstance(module, torch.nn.modules.batchnorm._BatchNorm)',
            source)
        self.assertIn('module.eval()', source)
        self.assertIn('module.train(was_training)', source)
        self.assertNotIn('ema_model.eval()', source)
        self.assertNotIn('ema_model.train()', source)

    def test_unlabeled_ground_truth_is_not_used(self):
        source = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertIn(
            'native_unlabeled_target = pseudo_stack[:, center]', source)
        self.assertIn(
            'labeled_target_stack = label_stack[:flags.labeled_bs]', source)
        self.assertIn(
            'labeled_labels = label_batch[:flags.labeled_bs]', source)
        self.assertNotIn('label_stack[flags.labeled_bs:]', source)
        self.assertNotIn('label_batch[flags.labeled_bs:]', source)

    def test_validation_and_checkpoint_rule_are_exact_parent_copy(self):
        parent = PARENT_TRAIN_PATH.read_text(encoding='utf-8')
        successor = TRAIN_PATH.read_text(encoding='utf-8')
        self.assertEqual(validation_block(successor), validation_block(parent))
        self.assertIn('iter_num > 0 and iter_num % 200 == 0', successor)
        self.assertIn('if performance > best_performance:', successor)

    def test_inference_is_strict_and_isolated(self):
        defaults = parser_defaults(TEST_PATH)
        self.assertEqual(defaults['--exp'], 'SliceEqOccAPTNA_PROMISE12')
        self.assertEqual(defaults['--save_result'], 'False')
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')
        source = TEST_PATH.read_text(encoding='utf-8')
        self.assertIn(
            'net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)


if __name__ == '__main__':
    unittest.main()
