import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_scpo.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_scpo.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_scpo.py'
PARENT_TRAIN = ROOT / 'code' / 'train_sliceeq_occ.py'


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
        if node.func.attr != 'add_argument' or not node.args or not isinstance(
                node.args[0], ast.Constant):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'default':
                try:
                    defaults[node.args[0].value] = ast.literal_eval(
                        keyword.value)
                except (ValueError, TypeError):
                    pass
    return defaults


def validation_block(source):
    start = source.index('                model.eval()\n')
    end = source.index('\n\n            if iter_num % 3000 == 0', start)
    return source[start:end]


class SliceEqSCPOContractTest(unittest.TestCase):
    def test_parent_sources_are_frozen(self):
        expected = {
            'code/train_sliceeq_occ.py':
                'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5',
            'code/test_sliceeq_occ.py':
                '856b235b008fb161b8f1ee4bb54c7ceb00d8a666113fa529828d0f45cf0c5386',
            'code/utils/sliceeq.py':
                '44a956a92eccdbb2109034a05ac5ec72f190b33f740ad95c1ed8b505bae168f7',
            'code/utils/sliceeq_occ.py':
                'de2fc77c40de4d543e6aa682589aaae652887de9429742d08feacc97e50d6080',
            'code/dataloaders/sliceeq_dataset.py':
                '9cc39cd6ed373e22ec854340c7975868025c2bf74223370c1eeb69e928fc19d5',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_recipe_is_runtime_locked(self):
        defaults = parser_defaults(TRAIN)
        self.assertEqual(defaults['--exp'], 'SliceEqOccSCPO_PROMISE12')
        expected_defaults = {
            '--max_iterations': 30000,
            '--batch_size': 24,
            '--deterministic': 1,
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
        for name, expected in expected_defaults.items():
            self.assertEqual(defaults[name], expected, name)
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn("actual != expected", source)
        self.assertIn(
            '49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9',
            source)
        self.assertIn(
            '_validate_pretrained_checkpoint_hash(pretrained_checkpoint)',
            source)

    def test_only_u_source_topology_changes_after_warmup(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn('raw_pseudo_stack = base.get_masks(', source)
        self.assertIn('ema_output, nms=0', source)
        self.assertIn('parent_2d_pseudo_stack = base.get_2DLargestCC(', source)
        self.assertEqual(
            source.count('slab_largest_connected_component('), 1)
        self.assertIn(
            'unlabeled_stack, pseudo_stack,\n'
            '                            unlabeled_weights, num_classes)',
            source)
        self.assertIn(
            'labeled_stack, labeled_target_stack,\n'
            '                            labeled_weights, num_classes)',
            source)
        self.assertNotIn('ema_model.eval()', source)
        self.assertNotIn('3DLargest', source)

    def test_model_batch_loss_and_update_contract_are_parent(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertEqual(source.count('student_batch = torch.cat('), 1)
        self.assertIn(
            '(labeled_images, labeled_reacquired_images,\n'
            '                     unlabeled_reacquired_images), dim=0)',
            source)
        self.assertIn(
            'consistency_loss, consistency_ce, consistency_dice = \\\n'
            '                    soft_segmentation_loss(', source)
        self.assertIn(
            'loss = supervised_loss + \\\n'
            '                    consistency_weight * consistency_loss',
            source)
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)

    def test_utility_is_fixed_26_connected_binary_slab_lcc(self):
        source = UTILITY.read_text(encoding='utf-8')
        self.assertIn("hard_stack.shape[1] != 3", source)
        self.assertIn('connectivity=3', source)
        self.assertIn('np.argmax(counts)', source)
        self.assertNotIn('threshold', source)
        self.assertNotIn('binary_closing', source)
        self.assertNotIn('binary_dilation', source)

    def test_unlabeled_ground_truth_is_never_used(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn(
            'labeled_target_stack = label_stack[:flags.labeled_bs]', source)
        self.assertIn(
            'labeled_labels = label_batch[:flags.labeled_bs]', source)
        self.assertNotIn('label_stack[flags.labeled_bs:]', source)
        self.assertNotIn('label_batch[flags.labeled_bs:]', source)

    def test_validation_and_inference_are_unchanged(self):
        parent = PARENT_TRAIN.read_text(encoding='utf-8')
        successor = TRAIN.read_text(encoding='utf-8')
        self.assertEqual(validation_block(successor), validation_block(parent))
        defaults = parser_defaults(TEST)
        self.assertEqual(defaults['--exp'], 'SliceEqOccSCPO_PROMISE12')
        self.assertEqual(defaults['--save_result'], 'False')
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')
        test_source = TEST.read_text(encoding='utf-8')
        self.assertIn('net.load_state_dict(state_dict, strict=True)', test_source)
        self.assertNotIn('strict=False', test_source)


if __name__ == '__main__':
    unittest.main()
