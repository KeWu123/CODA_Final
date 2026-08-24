import ast
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_adu.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_adu.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_adu.py'
PARENT_TRAIN = ROOT / 'code' / 'train_sliceeq_occ.py'
REAGGREGATOR = ROOT / 'code' / 'reaggregate_h7_10_result.py'
H7_10_RESULT = (
    ROOT / 'research' / 'experiments' / 'h7_slice_profile_reacquisition' /
    'results' / 'h7_10_operator_reliability_gate_2026-08-14.json')


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
        if node.func.attr != 'add_argument' or not node.args or \
                not isinstance(node.args[0], ast.Constant):
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


class SliceEqADUContractTest(unittest.TestCase):
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

    def test_defaults_match_parent_except_isolated_identity(self):
        defaults = parser_defaults(TRAIN)
        expected = {
            '--exp': 'SliceEqOccADU_PROMISE12',
            '--max_iterations': 30000,
            '--batch_size': 24,
            '--deterministic': 1,
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
        for forbidden in (
                '--adu_threshold', '--adu_weight', '--adu_temperature',
                '--adu_ramp', '--adu_mc_passes'):
            self.assertNotIn(forbidden, defaults)

    def test_recipe_values_and_shared_pretrain_are_runtime_locked(self):
        source = TRAIN.read_text(encoding='utf-8')
        for required in (
                "'root_path': (",
                "'max_iterations': 30000",
                "'batch_size': 24",
                "'deterministic': 1",
                "'base_lr': 0.01",
                "'patch_size': [256, 256]",
                "'seed': 1337",
                "'num_classes': 2",
                "'labeled_bs': 12",
                "'labelnum': 7",
                "'ema_decay': 0.99",
                "'consistency': 0.1",
                "'consistency_rampup': 200.0",
                "'sliceeq_sigma_min': 0.45",
                "'sliceeq_sigma_max': 0.85",
                "'sliceeq_phase_min': -0.25",
                "'sliceeq_phase_max': 0.25",
                "actual != expected",
                "_validate_pretrained_checkpoint_hash(",
                "49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9",
        ):
            self.assertIn(required, source)
        self.assertIn(
            '_validate_pretrained_checkpoint_hash(pretrained_checkpoint)',
            source)

    def test_parent_student_and_objective_are_not_attenuated(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn(
            '(labeled_images, labeled_reacquired_images,\n'
            '                     unlabeled_reacquired_images), dim=0)',
            source)
        self.assertEqual(source.count('student_batch = torch.cat('), 1)
        self.assertIn(
            'loss = supervised_loss + \\\n'
            '                    consistency_weight * consistency_loss',
            source)
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('base.update_model_ema('), 1)

    def test_adu_is_only_in_postwarmup_teacher_target_and_u_loss(self):
        source = TRAIN.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(TRAIN))
        train_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'self_train')
        train_source = ast.get_source_segment(source, train_function)
        warmup_position = train_source.index('if iter_num < 1000:')
        else_position = train_source.index('\n            else:', warmup_position)
        helper_call = train_source.index(
            '_isolated_second_teacher_logits(', warmup_position)
        self.assertGreater(helper_call, else_position)
        self.assertIn('primary_pseudo_stack', source)
        self.assertIn('secondary_pseudo_stack', source)
        self.assertIn(
            'unlabeled_stack, primary_pseudo_stack,\n'
            '                            unlabeled_weights, num_classes)',
            source)
        self.assertIn(
            'unlabeled_stack, secondary_pseudo_stack,\n'
            '                            unlabeled_weights, num_classes)',
            source)
        self.assertIn('acquisition_aligned_reliability(', source)
        self.assertIn(
            'reliability_weighted_soft_segmentation_loss(', source)

    def test_second_teacher_pass_is_bn_rng_invisible_and_train_mode(self):
        source = UTILITY.read_text(encoding='utf-8')
        self.assertIn('if not module.training:', source)
        self.assertIn('snapshot_buffers(module)', source)
        self.assertIn('restore_buffers(module, buffer_snapshot)', source)
        self.assertIn('torch.random.fork_rng(devices=[device_index])', source)
        self.assertIn('torch.cuda.manual_seed(seed)', source)
        self.assertNotIn('module.eval()', source)
        train_source = TRAIN.read_text(encoding='utf-8')
        self.assertNotIn('ema_model.eval()', train_source)
        self.assertNotIn('ema_model.train()', train_source)
        self.assertIn('ADU_SECOND_TEACHER_SEED_OFFSET = 7000003', train_source)

    def test_unlabeled_ground_truth_is_never_used(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn(
            'labeled_target_stack = label_stack[:flags.labeled_bs]', source)
        self.assertIn(
            'labeled_labels = label_batch[:flags.labeled_bs]', source)
        self.assertNotIn('label_stack[flags.labeled_bs:]', source)
        self.assertNotIn('label_batch[flags.labeled_bs:]', source)

    def test_validation_checkpoint_and_inference_are_unchanged(self):
        parent = PARENT_TRAIN.read_text(encoding='utf-8')
        successor = TRAIN.read_text(encoding='utf-8')
        self.assertEqual(validation_block(successor), validation_block(parent))
        self.assertIn('iter_num > 0 and iter_num % 200 == 0', successor)
        self.assertIn('if performance > best_performance:', successor)
        self.assertIn('if iter_num % 3000 == 0:', successor)
        defaults = parser_defaults(TEST)
        self.assertEqual(defaults['--exp'], 'SliceEqOccADU_PROMISE12')
        self.assertEqual(defaults['--save_result'], 'False')
        self.assertEqual(defaults['--auto_find_checkpoint'], 'False')
        test_source = TEST.read_text(encoding='utf-8')
        self.assertIn('net.load_state_dict(state_dict, strict=True)', test_source)
        self.assertNotIn('strict=False', test_source)

    def test_h7_10_post_run_reaggregation_is_reproducible(self):
        spec = importlib.util.spec_from_file_location(
            'h7_10_reaggregator', REAGGREGATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = json.loads(H7_10_RESULT.read_text(encoding='utf-8'))
        derived = module.reaggregate(report, H7_10_RESULT)
        self.assertEqual(derived['quality_complete_adu_pairs'], 84)
        self.assertEqual(derived['adu_named_patient_passes'], 7)
        self.assertEqual(derived['sct_named_patient_passes'], 0)
        self.assertEqual(
            derived['decision'],
            'authorize_exploratory_slice_eq_occ_adu_training')
        self.assertTrue(derived['not_a_final_hash_confirmatory_rerun'])


if __name__ == '__main__':
    unittest.main()
