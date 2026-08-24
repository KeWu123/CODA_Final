import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / 'code' / 'analyze_sliceeq_reliability_gate.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_reliability.py'
PROTOCOL = (
    ROOT / 'research' / 'experiments' / 'h7_slice_profile_reacquisition' /
    'h7_10_operator_reliability_gate_protocol.md')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def assigned_constants(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or \
                not isinstance(node.targets[0], ast.Name):
            continue
        try:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return constants


def parser_defaults(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    constants = assigned_constants(path)
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        for keyword in node.keywords:
            if keyword.arg != 'default':
                continue
            try:
                defaults[node.args[0].value] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                if isinstance(keyword.value, ast.Name):
                    defaults[node.args[0].value] = constants.get(
                        keyword.value.id)
    return defaults


class SliceEqReliabilityGateContractTest(unittest.TestCase):
    def test_frozen_parent_sources_are_unchanged(self):
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
            'code/networks/unet.py':
                '8f1318a99f2fda9d781a2c8bc6c69b0304a21a53f95c038a2faffe1d456f1a7e',
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(ROOT / relative_path), expected_hash, relative_path)

    def test_gate_defaults_are_locked(self):
        defaults = parser_defaults(ANALYZER)
        expected = {
            '--checkpoint_steps': [18000, 24000, 30000],
            '--device': 'cuda',
            '--seed': 1337,
            '--batch_schedule_seeds': [1337, 7331],
            '--labeled_slices': 191,
            '--labelnum': 7,
            '--batch_size': 12,
            '--mc_draws': 8,
            '--patch_size': [256, 256],
            '--sliceeq_radius': 1,
            '--sliceeq_sigma_min': 0.45,
            '--sliceeq_sigma_max': 0.85,
            '--sliceeq_phase_min': -0.25,
            '--sliceeq_phase_max': 0.25,
        }
        for name, value in expected.items():
            self.assertEqual(defaults[name], value, name)

    def test_gate_is_read_only_strict_and_train_mode(self):
        source = ANALYZER.read_text(encoding='utf-8')
        lowered = source.lower()
        self.assertIn('model.load_state_dict(state_dict, strict=True)', source)
        self.assertIn('model.requires_grad_(False)', source)
        self.assertIn('model.train()', source)
        self.assertIn("'analysis_kind': 'read_only_zero_training'", source)
        self.assertIn('os.replace(temporary_path, output_path)', source)
        self.assertNotIn('optimizer', lowered)
        self.assertNotIn('.backward(', source)
        self.assertNotIn('torch.save(', source)

    def test_only_labeled_training_lists_and_h5_labels_are_accessed(self):
        source = ANALYZER.read_text(encoding='utf-8')
        self.assertIn("'train.list'", source)
        self.assertIn("'train_slices.list'", source)
        self.assertNotIn("'val.list'", source)
        self.assertNotIn("'test.list'", source)
        self.assertIn('LabeledOnlySliceStackDataSets', source)
        self.assertIn('blocked non-labeled H5 access', source)
        self.assertIn("'unlabeled_h5_label_reads': 0", source)
        self.assertIn("'validation_label_reads': 0", source)
        self.assertIn("'test_label_reads': 0", source)

    def test_student_proxy_and_no_ema_reconstruction_are_explicit(self):
        source = ANALYZER.read_text(encoding='utf-8')
        protocol = ' '.join(
            PROTOCOL.read_text(encoding='utf-8').split())
        self.assertGreaterEqual(
            source.count("'student_as_proxy_teacher'"), 2)
        self.assertIn("'checkpoint_contains_ema_teacher': False", source)
        self.assertIn('frozen-student proxy gate', protocol)

    def test_preregistered_thresholds_match_protocol(self):
        constants = assigned_constants(ANALYZER)
        expected = {
            'SCT_MIN_RESIDUAL_VARIANCE_REDUCTION': 0.15,
            'SCT_MIN_RESIDUAL_BRIER_REDUCTION': 0.05,
            'SCT_MAX_FULL_BRIER_RATIO': 1.01,
            'SCT_MIN_CENTER_DICE_DELTA': -0.002,
            'ADU_MIN_SPEARMAN': 0.25,
            'ADU_MIN_TOP20_ERROR_RATIO': 1.50,
            'ADU_MIN_WEIGHTED_BRIER_REDUCTION': 0.05,
            'ADU_MIN_FRACTIONAL_WEIGHT': 0.90,
            'MIN_PATIENT_PASSES': 5,
            'MIN_CHECKPOINT_PASSES': 2,
            'ADU_SELECTION_MARGIN': 0.02,
        }
        for name, value in expected.items():
            self.assertEqual(constants[name], value, name)
        protocol = ' '.join(
            PROTOCOL.read_text(encoding='utf-8').split())
        for phrase in (
                'at least 15%', 'at least 5%', 'more than 1%',
                '0.002 absolute', 'at least 0.25', '1.5 times',
                'at least 0.90', 'at least 5/7', 'at least 2/3'):
            self.assertIn(phrase, protocol)

    def test_method_is_not_case_specific_or_postprocessing(self):
        combined = (
            ANALYZER.read_text(encoding='utf-8') +
            UTILITY.read_text(encoding='utf-8')).lower()
        for forbidden in ('case05', 'case36', 'case45', 'case49', '3d_lcc'):
            self.assertNotIn(forbidden, combined)
        self.assertIn('per-slice 2D LCC', ANALYZER.read_text(encoding='utf-8'))
        self.assertIn('temporary_stack_dropout', combined)
        self.assertIn('jensen_shannon_map', combined)

    def test_decision_requires_patient_and_checkpoint_consistency(self):
        source = ANALYZER.read_text(encoding='utf-8')
        self.assertIn('_cross_checkpoint_patient_consistency', source)
        self.assertIn('sct_passes >= MIN_CHECKPOINT_PASSES', source)
        self.assertIn('adu_passes >= MIN_CHECKPOINT_PASSES', source)
        self.assertIn('sct_patient_passes >= MIN_PATIENT_PASSES', source)
        self.assertIn('adu_patient_passes >= MIN_PATIENT_PASSES', source)
        self.assertIn("selected = 'sct'", source)
        self.assertIn("selected = 'adu'", source)
        self.assertIn("selected = None", source)

    def test_locked_checkpoint_and_complete_pair_guards_are_executable(self):
        source = ANALYZER.read_text(encoding='utf-8')
        self.assertIn(
            'list(args.checkpoint_steps) != DEFAULT_CHECKPOINT_STEPS',
            source)
        self.assertIn("stem.startswith(expected + '_dice_')", source)
        self.assertIn("'all_pairs_complete': bool(all_pairs_complete)", source)
        self.assertIn(
            'complete_checkpoint_count >= MIN_CHECKPOINT_PASSES', source)
        self.assertIn('json.dump(', source)
        self.assertIn('allow_nan=False', source)


if __name__ == '__main__':
    unittest.main()
