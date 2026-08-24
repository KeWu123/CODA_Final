import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / 'code' / 'analyze_sliceeq_gates.py'
PROTOCOL_PATH = (
    ROOT / 'research' / 'experiments' / 'h7_slice_profile_reacquisition' /
    'h7_3_gate_protocol.md')


def parser_defaults(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
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
                    if isinstance(keyword.value, ast.Name) and \
                            keyword.value.id in constants:
                        defaults[node.args[0].value] = constants[
                            keyword.value.id]
    return defaults


class SliceEqGatesContractTest(unittest.TestCase):
    def test_locked_checkpoint_split_and_operator_defaults(self):
        defaults = parser_defaults(ANALYSIS_PATH)
        expected = {
            '--root_path':
                '/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source',
            '--checkpoint':
                '/home/aiteam/zhengtaoma/CoDA/model/'
                'SliceEqOcc_PROMISE12_7_labeled/self_train/unet/'
                'iter_23000_dice_0.8152.pth',
            '--device': 'cuda',
            '--seed': 1337,
            '--labeled_slices': 191,
            '--batch_size': 12,
            '--max_labeled_batches': 16,
            '--max_unlabeled_batches': 16,
            '--sliceeq_radius': 1,
            '--sliceeq_sigma_min': 0.45,
            '--sliceeq_sigma_max': 0.85,
            '--sliceeq_phase_min': -0.25,
            '--sliceeq_phase_max': 0.25,
        }
        for name, value in expected.items():
            self.assertEqual(defaults[name], value, name)

    def test_analysis_is_read_only_and_uses_strict_checkpoint_loading(self):
        source = ANALYSIS_PATH.read_text(encoding='utf-8')
        self.assertIn('model.load_state_dict(state_dict, strict=True)', source)
        self.assertIn('model.eval()', source)
        self.assertIn("'analysis_kind': 'read_only_no_training'", source)
        self.assertNotIn('optimizer', source.lower())
        self.assertNotIn('torch.save(', source)
        self.assertNotIn('loss.backward(', source)

    def test_preregistered_thresholds_match_protocol(self):
        source = ANALYSIS_PATH.read_text(encoding='utf-8')
        protocol = PROTOCOL_PATH.read_text(encoding='utf-8')
        self.assertIn('GATE1_MAX_GRADIENT_SHARE = 0.20', source)
        self.assertIn('GATE2_MIN_PIXEL_CORRELATION = 0.30', source)
        self.assertIn('GATE2_MIN_MASS_CORRELATION = 0.30', source)
        self.assertIn('GATE2_MAX_OUTSIDE_MASS = 0.50', source)
        self.assertIn('GATE3_MAX_GRADIENT_COSINE = 0.98', source)
        for threshold in ('`0.20`', '`0.30`', '`0.50`', '`0.98`'):
            self.assertIn(threshold, protocol)

    def test_unlabeled_ground_truth_is_not_used_by_gate_statistics(self):
        source = ANALYSIS_PATH.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(ANALYSIS_PATH))
        run_unlabeled = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and
            node.name == '_run_unlabeled')
        function_source = ast.get_source_segment(source, run_unlabeled)
        self.assertNotIn("batch['label']", function_source)
        self.assertNotIn("batch['label_stack']", function_source)


if __name__ == '__main__':
    unittest.main()
