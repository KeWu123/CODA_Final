import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'code' / 'train_sliceeq_occ.py'
STRONG = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong.py'
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong_arcp.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_oaac_strong_arcp.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_arcp.py'
ANALYZER = ROOT / 'code' / 'analyze_sliceeq_arcp_gate.py'
PROTOCOL = (ROOT / 'research' / 'experiments' /
            'h7_slice_profile_reacquisition' /
            'h7_18_axial_response_calibrated_profile_protocol.md')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SliceEqARCPContractTest(unittest.TestCase):
    def test_frozen_parent_and_strong_are_unchanged(self):
        self.assertEqual(
            _sha256(PARENT),
            'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5')
        self.assertEqual(
            _sha256(STRONG),
            'b3219557df39aee680b5bc055b9ff6dc88f5acfa03e8ed1fa5042aacade5bf60')

    def test_arcp_wraps_strong_without_copying_training(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn('import train_sliceeq_occ_oaac_strong as strong', source)
        self.assertIn('result = strong.parent.self_train(', source)
        self.assertNotIn('.backward(', source)
        self.assertNotIn('optimizer.step', source)
        self.assertNotIn('update_model_ema', source)
        self.assertNotIn('test_single_volume', source)
        self.assertIn('EXPECTED_STRONG_TRAIN_SHA256', source)

    def test_recipe_and_evaluation_identity_are_locked(self):
        source = TRAIN.read_text(encoding='utf-8')
        for token in (
                "'exp': 'SliceEqOccOAACStrongARCP_PROMISE12'",
                "'max_iterations': 30000", "'batch_size': 24",
                "'seed': 1337", "'labeled_bs': 12", "'labelnum': 7",
                "'base_lr': 0.01", "'ema_decay': 0.99",
                "'consistency': 0.1", "'consistency_rampup': 200.0",
                "'sliceeq_sigma_min': 0.45",
                "'sliceeq_sigma_max': 0.85",
                'strong.parent.self_train(',
                'strong._context.assert_complete('):
            self.assertIn(token, source)
        self.assertNotIn('ema_model.eval()', source)

    def test_only_profile_weights_are_intercepted(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn('calibrate_profile_weights(', source)
        self.assertIn('replaced[4] = calibrated', source)
        self.assertIn('strong._oaac_paired_reacquisition(', source)
        self.assertIn('strong._oaac_reacquisition_diagnostics(', source)
        self.assertIn(
            'strong.parent.paired_slice_reacquisition =', source)
        self.assertIn('strong.parent.reacquisition_diagnostics =', source)

    def test_calibrator_has_image_only_signature(self):
        tree = ast.parse(UTILITY.read_text(encoding='utf-8'))
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'calibrate_profile_weights')
        names = [argument.arg for argument in function.args.args]
        self.assertEqual(names[:3], [
            'image_stack', 'weights', 'reference_matrix'])
        forbidden = {'label', 'target', 'pseudo', 'logits', 'loss', 'iteration'}
        self.assertTrue(forbidden.isdisjoint(names))

    def test_reference_reads_training_images_not_labels(self):
        source = UTILITY.read_text(encoding='utf-8')
        self.assertIn("stream['image'][:]", source)
        self.assertNotIn("stream['label'][:]", source)
        self.assertIn("'train_slices.list'", source)
        self.assertIn('case_matrices.append(case_matrix)', source)
        self.assertIn('np.mean(', source)

    def test_gate_scope_and_locked_thresholds(self):
        source = ANALYZER.read_text(encoding='utf-8')
        self.assertIn('LABELED_SLICES = 191', source)
        self.assertIn('SIGMA_GRID = (0.45, 0.65, 0.85)', source)
        self.assertIn('PHASE_GRID = (-0.25, 0.0, 0.25)', source)
        self.assertIn("'validation_or_test_read': False", source)
        self.assertIn("summary['cv_reduction'], 0.20", source)
        self.assertIn('fractional_retention, 0.90', source)
        self.assertNotIn('torch.load', source)

    def test_strict_test_entry_and_protocol_exist(self):
        source = TEST.read_text(encoding='utf-8')
        self.assertIn("default='SliceEqOccOAACStrongARCP_PROMISE12'", source)
        self.assertIn("default='False'", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)
        self.assertTrue(PROTOCOL.is_file())


if __name__ == '__main__':
    unittest.main()

