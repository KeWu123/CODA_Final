import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'code' / 'train_sliceeq_occ_h7_15_base.py'
STRONG = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong.py'
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong_mpd.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_oaac_strong_mpd.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_mpd.py'
PROTOCOL = (ROOT / 'research' / 'experiments' /
            'h7_slice_profile_reacquisition' /
            'h7_19_robust_moment_profile_gate_protocol.md')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SliceEqMPDContractTest(unittest.TestCase):
    def test_frozen_base_and_strong_are_unchanged(self):
        self.assertEqual(
            _sha256(BASE),
            '001d130576df8f669e7ea4c1fec01a362f10194c91b1c615e07b9bd4762fcc4d')
        self.assertEqual(
            _sha256(STRONG),
            'b3219557df39aee680b5bc055b9ff6dc88f5acfa03e8ed1fa5042aacade5bf60')

    def test_wrapper_reuses_strong_and_parent_training(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn('import train_sliceeq_occ_oaac_strong as strong', source)
        self.assertIn('result = strong.parent.self_train(', source)
        self.assertIn('strong.parent.sample_slice_profiles = sampler', source)
        self.assertNotIn('.backward(', source)
        self.assertNotIn('optimizer.step', source)
        self.assertNotIn('update_model_ema', source)
        self.assertNotIn('ema_model.eval()', source)

    def test_entire_parent_recipe_and_archive_interval_are_locked(self):
        source = TRAIN.read_text(encoding='utf-8')
        for token in (
                "'exp': 'SliceEqOccOAACStrongMPD_PROMISE12'",
                "'max_iterations': 30000", "'batch_size': 24",
                "'seed': 1337", "'labeled_bs': 12", "'labelnum': 7",
                "'base_lr': 0.01", "'ema_decay': 0.99",
                "'consistency': 0.1", "'consistency_rampup': 200.0",
                "'sliceeq_sigma_min': 0.45",
                "'sliceeq_sigma_max': 0.85",
                'EXPECTED_STRONG_TRAIN_SHA256'):
            self.assertIn(token, source)
        self.assertIn('PERIODIC_CHECKPOINT_INTERVAL = 1000',
                      BASE.read_text(encoding='utf-8'))

    def test_direct_design_scope_is_explicit_and_no_gate_is_required(self):
        train_source = TRAIN.read_text(encoding='utf-8')
        utility_source = UTILITY.read_text(encoding='utf-8')
        self.assertIn('user-authorized exploratory direct full run', train_source)
        self.assertIn('build_direct_design_artifact(', train_source)
        self.assertNotIn('load_gate_artifact(', train_source)
        self.assertIn("all_names[:191]", utility_source)
        self.assertIn("stream['image'][:]", utility_source)
        self.assertIn("stream['label'][:]", utility_source)
        self.assertIn("all_names[191:]", utility_source)
        self.assertIn('active_strata = active_count > 0', utility_source)
        self.assertIn("'structurally_empty_rfi_strata'", utility_source)
        self.assertNotIn('torch.load', utility_source)
        self.assertNotIn('val.list', utility_source)
        self.assertNotIn('test.list', utility_source)

    def test_optimizer_is_locked_to_one_global_symmetric_distribution(self):
        source = UTILITY.read_text(encoding='utf-8')
        for token in (
                'GRID_SIDE = 21', 'MOMENT_TOLERANCE = 0.02',
                'IMAGE_RESIDUAL_TOLERANCE = 0.05',
                'DENSITY_RATIO_CAP = 3.0',
                'ENTROPY_FRACTION_MIN = 0.70',
                'UTILITY_OPTIMUM_FRACTION = 0.99',
                "method='SLSQP'", '_mirror_projection(sigmas, phases)',
                "'execution_mode': 'user_override_skip_lopo_direct_full_training'"):
            self.assertIn(token, source)
        tree = ast.parse(source)
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn('collect_exact_design_statistics', functions)
        self.assertIn('build_direct_design_artifact', functions)

    def test_test_entry_is_strict_and_separate(self):
        source = TEST.read_text(encoding='utf-8')
        self.assertIn("default='SliceEqOccOAACStrongMPD_PROMISE12'", source)
        self.assertIn("default='False'", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)
        self.assertTrue(PROTOCOL.is_file())


if __name__ == '__main__':
    unittest.main()
