import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'code' / 'train_sliceeq_occ_h7_15_base.py'
STRONG = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong.py'
MPD = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong_mpd.py'
MPD_UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_mpd.py'
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong_mpd_pars.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_oaac_strong_mpd_pars.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_pars.py'
PROTOCOL = (ROOT / 'research' / 'experiments' /
            'h7_slice_profile_reacquisition' /
            'h7_20_patient_axial_acquisition_risk_protocol.md')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SliceEqPARSContractTest(unittest.TestCase):
    def test_all_frozen_parents_are_unchanged(self):
        self.assertEqual(
            _sha256(BASE),
            '001d130576df8f669e7ea4c1fec01a362f10194c91b1c615e07b9bd4762fcc4d')
        self.assertEqual(
            _sha256(STRONG),
            'b3219557df39aee680b5bc055b9ff6dc88f5acfa03e8ed1fa5042aacade5bf60')
        self.assertEqual(
            _sha256(MPD),
            '9255640e81259309350f10f80e8a1319ae2a3deaf5a1164292c5baa9292c8f5f')
        self.assertEqual(
            _sha256(MPD_UTILITY),
            '9b215cefe8f22172c2811af63365c2cb21e40daba4f01f4ceae0d5cabf44c817')

    def test_wrapper_changes_only_profile_and_batch_sampler_hooks(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn(
            'import train_sliceeq_occ_oaac_strong_mpd as mpd', source)
        self.assertIn(
            'mpd.strong.parent.sample_slice_profiles = profile_sampler', source)
        self.assertIn(
            'mpd.strong.parent.TwoStreamBatchSampler = sampler_factory', source)
        self.assertIn('mpd.strong.parent.self_train(', source)
        for forbidden in (
                '.backward(', 'optimizer.step', 'update_model_ema',
                'ema_model.eval()', 'base_lr =', 'consistency_weight ='):
            self.assertNotIn(forbidden, source)

    def test_runtime_recipe_and_archive_interval_are_locked(self):
        source = TRAIN.read_text(encoding='utf-8')
        for token in (
                "'exp': 'SliceEqOccOAACStrongMPDPARS_PROMISE12'",
                "'max_iterations': 30000", "'batch_size': 24",
                "'seed': 1337", "'labeled_bs': 12", "'labelnum': 7",
                "'base_lr': 0.01", "'ema_decay': 0.99",
                "'consistency': 0.1", "'consistency_rampup': 200.0",
                "'sliceeq_sigma_min': 0.45",
                "'sliceeq_sigma_max': 0.85",
                'EXPECTED_MPD_TRAIN_SHA256', 'EXPECTED_MPD_UTILITY_SHA256'):
            self.assertIn(token, source)
        self.assertIn(
            'PERIODIC_CHECKPOINT_INTERVAL = 1000',
            BASE.read_text(encoding='utf-8'))

    def test_design_is_fixed_exact_l_and_not_online_hard_mining(self):
        source = UTILITY.read_text(encoding='utf-8')
        for token in (
                'LABELED_SLICE_COUNT = 191', 'AXIAL_THIRDS = 3',
                'SAMPLER_SEED = 1341', 'DENSITY_RATIO_CAP = 1.50',
                'ENTROPY_FRACTION_MIN = 0.90',
                'UTILITY_OPTIMUM_FRACTION = 0.99',
                "method='SLSQP'", 'collect_exact_design_statistics(root_path)',
                "'unlabeled_labels_read': 0",
                "'model_predictions_or_losses_read': False"):
            self.assertIn(token, source)
        for forbidden in (
                'val.list', 'test.list', 'torch.load', 'confidence',
                'pseudo_label', 'uncertainty_score'):
            self.assertNotIn(forbidden, source)
        tree = ast.parse(source)
        classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn('PatientAxialAcquisitionBatchSampler', classes)
        self.assertIn('FrozenPARSBatchSamplerFactory', classes)

    def test_sampler_preserves_stream_size_and_uses_private_numpy_generator(self):
        source = UTILITY.read_text(encoding='utf-8')
        self.assertIn('np.random.default_rng(int(seed))', source)
        self.assertIn('self._warmup_batches - self._batches_seen', source)
        self.assertIn('parent_sampler = self._parent_sampler_class(', source)
        self.assertIn(
            'return len(self.primary_indices) // self.primary_batch_size', source)
        self.assertIn('self.primary_batch_size', source)
        self.assertIn('self.secondary_batch_size', source)
        self.assertNotIn('np.random.seed(', source)

    def test_test_entry_is_strict_and_separate(self):
        source = TEST.read_text(encoding='utf-8')
        self.assertIn(
            "default='SliceEqOccOAACStrongMPDPARS_PROMISE12'", source)
        self.assertIn("default='False'", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)
        self.assertTrue(PROTOCOL.is_file())


if __name__ == '__main__':
    unittest.main()
