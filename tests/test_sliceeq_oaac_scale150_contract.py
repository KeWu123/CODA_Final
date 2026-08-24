import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'code' / 'train_sliceeq_occ_h7_15_base.py'
STRONG = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong.py'
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_oaac_scale150.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_oaac_scale150.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_oaac_scale150.py'


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SliceEqOAACScale150ContractTest(unittest.TestCase):
    def test_parent_and_scale125_sources_are_unchanged(self):
        self.assertEqual(
            _sha256(PARENT),
            '001d130576df8f669e7ea4c1fec01a362f10194c91b1c615e07b9bd4762fcc4d')
        self.assertEqual(
            _sha256(STRONG),
            'b3219557df39aee680b5bc055b9ff6dc88f5acfa03e8ed1fa5042aacade5bf60')

    def test_only_locked_scale150_ranges_are_used(self):
        source = UTILITY.read_text(encoding='utf-8')
        self.assertIn('OAAC_SCALE = 1.50', source)
        self.assertIn('LOG_GAMMA_RANGE = (-0.30, 0.30)', source)
        self.assertIn('LOG_CONTRAST_RANGE = (-0.225, 0.225)', source)
        self.assertIn('BRIGHTNESS_SPAN_RANGE = (-0.15, 0.15)', source)
        self.assertNotIn('probability', source.lower())

    def test_training_recipe_and_identity_are_locked(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn(
            'import train_sliceeq_occ_h7_15_base as parent', source)
        self.assertIn(
            'from utils.sliceeq_oaac_scale150 import '
            'ordered_appearance_transform', source)
        self.assertIn(
            "parent.parser.set_defaults(exp='SliceEqOccOAACScale150_PROMISE12')",
            source)
        for token in (
                "'base_lr': 0.01", "'ema_decay': 0.99",
                "'consistency': 0.1", "'consistency_rampup': 200.0",
                "'batch_size': 24", "'labeled_bs': 12",
                "'max_iterations': 30000", "'seed': 1337",
                "'sliceeq_sigma_min': 0.45",
                "'sliceeq_sigma_max': 0.85"):
            self.assertIn(token, source)
        self.assertIn('probability=1.0', source)
        self.assertNotIn('ema_model.eval()', source)

    def test_parent_retains_1000_archive_and_original_best_rule(self):
        source = PARENT.read_text(encoding='utf-8')
        self.assertIn('PERIODIC_CHECKPOINT_INTERVAL = 1000', source)
        self.assertIn('if performance > best_performance:', source)
        self.assertIn('iter_num > 0 and iter_num % 200 == 0', source)
        self.assertIn('if iter_num % PERIODIC_CHECKPOINT_INTERVAL == 0:', source)

    def test_strict_test_entry(self):
        source = TEST.read_text(encoding='utf-8')
        self.assertIn("default='SliceEqOccOAACScale150_PROMISE12'", source)
        self.assertIn("default='False'", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)


if __name__ == '__main__':
    unittest.main()
