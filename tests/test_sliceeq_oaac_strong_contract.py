import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PARENT = ROOT / 'code' / 'train_sliceeq_occ.py'
ORIGINAL_OAAC = ROOT / 'code' / 'train_sliceeq_occ_oaac.py'
BASE = ROOT / 'code' / 'train_sliceeq_occ_h7_15_base.py'
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_oaac_strong.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_oaac_strong.py'


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validation_block(source):
    start = source.index('                model.eval()\n')
    end = source.index('\n\n            if iter_num % ', start)
    return source[start:end]


class SliceEqOAACStrongContractTest(unittest.TestCase):
    def test_original_parent_and_oaac_are_unchanged(self):
        self.assertEqual(
            _sha256(ORIGINAL_PARENT),
            'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5')
        self.assertEqual(
            _sha256(ORIGINAL_OAAC),
            'a4a12e54808eafb1a63761a7ee323e467f5ba18b98f868d27e7a3d00f1209e43')

    def test_only_new_base_periodic_archive_changes_to_1000(self):
        original = ORIGINAL_PARENT.read_text(encoding='utf-8')
        successor = BASE.read_text(encoding='utf-8')
        self.assertEqual(
            _validation_block(original), _validation_block(successor))
        self.assertIn('PERIODIC_CHECKPOINT_INTERVAL = 1000', successor)
        self.assertIn(
            'if iter_num % PERIODIC_CHECKPOINT_INTERVAL == 0:', successor)
        self.assertNotIn('if iter_num % 3000 == 0:', successor)
        self.assertIn('if iter_num % 3000 == 0:', original)
        self.assertEqual(original.count('optimizer.step()'), 1)
        self.assertEqual(successor.count('optimizer.step()'), 1)
        self.assertEqual(original.count('base.update_model_ema('), 1)
        self.assertEqual(successor.count('base.update_model_ema('), 1)

    def test_strong_all_ranges_are_exact_joint_125_scale(self):
        source = UTILITY.read_text(encoding='utf-8')
        self.assertIn('OAAC_SCALE = 1.25', source)
        self.assertIn('LOG_GAMMA_RANGE = (-0.25, 0.25)', source)
        self.assertIn(
            'LOG_CONTRAST_RANGE = (-0.1875, 0.1875)', source)
        self.assertIn(
            'BRIGHTNESS_SPAN_RANGE = (-0.125, 0.125)', source)
        self.assertNotIn('probability', source.lower())

    def test_strong_entry_is_isolated_and_recipe_locked(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn(
            'import train_sliceeq_occ_h7_15_base as parent', source)
        self.assertIn(
            "from utils.sliceeq_oaac_strong import "
            'ordered_appearance_transform', source)
        self.assertIn(
            "parent.parser.set_defaults(exp='SliceEqOccOAACStrong_PROMISE12')",
            source)
        for token in (
                "'base_lr': 0.01", "'ema_decay': 0.99",
                "'consistency': 0.1", "'consistency_rampup': 200.0",
                "'batch_size': 24", "'labeled_bs': 12",
                "'max_iterations': 30000", "'seed': 1337"):
            self.assertIn(token, source)
        self.assertIn('probability=1.0', source)
        self.assertNotIn('ema_model.eval()', source)

    def test_test_entry_is_strict_and_separate(self):
        source = TEST.read_text(encoding='utf-8')
        self.assertIn("default='SliceEqOccOAACStrong_PROMISE12'", source)
        self.assertIn("default='False'", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)
        self.assertNotIn('strict=False', source)


if __name__ == '__main__':
    unittest.main()
