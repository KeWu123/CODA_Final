import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
ORIGINAL_STRONG = CODE / 'train_sliceeq_occ_oaac_strong.py'
PORTABLE_STRONG = CODE / 'train_sliceeq_occ_oaac_strong_portable.py'
PORTABLE_MPD = CODE / 'train_sliceeq_occ_oaac_strong_mpd_portable.py'


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SliceEqPortableEntryTest(unittest.TestCase):
    def test_frozen_strong_source_remains_unchanged(self):
        self.assertEqual(
            _sha256(ORIGINAL_STRONG),
            'b3219557df39aee680b5bc055b9ff6dc88f5acfa03e8ed1fa5042aacade5bf60')

    def test_portable_strong_changes_storage_contract_not_training(self):
        source = PORTABLE_STRONG.read_text(encoding='utf-8')
        self.assertIn(
            'import train_sliceeq_occ_h7_15_base as parent', source)
        self.assertIn('result = parent.self_train(', source)
        self.assertIn(
            'validate_promise12_root(\n        args.root_path, '
            'strict_split=True, check_hdf5=True)', source)
        self.assertNotIn("'root_path': (", source)
        self.assertNotIn('.backward(', source)
        self.assertNotIn('optimizer.step', source)

    def test_portable_mpd_reuses_portable_strong_and_frozen_sampler(self):
        source = PORTABLE_MPD.read_text(encoding='utf-8')
        self.assertIn(
            'import train_sliceeq_occ_oaac_strong_portable as strong', source)
        self.assertIn('strong.parent.sample_slice_profiles = sampler', source)
        self.assertIn('result = strong.parent.self_train(', source)
        self.assertIn('EXPECTED_STRONG_TRAIN_SHA256', source)
        self.assertNotIn("'root_path': (", source)
        self.assertNotIn('.backward(', source)
        self.assertNotIn('optimizer.step', source)


if __name__ == '__main__':
    unittest.main()
