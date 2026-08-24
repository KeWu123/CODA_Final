import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'code' / 'train_sliceeq_occ_oaac.py'
TEST = ROOT / 'code' / 'test_sliceeq_occ_oaac.py'
UTILITY = ROOT / 'code' / 'utils' / 'sliceeq_oaac.py'
PROTOCOL = (ROOT / 'research' / 'experiments' /
            'h7_slice_profile_reacquisition' / 'h7_13_oaac_protocol.md')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SliceEqOAACContractTest(unittest.TestCase):
    def test_frozen_parent_sources_are_unchanged(self):
        expected = {
            ROOT / 'code' / 'train_sliceeq_occ.py': (
                'f9391d1979eba4b87ec5fc6368bbc376201bcf66e83feb1106ef36efb0ec93e5'),
            ROOT / 'code' / 'test_sliceeq_occ.py': (
                '856b235b008fb161b8f1ee4bb54c7ceb00d8a666113fa529828d0f45cf0c5386'),
            ROOT / 'code' / 'utils' / 'sliceeq.py': (
                '44a956a92eccdbb2109034a05ac5ec72f190b33f740ad95c1ed8b505bae168f7'),
            ROOT / 'code' / 'utils' / 'sliceeq_occ.py': (
                'de2fc77c40de4d543e6aa682589aaae652887de9429742d08feacc97e50d6080'),
        }
        for path, digest in expected.items():
            self.assertEqual(_sha256(path), digest)

    def test_pretraining_protocol_is_frozen(self):
        self.assertEqual(
            _sha256(PROTOCOL),
            '146fecfa6e02b2f095f319fb03d8c827b25e26e0af6a7431277d7c35480827f8')

    def test_recipe_and_pretrain_are_runtime_locked(self):
        source = TRAIN.read_text(encoding='utf-8')
        for token in (
                "'exp': 'SliceEqOccOAAC_PROMISE12'",
                "'max_iterations': 30000", "'batch_size': 24",
                "'seed': 1337", "'labeled_bs': 12", "'labelnum': 7",
                "'ema_decay': 0.99", "'consistency': 0.1",
                "'sliceeq_sigma_min': 0.45",
                "'sliceeq_sigma_max': 0.85",
                'EXPECTED_PRETRAINED_SHA256',
                'EXPECTED_PARENT_TRAIN_SHA256',
                "APPEARANCE_SEED = 1339"):
            self.assertIn(token, source)

    def test_only_every_second_reacquired_image_is_transformed(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn('_context.paired_calls % 2 == 1', source)
        self.assertIn(
            'strong_image, metadata = ordered_appearance_transform', source)
        self.assertIn('return clean_image, hard_target, occupancy', source)
        self.assertIn('return strong_image, hard_target, occupancy', source)
        self.assertIn('_context.assert_complete(', source)
        self.assertIn(
            'expected_paired_calls=2 * (args.max_iterations - 1000)',
            source)
        self.assertIn(
            'OAAC expected the labeled re-acquisition at step start', source)
        self.assertIn(
            'OAAC expected the unlabeled re-acquisition after labeled', source)

    def test_parent_training_and_validation_are_reused(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn('training_result = parent.self_train(', source)
        self.assertIn(
            'args, pretrained_checkpoint, snapshot_path)', source)
        self.assertNotIn('test_single_volume', source)
        self.assertNotIn('best_performance', source)
        self.assertNotIn('.backward(', source)
        self.assertNotIn('optimizer.step', source)
        self.assertNotIn('update_model_ema', source)

    def test_cuda_smoke_and_text_activity_log_are_mandatory(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn('def _run_cuda_smoke(device):', source)
        self.assertIn("_run_cuda_smoke(torch.device('cuda'))", source)
        self.assertIn('OAAC CUDA smoke passed:', source)
        self.assertIn('OAAC appearance iteration %d:', source)
        self.assertIn('appearance_below_source_min_fraction',
                      UTILITY.read_text(encoding='utf-8'))
        self.assertIn('appearance_above_source_max_fraction',
                      UTILITY.read_text(encoding='utf-8'))

    def test_transform_is_coordinate_preserving_and_information_preserving(self):
        source = UTILITY.read_text(encoding='utf-8')
        self.assertIn('torch.exp(log_gamma)', source)
        self.assertIn('torch.exp(log_contrast)', source)
        self.assertNotIn('randn', source)
        self.assertNotIn('interpolate', source)
        self.assertNotIn('grid_sample', source)
        self.assertNotIn('softmax', source)
        self.assertNotIn('.clamp(', source)
        self.assertNotIn('hard_target', source)
        self.assertNotIn('occupancy', source)
        self.assertNotIn('label', source.lower())

    def test_strict_test_identity_and_no_checkpoint_search(self):
        source = TEST.read_text(encoding='utf-8')
        self.assertIn("default='SliceEqOccOAAC_PROMISE12'", source)
        self.assertIn("default='False'", source)
        self.assertIn("choices=['False', 'false', '0', 'no']", source)
        self.assertIn('net.load_state_dict(state_dict, strict=True)', source)


if __name__ == '__main__':
    unittest.main()
