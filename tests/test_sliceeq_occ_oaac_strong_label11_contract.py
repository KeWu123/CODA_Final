"""Static contracts for the isolated PROMISE12 11-label workflow."""

import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PRETRAIN = ROOT / 'code' / 'pretrain_promise12_label11.py'
SELF_TRAIN = ROOT / 'code' / 'train_sliceeq_occ_oaac_strong_label11.py'
README = ROOT / 'docs' / 'SLICEEQ_OCC_OAAC_STRONG_LABEL11_README.md'


def _attributes(tree):
    return [node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)]


class Label11ContractTests(unittest.TestCase):
    def test_budget_is_first_11_cases_and_306_slices(self):
        for path in (PRETRAIN, SELF_TRAIN):
            source = path.read_text(encoding='utf-8')
            self.assertIn('LABELNUM = 11', source)
            self.assertIn('LABELED_SLICES = 306', source)
            self.assertIn("train.list", source)
            self.assertIn("train_slices.list", source)

    def test_pretrain_entry_cannot_start_self_training(self):
        tree = ast.parse(PRETRAIN.read_text(encoding='utf-8'))
        attributes = _attributes(tree)
        self.assertIn('pre_train', attributes)
        self.assertNotIn('self_train', attributes)

    def test_self_train_requires_explicit_matching_pretrain(self):
        source = SELF_TRAIN.read_text(encoding='utf-8')
        tree = ast.parse(source)
        self.assertIn('self_train', _attributes(tree))
        self.assertIn("'--pretrained_checkpoint' not in _ORIGINAL_ARGV", source)
        self.assertIn("'net' not in checkpoint", source)
        self.assertIn("'opt' not in checkpoint", source)

    def test_final_recipe_remains_frozen(self):
        source = SELF_TRAIN.read_text(encoding='utf-8')
        required = (
            "'max_iterations': 30000",
            "'batch_size': 24",
            "'base_lr': 0.01",
            "'seed': 1337",
            "'labeled_bs': 12",
            "'ema_decay': 0.99",
            "'sliceeq_sigma_min': 0.45",
            "'sliceeq_sigma_max': 0.85",
            "'sliceeq_phase_min': -0.25",
            "'sliceeq_phase_max': 0.25",
        )
        for token in required:
            self.assertIn(token, source)

    def test_documented_commands_use_label11_entries(self):
        source = README.read_text(encoding='utf-8')
        self.assertIn('pretrain_promise12_label11.py', source)
        self.assertIn('train_sliceeq_occ_oaac_strong_label11.py', source)
        self.assertIn('--labelnum 11', source)
        self.assertIn('SliceEqOccOAACStrong_PROMISE12_11_labeled', source)


if __name__ == '__main__':
    unittest.main()
