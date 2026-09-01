"""Contracts for the 11-label paired-L/U 24-view experiment."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'code' / 'train_slicepair_paired_lu_24_label11.py'
PARENT = ROOT / 'code' / 'train_sliceeq_occ_ablation.py'
PIPELINE = ROOT / 'run_slicepair_paired_lu_24_label11_pipeline.sh'


class SlicePairPairedLU24Label11ContractTest(unittest.TestCase):
    def test_entry_is_syntax_valid_and_locks_the_label_budget(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIsNotNone(ast.parse(source))
        self.assertIn('LABELNUM = 11', source)
        self.assertIn('LABELED_SLICES = 306', source)
        self.assertIn("_inject_default('--labelnum', LABELNUM)", source)
        self.assertIn("'labelnum': LABELNUM", source)
        self.assertIn('patients_to_slices(', source)

    def test_method_is_reused_without_a_second_algorithm(self):
        source = TRAIN.read_text(encoding='utf-8')
        parent_source = PARENT.read_text(encoding='utf-8')
        self.assertIn('import train_sliceeq_occ_ablation as parent', source)
        self.assertIn("_inject_default('--occ_ablation', 'paired_lu_24')",
                      source)
        self.assertIn("_inject_default('--appearance_mode', 'oaac_strong')",
                      source)
        self.assertIn('parent.self_train(', source)
        self.assertIn('6 native-L + 6 paired-L + 12 paired-U', source)
        self.assertIn("'replace_labeled_half': True", parent_source)

    def test_matching_label11_pretrain_is_mandatory(self):
        source = TRAIN.read_text(encoding='utf-8')
        self.assertIn("'--pretrained_checkpoint' not in _ORIGINAL_ARGV",
                      source)
        self.assertIn("'net' not in checkpoint", source)
        self.assertIn("'opt' not in checkpoint", source)
        self.assertIn("'label11', '11_labeled', '11label'", source)

    def test_recipe_changes_only_the_labeled_budget(self):
        source = TRAIN.read_text(encoding='utf-8')
        for token in (
                "'max_iterations': 30000", "'batch_size': 24",
                "'labeled_bs': 12", "'base_lr': 0.01", "'seed': 1337",
                "'ema_decay': 0.99", "'occ_ablation': 'paired_lu_24'",
                "'appearance_mode': 'oaac_strong'"):
            self.assertIn(token, source)

    def test_pipeline_trains_then_tests_validation_best(self):
        source = PIPELINE.read_text(encoding='utf-8')
        self.assertIn('train_slicepair_paired_lu_24_label11.py', source)
        self.assertIn('--labelnum 11', source)
        self.assertIn('--max_iterations 30000', source)
        self.assertIn('--batch_size 24 --labeled_bs 12', source)
        self.assertIn('unet_best_model.pth', source)
        self.assertIn('--checkpoint_path "${CHECKPOINT}"', source)
        self.assertIn('--auto_find_checkpoint False', source)
        self.assertIn('performance.txt', source)


if __name__ == '__main__':
    unittest.main()
