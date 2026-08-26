from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / 'run_coda_final_paper_suite.sh'


class CODAFinalPipelineContractTest(unittest.TestCase):
    def test_complete_stage_order_is_explicit(self):
        source = PIPELINE.read_text(encoding='utf-8')
        self.assertIn(
            'STAGES:-baseline baseline_36 image_only_36 hard_targets '
            'occ_l_only occ_u_only full oaac_strong mpd', source)
        for identifier in ('B0', 'C0', 'C1', 'C2', 'F10', 'F01',
                           'C3', 'C4', 'C5'):
            self.assertIn(identifier, source)

    def test_shared_protocol_and_split_are_locked(self):
        source = PIPELINE.read_text(encoding='utf-8')
        for token in (
                '"train.list:35"', '"val.list:5"', '"test.list:10"',
                '"train_slices.list:940"', '--max_iterations 30000',
                '--batch_size 24', '--labeled_bs 12', '--labelnum 7',
                '--seed 1337',
                '49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9'):
            self.assertIn(token, source)

    def test_portable_training_and_strict_testing_are_used(self):
        source = PIPELINE.read_text(encoding='utf-8')
        self.assertIn('train_sliceeq_occ_oaac_strong_portable.py', source)
        self.assertIn(
            'train_sliceeq_occ_oaac_strong_mpd_portable.py', source)
        self.assertIn('--checkpoint_path "${checkpoint}"', source)
        self.assertIn('--auto_find_checkpoint False', source)
        self.assertIn('unet_best_model.pth', source)

    def test_audit_figures_and_result_table_are_part_of_suite(self):
        source = PIPELINE.read_text(encoding='utf-8')
        self.assertIn('visualize_sliceeq_reacquisition.py', source)
        self.assertIn('analyze_sliceeq_mpd_robustness.py', source)
        self.assertIn('summarize_sliceeq_ablation.py', source)
        self.assertIn('SKIP_COMPLETED', source)

    def test_current_paper_ablation_is_matched_and_selector_safe(self):
        source = PIPELINE.read_text(encoding='utf-8')
        for stage in ('paper_a0', 'paper_a1', 'paper_a2', 'paper_a3',
                      'paper_a4', 'paper_a5'):
            self.assertIn(stage, source)
        self.assertIn('--appearance_mode oaac_strong', source)
        self.assertIn('paper_a0) echo "baseline"', source)
        self.assertIn('paper_a1) echo "image_only"', source)
        self.assertIn('paper_a2) echo "aligned_occ"', source)
        self.assertIn('paper_a3) echo "paired_lu_24"', source)
        self.assertIn(
            'paper_a4) echo "SliceEqOccOAACStrong_PROMISE12"', source)
        self.assertIn(
            'paper_a5) echo "SliceEqOccOAACStrongMPD_PROMISE12"', source)
        self.assertIn('performance_is_validation_best', source)
        self.assertIn('checkpoint_is_lfs_pointer', source)
        self.assertIn('git lfs install && git lfs pull', source)
        self.assertIn('performance_before_validation_retest_${RUN_TAG}.txt',
                      source)
        self.assertIn('unet_best_model\\.pth', source)
        self.assertIn('--preset "${SUMMARY_PRESET}"', source)


if __name__ == '__main__':
    unittest.main()
