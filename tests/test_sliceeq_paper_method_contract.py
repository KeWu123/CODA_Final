from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / 'docs' / 'SLICEEQ_AC_COMPLETE_PAPER_METHOD_ZH.md'
PIPELINE = ROOT / 'run_coda_final_paper_suite.sh'


class SliceEqPaperMethodContractTest(unittest.TestCase):
    def test_method_has_one_acquisition_aligned_story(self):
        source = METHOD.read_text(encoding='utf-8')
        for token in (
                'SliceEq-AC', 'Slice-profile Re-Acquisition',
                'Aligned Fractional Occupancy',
                'Ordered Acquisition-Appearance Consistency',
                'Moment-constrained Profile Design',
                '不增加网络参数和推理延迟'):
            self.assertIn(token, source)

    def test_every_trainable_stage_has_a_scientific_question(self):
        source = METHOD.read_text(encoding='utf-8')
        for stage in (
                'baseline', 'baseline_36', 'image_only_36', 'hard_targets',
                'occ_l_only', 'occ_u_only', 'full', 'oaac_strong', 'mpd'):
            self.assertIn('`{}`'.format(stage), source)

    def test_pipeline_and_paper_component_order_match(self):
        method = METHOD.read_text(encoding='utf-8')
        pipeline = PIPELINE.read_text(encoding='utf-8')
        for row in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5'):
            self.assertIn(row, method)
            self.assertIn(row, pipeline)
        self.assertIn('SRA+AFO+OAAC', pipeline)
        self.assertIn('SliceEq-AC+MPD', pipeline)

    def test_checkpoint_evidence_is_not_overclaimed(self):
        source = METHOD.read_text(encoding='utf-8')
        self.assertIn('0.851960', source)
        self.assertIn('0.854573', source)
        self.assertIn('不能称为独立无偏 test 主结果', source)


if __name__ == '__main__':
    unittest.main()
