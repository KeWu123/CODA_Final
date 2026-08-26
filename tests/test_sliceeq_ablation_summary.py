import csv
from pathlib import Path
import sys
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_ROOT))

from summarize_sliceeq_ablation import (  # noqa: E402
    PAPER_MPD_ROWS, collect, write_csv, write_markdown)


class SliceEqAblationSummaryTest(unittest.TestCase):
    def test_collects_validation_best_and_average_test_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / (
                'SliceEqOccIncremental_baseline_36_35_5_10_Pre10000_'
                'Self30000_label7_seed1337_7_labeled/self_train/unet')
            snapshot.mkdir(parents=True)
            (snapshot / 'log.txt').write_text(
                'iteration 200 : mean_dice : 0.700000, best_dice : 0.700000\n'
                'iteration 400 : mean_dice : 0.750000, best_dice : 0.750000\n',
                encoding='utf-8')
            (snapshot / 'performance.txt').write_text(
                'Model path: /tmp/unet_best_model.pth\n'
                'Case00 -> Dice: 0.1, Jaccard: 0.1, HD95: 1, ASD: 1\n'
                'Average metric:\nDice: 0.810000\nJaccard: 0.700000\n'
                'HD95: 3.000000\nASD: 1.500000\n', encoding='utf-8')

            records = collect(root)
            record = next(item for item in records if item['ID'] == 'C0')
            self.assertEqual(record['Val-selected Iter'], '400')
            self.assertEqual(record['Val Dice'], '0.750000')
            self.assertEqual(record['Test Dice'], '0.810000')
            self.assertEqual(record['Test Checkpoint'],
                             'unet_best_model.pth')
            self.assertEqual(record['Status'], 'complete')

            csv_path = root / 'results.csv'
            md_path = root / 'results.md'
            write_csv(records, csv_path)
            write_markdown(records, md_path)
            with csv_path.open(encoding='utf-8', newline='') as source:
                self.assertEqual(len(list(csv.DictReader(source))), len(records))
            self.assertIn('ViewMatch-36', md_path.read_text(encoding='utf-8'))
            self.assertTrue(any(
                item['ID'] == 'C5' and 'MPD' in item['Method']
                for item in records))

    def test_paper_preset_rejects_test_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / (
                'SliceEqOccOAACStrongMPD_PROMISE12_7_labeled/'
                'self_train/unet')
            snapshot.mkdir(parents=True)
            (snapshot / 'log.txt').write_text(
                'iteration 25800 : mean_dice : 0.836008, '
                'best_dice : 0.836008\n', encoding='utf-8')
            (snapshot / 'performance.txt').write_text(
                'Model path: /tmp/iter_29000.pth\nAverage metric:\n'
                'Dice: 0.854573\nJaccard: 0.749330\n'
                'HD95: 3.256519\nASD: 1.324697\n', encoding='utf-8')

            records = collect(root, PAPER_MPD_ROWS)
            self.assertEqual([row['ID'] for row in records],
                             ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
            a5 = records[-1]
            self.assertEqual(a5['Test Checkpoint'], 'iter_29000.pth')
            self.assertEqual(a5['Status'], 'selector-mismatch')


if __name__ == '__main__':
    unittest.main()
