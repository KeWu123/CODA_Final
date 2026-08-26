"""Collect validation-selected SliceEqOcc ablation results into CSV and MD."""

import argparse
import csv
from pathlib import Path
import re


FULL_ROWS = (
    ('B0', 'MT-24', 'SliceEqOccIncremental_baseline_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('C0', 'ViewMatch-36', 'SliceEqOccIncremental_baseline_36_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('C1', 'SRA-Image-36', 'SliceEqOccIncremental_image_only_36_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('C2', 'SRA-Hard-36', 'SliceEqOccIncremental_hard_targets_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('F10', 'AFO-L-only', 'SliceEqOccIncremental_occ_l_only_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('F01', 'AFO-U-only', 'SliceEqOccIncremental_occ_u_only_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('C3', 'SliceEqOcc', 'SliceEqOccIncremental_full_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('C4', 'SliceEqOcc+OAAC-S1.25', 'SliceEqOccOAACStrong_PROMISE12'),
    ('C5', 'SliceEqOcc+OAAC-S1.25+MPD',
     'SliceEqOccOAACStrongMPD_PROMISE12'),
)

PAPER_MPD_ROWS = (
    ('A0', 'Appearance-matched teacher-student (24 views)',
     'SlicePairViewBudget_A0_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('A1', 'Image profile + center hard target (24 views)',
     'SlicePairViewBudget_A1_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('A2', 'Paired-U, uniform law (24 views)',
     'SlicePairViewBudget_A2_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('A3', 'Paired L/U, labeled replacement (24 views)',
     'SliceEqOccIncremental_paired_lu_24_35_5_10_Pre10000_Self30000_label7_seed1337'),
    ('A4', 'Paired L/U, additional labeled view (36 views)',
     'SliceEqOccOAACStrong_PROMISE12'),
    ('A5', 'SlicePair, MPD law (36 views)',
     'SliceEqOccOAACStrongMPD_PROMISE12'),
)

PRESETS = {
    'full': FULL_ROWS,
    'paper_mpd': PAPER_MPD_ROWS,
}

# Backward-compatible alias for imports in older analysis notebooks.
ROWS = FULL_ROWS

VAL_PATTERN = re.compile(
    r'iteration\s+(\d+)\s*:\s*mean_dice\s*:\s*([0-9.]+)')
METRIC_PATTERN = re.compile(
    r'^\s*(Dice|Jaccard|HD95|ASD):\s*([0-9.]+)\s*$', re.MULTILINE)
MODEL_PATH_PATTERN = re.compile(r'^Model path:\s*(.+?)\s*$', re.MULTILINE)


def _best_validation(log_path):
    if not log_path.is_file():
        return '', ''
    matches = [
        (int(iteration), float(dice))
        for iteration, dice in VAL_PATTERN.findall(
            log_path.read_text(encoding='utf-8', errors='replace'))
    ]
    if not matches:
        return '', ''
    iteration, dice = max(matches, key=lambda item: item[1])
    return str(iteration), '{:.6f}'.format(dice)


def _test_metrics(performance_path):
    empty = {name: '' for name in ('Dice', 'Jaccard', 'HD95', 'ASD')}
    if not performance_path.is_file():
        return empty
    text = performance_path.read_text(encoding='utf-8', errors='replace')
    average = text.split('Average metric:', 1)[-1]
    values = dict(METRIC_PATTERN.findall(average))
    empty.update(values)
    return empty


def _test_checkpoint(performance_path):
    if not performance_path.is_file():
        return ''
    text = performance_path.read_text(encoding='utf-8', errors='replace')
    match = MODEL_PATH_PATTERN.search(text)
    if match is None:
        return ''
    return re.split(r'[/\\]', match.group(1).strip())[-1]


def collect(model_root, rows=ROWS):
    records = []
    for identifier, method, experiment in rows:
        snapshot = model_root / '{}_7_labeled'.format(experiment) / \
            'self_train' / 'unet'
        iteration, val_dice = _best_validation(snapshot / 'log.txt')
        performance_path = snapshot / 'performance.txt'
        metrics = _test_metrics(performance_path)
        test_checkpoint = _test_checkpoint(performance_path)
        if not iteration or not metrics['Dice']:
            status = 'missing'
        elif not test_checkpoint:
            status = 'selector-unknown'
        elif test_checkpoint != 'unet_best_model.pth':
            status = 'selector-mismatch'
        else:
            status = 'complete'
        records.append({
            'ID': identifier,
            'Method': method,
            'Experiment': experiment,
            'Val-selected Iter': iteration,
            'Val Dice': val_dice,
            'Test Checkpoint': test_checkpoint,
            'Test Dice': metrics['Dice'],
            'Jaccard': metrics['Jaccard'],
            'HD95': metrics['HD95'],
            'ASD': metrics['ASD'],
            'Status': status,
        })
    return records


def write_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def write_markdown(records, path):
    headers = list(records[0].keys())
    lines = [
        '# SliceEqOcc ablation results', '',
        '| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join('---' for _ in headers) + '|',
    ]
    for record in records:
        lines.append('| ' + ' | '.join(record[name] for name in headers) + ' |')
    lines.extend([
        '',
        '> Checkpoints are selected by the 5-case validation split. HD95 and '
        'ASD are legacy voxel-index distances.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=Path, default=Path('../model'))
    parser.add_argument(
        '--preset', choices=sorted(PRESETS), default='full',
        help='row set to export')
    parser.add_argument(
        '--output_prefix', type=Path,
        default=Path('../ablation_results/sliceeq_occ_paper_ablation'))
    flags = parser.parse_args()
    records = collect(flags.model_root.resolve(), PRESETS[flags.preset])
    prefix = flags.output_prefix.resolve()
    write_csv(records, prefix.with_suffix('.csv'))
    write_markdown(records, prefix.with_suffix('.md'))
    print('Wrote {}'.format(prefix.with_suffix('.csv')))
    print('Wrote {}'.format(prefix.with_suffix('.md')))


if __name__ == '__main__':
    main()
