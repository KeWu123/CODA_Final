"""Collect validation-selected SliceEqOcc ablation results into CSV and MD."""

import argparse
import csv
from pathlib import Path
import re


ROWS = (
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

VAL_PATTERN = re.compile(
    r'iteration\s+(\d+)\s*:\s*mean_dice\s*:\s*([0-9.]+)')
METRIC_PATTERN = re.compile(
    r'^\s*(Dice|Jaccard|HD95|ASD):\s*([0-9.]+)\s*$', re.MULTILINE)


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


def collect(model_root):
    records = []
    for identifier, method, experiment in ROWS:
        snapshot = model_root / '{}_7_labeled'.format(experiment) / \
            'self_train' / 'unet'
        iteration, val_dice = _best_validation(snapshot / 'log.txt')
        metrics = _test_metrics(snapshot / 'performance.txt')
        records.append({
            'ID': identifier,
            'Method': method,
            'Experiment': experiment,
            'Val-selected Iter': iteration,
            'Val Dice': val_dice,
            'Test Dice': metrics['Dice'],
            'Jaccard': metrics['Jaccard'],
            'HD95': metrics['HD95'],
            'ASD': metrics['ASD'],
            'Status': 'complete' if iteration and metrics['Dice'] else 'missing',
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
        '--output_prefix', type=Path,
        default=Path('../ablation_results/sliceeq_occ_paper_ablation'))
    flags = parser.parse_args()
    records = collect(flags.model_root.resolve())
    prefix = flags.output_prefix.resolve()
    write_csv(records, prefix.with_suffix('.csv'))
    write_markdown(records, prefix.with_suffix('.md'))
    print('Wrote {}'.format(prefix.with_suffix('.csv')))
    print('Wrote {}'.format(prefix.with_suffix('.md')))


if __name__ == '__main__':
    main()
