"""Command-line entry point for the train-only SliceEq MPD audit."""

import argparse
import logging
import os

from utils.sliceeq_mpd_audit import (
    DEFAULT_AXIAL_BINS, DEFAULT_GRID_SIDES, DEFAULT_MOMENT_TOLERANCES,
    DEFAULT_RESIDUAL_TOLERANCES, DEFAULT_UTILITY_FRACTIONS,
    run_offline_audit)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Audit MPD discretization, constraints and patient stability '
            'using only the seven labeled training patients.'))
    parser.add_argument('--root_path', required=True)
    parser.add_argument(
        '--output_dir', default='../mpd_offline_audit')
    parser.add_argument('--reference_artifact', default=None)
    parser.add_argument(
        '--grid_sides', nargs='+', type=int,
        default=list(DEFAULT_GRID_SIDES))
    parser.add_argument(
        '--axial_bins', nargs='+', type=int,
        default=list(DEFAULT_AXIAL_BINS))
    parser.add_argument(
        '--moment_tolerances', nargs='+', type=float,
        default=list(DEFAULT_MOMENT_TOLERANCES))
    parser.add_argument(
        '--residual_tolerances', nargs='+', type=float,
        default=list(DEFAULT_RESIDUAL_TOLERANCES))
    parser.add_argument(
        '--utility_fractions', nargs='+', type=float,
        default=list(DEFAULT_UTILITY_FRACTIONS))
    parser.add_argument('--output_size', nargs=2, type=int, default=[256, 256])
    parser.add_argument('--skip_lopo', action='store_true')
    return parser


def main(args):
    report = run_offline_audit(
        root_path=args.root_path,
        output_dir=args.output_dir,
        reference_artifact=args.reference_artifact,
        grid_sides=args.grid_sides,
        axial_bins=args.axial_bins,
        moment_tolerances=args.moment_tolerances,
        residual_tolerances=args.residual_tolerances,
        utility_fractions=args.utility_fractions,
        output_size=tuple(args.output_size),
        run_lopo=not args.skip_lopo)
    baseline = report['baseline_design']['summary']
    print('MPD offline audit completed')
    print('Output: {}'.format(os.path.abspath(args.output_dir)))
    print('Locked-design worst RFI: {:.6f}'.format(
        baseline['worst_utility']))
    print('Locked-design worst relative gain: {:.2%}'.format(
        baseline['worst_relative_gain']))
    print('Open audit_summary.md for the compact report.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main(build_parser().parse_args())
    except Exception:
        logging.exception('SliceEq MPD offline audit failed')
        raise
