"""Render publication-ready examples of paired slice re-acquisition.

The script uses only the labeled prefix of train_slices.list. It visualizes
the exact operation used by SliceEqOcc: the same profile weights combine three
neighboring MR slices and their masks, producing an image and a fractional
occupancy target that remain paired.
"""

import argparse
import json
import os
import re

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402


SLICE_PATTERN = re.compile(r'^(?P<case>.+)_slice_?(?P<index>\d+)$')


def parse_slice_name(name):
    match = SLICE_PATTERN.match(name)
    if match is None:
        raise ValueError('unsupported slice name: {}'.format(name))
    return match.group('case'), int(match.group('index'))


def profile_weights(sigma, phase):
    if sigma <= 0.0:
        raise ValueError('sigma must be positive')
    if not -0.5 <= phase <= 0.5:
        raise ValueError('phase must lie in [-0.5, 0.5]')
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    logits = -0.5 * ((offsets - phase) / sigma) ** 2
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


def _read_h5(root_path, name):
    path = os.path.join(root_path, 'data', 'slices', name + '.h5')
    if not os.path.isfile(path):
        raise FileNotFoundError('missing H5 slice: {}'.format(path))
    with h5py.File(path, 'r') as stream:
        image = np.asarray(stream['image'][:], dtype=np.float32)
        label = np.asarray(stream['label'][:] > 0, dtype=np.float32)
    if image.ndim != 2 or image.shape != label.shape:
        raise ValueError('invalid H5 image/label arrays: {}'.format(path))
    return image, label


def _labeled_names(root_path):
    list_path = os.path.join(root_path, 'train_slices.list')
    with open(list_path, 'r', encoding='utf-8-sig') as stream:
        all_names = [line.strip() for line in stream if line.strip()]
    if len(all_names) < 191:
        raise ValueError('train_slices.list has fewer than 191 entries')
    names = all_names[:191]
    cases = [parse_slice_name(name)[0] for name in names]
    if len(list(dict.fromkeys(cases))) != 7:
        raise ValueError('the first 191 slices do not contain seven patients')
    return names


def _neighbor_names(names, center_name):
    grouped = {}
    for name in names:
        case_name, index = parse_slice_name(name)
        grouped.setdefault(case_name, []).append((index, name))
    case_name, _ = parse_slice_name(center_name)
    if case_name not in grouped:
        raise ValueError('center slice is outside the labeled prefix')
    ordered = [name for _, name in sorted(grouped[case_name])]
    position = ordered.index(center_name)
    return [
        ordered[max(0, position - 1)],
        ordered[position],
        ordered[min(len(ordered) - 1, position + 1)],
    ]


def select_demonstration_slice(root_path, names, requested=None):
    """Select the labeled slice with the largest neighbor-label disagreement."""
    if requested:
        if requested not in names:
            raise ValueError(
                '--center_slice must be in the first 191 training slices')
        return requested
    label_cache = {}
    best_name = None
    best_score = -1
    for name in names:
        neighbors = _neighbor_names(names, name)
        labels = []
        for neighbor in neighbors:
            if neighbor not in label_cache:
                label_cache[neighbor] = _read_h5(root_path, neighbor)[1]
            labels.append(label_cache[neighbor])
        previous, center, following = labels
        score = int(np.logical_or(
            previous != center, following != center).sum())
        # Prefer a true interior slice when scores tie.
        interior = int(neighbors[0] != name and neighbors[2] != name)
        candidate = (score, interior)
        if best_name is None or candidate > best_score:
            best_name = name
            best_score = candidate
    return best_name


def _shared_image_limits(images):
    values = np.concatenate([image.reshape(-1) for image in images])
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low + high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _style_axes(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _add_operators(figure, axes):
    for left, symbol in ((0, '+'), (1, '+'), (2, '=')):
        left_box = axes[left].get_position()
        right_box = axes[left + 1].get_position()
        x = 0.5 * (left_box.x1 + right_box.x0)
        y = 0.5 * (left_box.y0 + left_box.y1)
        figure.text(
            x, y, symbol, ha='center', va='center', fontsize=20,
            color='#334155', fontweight='bold')


def render_figures(root_path, output_dir, center_slice=None,
                   sigma=0.65, phase=0.0, dpi=220):
    names = _labeled_names(root_path)
    center_slice = select_demonstration_slice(
        root_path, names, requested=center_slice)
    neighbors = _neighbor_names(names, center_slice)
    loaded = [_read_h5(root_path, name) for name in neighbors]
    images = np.stack([item[0] for item in loaded], axis=0)
    labels = np.stack([item[1] for item in loaded], axis=0)
    weights = profile_weights(sigma, phase)
    reacquired_image = np.sum(images * weights[:, None, None], axis=0)
    occupancy = np.sum(labels * weights[:, None, None], axis=0)
    hard_occupancy = occupancy >= 0.5
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titleweight': 'semibold',
        'axes.titlecolor': '#0F2F59',
        'figure.facecolor': 'white',
    })
    image_limits = _shared_image_limits(list(images) + [reacquired_image])
    normalized_images = np.clip(
        (images - image_limits[0]) /
        (image_limits[1] - image_limits[0]), 0.0, 1.0)
    normalized_reacquired = np.sum(
        normalized_images * weights[:, None, None], axis=0)
    figure, axes = plt.subplots(1, 4, figsize=(13.6, 3.25))
    image_titles = [
        r'$w_{-1}x_{z-1}$' + '\n' + '{:.3f} x previous'.format(weights[0]),
        r'$w_{0}x_z$' + '\n' + '{:.3f} x center'.format(weights[1]),
        r'$w_{+1}x_{z+1}$' + '\n' + '{:.3f} x next'.format(weights[2]),
        r'Re-acquired image $\tilde{x}_z$' + '\n' + 'weighted pixel-wise sum',
    ]
    for index, axis in enumerate(axes[:3]):
        axis.imshow(
            weights[index] * normalized_images[index], cmap='gray',
            vmin=0.0, vmax=1.0)
        axis.set_title(image_titles[index], pad=8)
        _style_axes(axis)
    axes[3].imshow(
        normalized_reacquired, cmap='gray', vmin=0.0, vmax=1.0)
    axes[3].set_title(image_titles[3], pad=8)
    _style_axes(axes[3])
    _add_operators(figure, axes)
    figure.suptitle(
        'Slice-profile re-acquisition: three adjacent MR slices form one '
        'virtual slice', fontsize=14, fontweight='bold', color='#0F2F59',
        y=0.99)
    figure.subplots_adjust(left=0.025, right=0.985, bottom=0.04, top=0.78,
                           wspace=0.52)
    image_path = os.path.join(output_dir, 'sliceeq_image_reacquisition.png')
    figure.savefig(image_path, dpi=dpi, bbox_inches='tight')
    figure.savefig(
        os.path.join(output_dir, 'sliceeq_image_reacquisition.pdf'),
        bbox_inches='tight')
    plt.close(figure)

    occupancy_cmap = LinearSegmentedColormap.from_list(
        'occupancy', ['#07111F', '#56B4E9', '#009E73'])
    figure, axes = plt.subplots(1, 4, figsize=(13.6, 3.25))
    target_titles = [
        r'$w_{-1}y_{z-1}$' + '\n' + '{:.3f} x previous mask'.format(weights[0]),
        r'$w_{0}y_z$' + '\n' + '{:.3f} x center mask'.format(weights[1]),
        r'$w_{+1}y_{z+1}$' + '\n' + '{:.3f} x next mask'.format(weights[2]),
        r'Fractional occupancy $q_z$' + '\n' + 'continuous target in [0, 1]',
    ]
    for index, axis in enumerate(axes[:3]):
        axis.imshow(
            weights[index] * labels[index], cmap=occupancy_cmap,
            vmin=0.0, vmax=1.0)
        axis.set_title(target_titles[index], pad=8)
        _style_axes(axis)
    occupancy_image = axes[3].imshow(
        occupancy, cmap=occupancy_cmap, vmin=0.0, vmax=1.0)
    axes[3].contour(
        hard_occupancy.astype(np.float32), levels=[0.5],
        colors=['#E69F00'], linewidths=0.8)
    axes[3].set_title(target_titles[3], pad=8)
    _style_axes(axes[3])
    _add_operators(figure, axes)
    colorbar = figure.colorbar(
        occupancy_image, ax=axes, orientation='horizontal',
        fraction=0.045, pad=0.08, aspect=45)
    colorbar.set_label('Foreground occupancy')
    figure.suptitle(
        'Paired target transformation: the same profile yields the exact '
        'fractional occupancy', fontsize=14, fontweight='bold',
        color='#0F2F59', y=0.99)
    figure.subplots_adjust(left=0.025, right=0.985, bottom=0.20, top=0.78,
                           wspace=0.52)
    target_path = os.path.join(output_dir, 'sliceeq_paired_occupancy.png')
    figure.savefig(target_path, dpi=dpi, bbox_inches='tight')
    figure.savefig(
        os.path.join(output_dir, 'sliceeq_paired_occupancy.pdf'),
        bbox_inches='tight')
    plt.close(figure)

    metadata = {
        'center_slice': center_slice,
        'neighbor_slices': neighbors,
        'sigma': float(sigma),
        'phase': float(phase),
        'weights': [float(value) for value in weights],
        'fractional_pixel_fraction': float(np.mean(
            np.logical_and(occupancy > 0.0, occupancy < 1.0))),
        'occupancy_change_from_center_mean': float(np.mean(
            np.abs(occupancy - labels[1]))),
        'data_scope': 'first 191 labeled training slices only',
    }
    metadata_path = os.path.join(output_dir, 'sliceeq_demo_manifest.json')
    with open(metadata_path, 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return image_path, target_path, metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--output_dir', default='../paper_figures/demo')
    parser.add_argument('--center_slice', default=None)
    parser.add_argument('--sigma', type=float, default=0.65)
    parser.add_argument('--phase', type=float, default=0.0)
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args()
    outputs = render_figures(
        os.path.abspath(args.root_path), os.path.abspath(args.output_dir),
        center_slice=args.center_slice, sigma=args.sigma,
        phase=args.phase, dpi=args.dpi)
    print('Generated SliceEq demonstration figures:')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
