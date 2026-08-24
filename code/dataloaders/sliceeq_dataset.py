"""Adjacent-slice dataset for SliceEq without modifying the locked dataset module."""

import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


SLICE_MARKER = '_slice_'


def parse_slice_name(sample_name):
    """Return `(case, integer_index)` from a locked train-slice entry."""
    if not isinstance(sample_name, str) or SLICE_MARKER not in sample_name:
        raise ValueError(
            'SliceEq requires names of the form <case>_slice_<integer>: {}'.format(
                sample_name))
    case_name, separator, index_text = sample_name.rpartition(SLICE_MARKER)
    if not separator or not case_name or not index_text:
        raise ValueError('Malformed SliceEq slice name: {}'.format(sample_name))
    try:
        slice_index = int(index_text)
    except ValueError as error:
        raise ValueError(
            'SliceEq slice suffix must be an integer: {}'.format(
                sample_name)) from error
    return case_name, slice_index


def build_neighbor_table(sample_list, radius=1):
    """Build a strict within-case adjacency table in original list order.

    At a true volume endpoint, an out-of-range offset is clamped to the endpoint.
    Internal gaps are rejected because silently crossing one would fabricate anatomy.
    """
    if radius < 1:
        raise ValueError('SliceEq radius must be at least one')

    lookup = {}
    case_indices = {}
    parsed = []
    for sample_name in sample_list:
        case_name, slice_index = parse_slice_name(sample_name)
        key = (case_name, slice_index)
        if key in lookup:
            raise ValueError('Duplicate SliceEq slice index: {}'.format(key))
        lookup[key] = sample_name
        case_indices.setdefault(case_name, []).append(slice_index)
        parsed.append(key)

    case_bounds = {}
    for case_name, indices in case_indices.items():
        sorted_indices = sorted(indices)
        expected = list(range(sorted_indices[0], sorted_indices[-1] + 1))
        if sorted_indices != expected:
            missing = sorted(set(expected) - set(sorted_indices))
            raise ValueError(
                'Non-contiguous SliceEq indices for {} (first missing: {})'.format(
                    case_name, missing[0]))
        case_bounds[case_name] = (sorted_indices[0], sorted_indices[-1])

    table = []
    offsets = tuple(range(-radius, radius + 1))
    for case_name, slice_index in parsed:
        first_index, last_index = case_bounds[case_name]
        neighbors = []
        was_clamped = False
        for offset in offsets:
            requested = slice_index + offset
            resolved = min(max(requested, first_index), last_index)
            was_clamped = was_clamped or resolved != requested
            neighbors.append(lookup[(case_name, resolved)])
        table.append((tuple(neighbors), was_clamped))
    return table


class StackRandomGenerator(object):
    """Apply one baseline-compatible spatial draw to an entire slice stack."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_stack = sample['image_stack']
        label_stack = sample['label_stack']

        # Match RandomGenerator's branch probabilities and random-call order.
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            image_stack = np.rot90(
                image_stack, k, axes=(1, 2))
            label_stack = np.rot90(
                label_stack, k, axes=(1, 2))
            axis = np.random.randint(0, 2) + 1
            image_stack = np.flip(image_stack, axis=axis).copy()
            label_stack = np.flip(label_stack, axis=axis).copy()
        elif random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            image_stack = ndimage.rotate(
                image_stack, angle, axes=(1, 2), order=0, reshape=False)
            label_stack = ndimage.rotate(
                label_stack, angle, axes=(1, 2), order=0, reshape=False)

        height, width = image_stack.shape[-2:]
        factors = (
            1,
            self.output_size[0] / height,
            self.output_size[1] / width,
        )
        image_stack = zoom(image_stack, factors, order=0)
        label_stack = zoom(label_stack, factors, order=0)

        image_stack = torch.from_numpy(
            image_stack.astype(np.float32)).unsqueeze(1)
        label_stack = torch.from_numpy(label_stack.astype(np.uint8))
        center_index = image_stack.shape[0] // 2
        return {
            'image': image_stack[center_index],
            'label': label_stack[center_index],
            'image_stack': image_stack,
            'label_stack': label_stack,
        }


class SliceStackDataSets(Dataset):
    """PROMISE12 training dataset returning a fixed real neighboring stack."""

    def __init__(self, base_dir, transform, radius=1, num=None):
        self._base_dir = base_dir
        self.transform = transform
        self.radius = radius
        list_path = os.path.join(base_dir, 'train_slices.list')
        with open(list_path, 'r', encoding='utf-8-sig') as stream:
            self.sample_list = [line.strip() for line in stream if line.strip()]
        if num is not None:
            self.sample_list = self.sample_list[:num]
        self.neighbor_table = build_neighbor_table(
            self.sample_list, radius=radius)
        print('total {} stacked samples'.format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def _read_slice(self, sample_name):
        path = os.path.join(
            self._base_dir, 'data', 'slices', sample_name + '.h5')
        with h5py.File(path, 'r') as stream:
            image = stream['image'][:]
            label = stream['label'][:]
        if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
            raise ValueError(
                'SliceEq expects matched 2D image/label arrays: {}'.format(path))
        return image, label

    def __getitem__(self, index):
        case = self.sample_list[index]
        neighbor_names, was_clamped = self.neighbor_table[index]
        images = []
        labels = []
        for neighbor_name in neighbor_names:
            image, label = self._read_slice(neighbor_name)
            images.append(image)
            labels.append(label)

        sample = {
            'image_stack': np.stack(images, axis=0),
            'label_stack': np.stack(labels, axis=0),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        sample['case'] = case
        sample['neighbor_clamped'] = torch.tensor(
            float(was_clamped), dtype=torch.float32)
        return sample

