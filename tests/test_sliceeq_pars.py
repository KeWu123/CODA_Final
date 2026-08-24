import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402

from utils.sliceeq_pars import (  # noqa: E402
    PatientAxialAcquisitionBatchSampler, build_sampling_manifest,
    design_axial_distribution)


def _synthetic_names():
    names = []
    # Seven complete labeled patients with exactly 191 slices.
    for patient, count in enumerate([27, 27, 27, 27, 27, 28, 28]):
        names.extend('CaseL{:02d}_slice_{}'.format(patient, index)
                     for index in range(count))
    # Three complete unlabeled patients.
    for patient, count in enumerate([12, 15, 18]):
        names.extend('CaseU{:02d}_slice_{}'.format(patient, index)
                     for index in range(count))
    return names


class SliceEqPARSTest(unittest.TestCase):
    def test_manifest_is_patient_complete_and_index_exact(self):
        manifest = build_sampling_manifest(_synthetic_names())
        self.assertEqual(len(manifest['labeled_cases']), 7)
        self.assertEqual(len(manifest['unlabeled_cases']), 3)
        self.assertEqual(manifest['labeled_indices'], list(range(191)))
        self.assertEqual(
            manifest['unlabeled_indices'],
            list(range(191, len(_synthetic_names()))))
        self.assertTrue(np.all(manifest['labeled_third_counts'] > 0))
        self.assertAlmostEqual(
            manifest['parent_axial_probabilities'].sum(), 1.0, places=12)

    def test_two_stage_design_improves_weak_exposure_and_stays_diverse(self):
        opportunity = np.asarray([
            [0.25, 0.55, 0.35],
            [0.22, 0.48, 0.30],
            [0.28, 0.50, 0.31],
            [0.24, 0.52, 0.33],
            [0.20, 0.46, 0.29],
            [0.26, 0.49, 0.34],
            [0.23, 0.51, 0.32],
        ], dtype=np.float64)
        parent = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        result = design_axial_distribution(opportunity, parent)
        q = result['probabilities']
        self.assertTrue(all(result['checks'].values()))
        self.assertTrue(np.all(q >= 0.0))
        self.assertAlmostEqual(q.sum(), 1.0, places=9)
        self.assertGreaterEqual(
            result['worst_designed_exposure'] + 1e-9,
            result['worst_parent_exposure'])
        self.assertLessEqual(result['max_density_ratio'], 1.5 + 1e-8)

    def test_sampler_is_private_reproducible_and_patient_balanced(self):
        manifest = build_sampling_manifest(_synthetic_names())
        q = np.asarray([0.40, 0.35, 0.25], dtype=np.float64)
        global_before = np.random.get_state()
        first = PatientAxialAcquisitionBatchSampler(
            manifest['labeled_indices'], manifest['unlabeled_indices'],
            24, 12, manifest, q, seed=1341)
        repeated = PatientAxialAcquisitionBatchSampler(
            manifest['labeled_indices'], manifest['unlabeled_indices'],
            24, 12, manifest, q, seed=1341)
        other = PatientAxialAcquisitionBatchSampler(
            manifest['labeled_indices'], manifest['unlabeled_indices'],
            24, 12, manifest, q, seed=1342)
        first_batches = list(iter(first))
        self.assertEqual(first_batches, list(iter(repeated)))
        self.assertNotEqual(first_batches, list(iter(other)))
        self.assertEqual(len(first_batches), 15)
        self.assertTrue(all(len(batch) == 24 for batch in first_batches))
        self.assertEqual(global_before[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(global_before[1], np.random.get_state()[1]))
        self.assertEqual(global_before[2:], np.random.get_state()[2:])

        labeled_case_by_index = {}
        for case_name in manifest['labeled_cases']:
            for third in range(3):
                for index in manifest['case_groups'][case_name][third]:
                    labeled_case_by_index[index] = case_name
        counts = {case_name: 0 for case_name in manifest['labeled_cases']}
        for batch in first_batches:
            for index in batch[:12]:
                counts[labeled_case_by_index[index]] += 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_parent_sampler_is_preserved_for_exact_warmup_batch_count(self):
        manifest = build_sampling_manifest(_synthetic_names())

        class DummyParentSampler:
            def __init__(self, primary, secondary, batch_size,
                         secondary_batch_size):
                self.primary = list(primary)
                self.secondary = list(secondary)
                self.primary_batch = batch_size - secondary_batch_size
                self.secondary_batch = secondary_batch_size

            def __len__(self):
                return len(self.primary) // self.primary_batch

            def __iter__(self):
                batch = tuple(
                    self.primary[:self.primary_batch] +
                    self.secondary[:self.secondary_batch])
                for _ in range(len(self)):
                    yield batch

        parent_batch = tuple(
            manifest['labeled_indices'][:12] +
            manifest['unlabeled_indices'][:12])
        sampler = PatientAxialAcquisitionBatchSampler(
            manifest['labeled_indices'], manifest['unlabeled_indices'],
            24, 12, manifest, np.asarray([0.4, 0.35, 0.25]),
            seed=1341, parent_sampler_class=DummyParentSampler,
            warmup_batches=16)
        first_epoch = list(iter(sampler))
        second_epoch = list(iter(sampler))
        self.assertTrue(all(batch == parent_batch for batch in first_epoch))
        self.assertEqual(second_epoch[0], parent_batch)
        self.assertTrue(any(batch != parent_batch for batch in second_epoch[1:]))
        self.assertEqual(sampler._batches_seen, 30)


if __name__ == '__main__':
    unittest.main()
