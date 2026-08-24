"""H7.20 patient-axial acquisition-risk sampling utilities.

PARS changes only which training slice indices enter the fixed SliceEqOcc
batch.  It is deliberately not online hard mining: the sole axial probability
law is designed once from exact labeled-training occupancy under the frozen
MPD profile distribution.  Runtime sampling never reads labels, predictions,
losses, uncertainty, validation, test data, or iteration state.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize

from utils.sliceeq_mpd import (
    DesignError, atomic_json_dump, collect_exact_design_statistics,
    distribution_entropy, sha256_file)


SCHEMA_VERSION = 'h7.20-pars-v1'
LABELED_SLICE_COUNT = 191
AXIAL_THIRDS = 3
SAMPLER_SEED = 1341
DENSITY_RATIO_CAP = 1.50
ENTROPY_FRACTION_MIN = 0.90
UTILITY_OPTIMUM_FRACTION = 0.99
NUMERICAL_EPSILON = 1e-12
SLICE_MARKER = '_slice_'


def _parse_slice_name(sample_name):
    """Dependency-light equivalent of the frozen SliceEq name parser."""
    if not isinstance(sample_name, str) or SLICE_MARKER not in sample_name:
        raise ValueError('expected <case>_slice_<integer>: {}'.format(
            sample_name))
    case_name, separator, index_text = sample_name.rpartition(SLICE_MARKER)
    if not separator or not case_name or not index_text:
        raise ValueError('malformed slice name: {}'.format(sample_name))
    try:
        slice_index = int(index_text)
    except ValueError as error:
        raise ValueError('slice suffix is not an integer: {}'.format(
            sample_name)) from error
    return case_name, slice_index


def _read_train_slice_names(root_path):
    path = os.path.join(root_path, 'train_slices.list')
    if not os.path.isfile(path):
        raise FileNotFoundError('missing training slice list: {}'.format(path))
    with open(path, 'r', encoding='utf-8-sig') as stream:
        names = [line.strip() for line in stream if line.strip()]
    if len(names) <= LABELED_SLICE_COUNT:
        raise DesignError('training slice list has no unlabeled suffix')
    return names, path


def build_sampling_manifest(sample_names, labeled_slice_count=LABELED_SLICE_COUNT):
    """Build index-only patient/third groups without opening any H5 file."""
    sample_names = list(sample_names)
    if labeled_slice_count != LABELED_SLICE_COUNT:
        raise ValueError('H7.20 locks the seven-patient 191-slice prefix')
    if len(sample_names) <= labeled_slice_count:
        raise ValueError('sample list has no unlabeled suffix')

    parsed = [_parse_slice_name(name) for name in sample_names]
    case_to_entries = {}
    case_order = []
    for global_index, (case_name, slice_index) in enumerate(parsed):
        if case_name not in case_to_entries:
            case_order.append(case_name)
            case_to_entries[case_name] = []
        case_to_entries[case_name].append((slice_index, global_index))

    case_groups = {}
    for case_name in case_order:
        entries = sorted(case_to_entries[case_name])
        slice_indices = [entry[0] for entry in entries]
        expected = list(range(slice_indices[0], slice_indices[-1] + 1))
        if slice_indices != expected:
            raise DesignError(
                'non-contiguous slices for patient {}'.format(case_name))
        thirds = {third: [] for third in range(AXIAL_THIRDS)}
        for rank, (_, global_index) in enumerate(entries):
            third = min(AXIAL_THIRDS - 1,
                        (AXIAL_THIRDS * rank) // len(entries))
            thirds[int(third)].append(int(global_index))
        if any(not thirds[third] for third in range(AXIAL_THIRDS)):
            raise DesignError(
                'patient has an empty index third: {}'.format(case_name))
        case_groups[case_name] = thirds

    labeled_cases = []
    unlabeled_cases = []
    for case_name in case_order:
        indices = [index for third in range(AXIAL_THIRDS)
                   for index in case_groups[case_name][third]]
        in_labeled = [index < labeled_slice_count for index in indices]
        if any(in_labeled) and not all(in_labeled):
            raise DesignError(
                'labeled/unlabeled boundary splits patient {}'.format(
                    case_name))
        (labeled_cases if all(in_labeled) else unlabeled_cases).append(
            case_name)

    if len(labeled_cases) != 7:
        raise DesignError(
            'H7.20 requires seven complete labeled patients; found {}'.format(
                len(labeled_cases)))
    if not unlabeled_cases:
        raise DesignError('H7.20 requires at least one unlabeled patient')

    labeled_indices = sorted(index for case_name in labeled_cases
                             for third in range(AXIAL_THIRDS)
                             for index in case_groups[case_name][third])
    unlabeled_indices = sorted(index for case_name in unlabeled_cases
                               for third in range(AXIAL_THIRDS)
                               for index in case_groups[case_name][third])
    if labeled_indices != list(range(labeled_slice_count)):
        raise DesignError('labeled index prefix is not exact')
    if unlabeled_indices != list(range(labeled_slice_count, len(sample_names))):
        raise DesignError('unlabeled index suffix is not exact')

    labeled_third_counts = np.asarray([
        sum(len(case_groups[case_name][third])
            for case_name in labeled_cases)
        for third in range(AXIAL_THIRDS)], dtype=np.float64)
    parent_axial_probabilities = labeled_third_counts / labeled_third_counts.sum()
    return {
        'sample_names': sample_names,
        'case_order': case_order,
        'case_groups': case_groups,
        'labeled_cases': labeled_cases,
        'unlabeled_cases': unlabeled_cases,
        'labeled_indices': labeled_indices,
        'unlabeled_indices': unlabeled_indices,
        'labeled_third_counts': labeled_third_counts.astype(np.int64),
        'parent_axial_probabilities': parent_axial_probabilities,
    }


def _sampling_distribution_hash(probabilities, train_slices_sha256,
                                mpd_distribution_sha256):
    payload = {
        'schema_version': SCHEMA_VERSION,
        'probabilities': [float(value) for value in probabilities],
        'train_slices_sha256': str(train_slices_sha256),
        'mpd_distribution_sha256': str(mpd_distribution_sha256),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def design_axial_distribution(opportunity, parent_probabilities):
    """Two-stage robust design of one global three-bin axial law.

    ``opportunity[p,t]`` is the expected exact fractional-information yield of
    sampling one slice from patient p and index-third t under frozen MPD.
    Patients are normalized by their parent expected opportunity so the
    max-min objective cannot be dominated by a large/easy anatomy.
    """
    opportunity = np.asarray(opportunity, dtype=np.float64)
    parent = np.asarray(parent_probabilities, dtype=np.float64)
    if opportunity.ndim != 2 or opportunity.shape[1] != AXIAL_THIRDS:
        raise ValueError('opportunity must have shape [patients,3]')
    if parent.shape != (AXIAL_THIRDS,):
        raise ValueError('parent probabilities must have shape [3]')
    if np.any(opportunity < 0.0) or not np.isfinite(opportunity).all():
        raise ValueError('opportunity must be finite and nonnegative')
    if np.any(parent <= 0.0) or not np.isfinite(parent).all() or not np.isclose(
            parent.sum(), 1.0, atol=1e-12):
        raise ValueError('parent probabilities must be a positive simplex')

    patient_parent_utility = opportunity @ parent
    if np.any(patient_parent_utility <= NUMERICAL_EPSILON):
        raise DesignError('a labeled patient has no acquisition opportunity')
    normalized = opportunity / patient_parent_utility[:, None]
    active = normalized > NUMERICAL_EPSILON
    if not np.all(active.any(axis=0)):
        raise DesignError('an axial third has no exact acquisition opportunity')

    rows = []
    row_keys = []
    for patient in range(opportunity.shape[0]):
        for third in range(AXIAL_THIRDS):
            if active[patient, third]:
                row = np.zeros(AXIAL_THIRDS, dtype=np.float64)
                row[third] = normalized[patient, third]
                rows.append(row)
                row_keys.append((patient, third))
    exposure = np.stack(rows, axis=0)
    entropy_floor = ENTROPY_FRACTION_MIN * distribution_entropy(parent)
    caps = DENSITY_RATIO_CAP * parent

    constraints_stage1 = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1.0},
        {'type': 'ineq', 'fun': lambda x: exposure @ x[:-1] - x[-1]},
        {'type': 'ineq', 'fun': lambda x: (
            distribution_entropy(x[:-1]) - entropy_floor)},
    ]
    initial_t = float(np.min(exposure @ parent))
    initial = np.concatenate((parent, [initial_t]))
    result1 = minimize(
        lambda x: -x[-1], initial, method='SLSQP',
        bounds=[(0.0, float(cap)) for cap in caps] + [(0.0, None)],
        constraints=constraints_stage1,
        options={'ftol': 1e-12, 'maxiter': 2000, 'disp': False})
    if not result1.success:
        raise DesignError(
            'PARS stage-one robust design failed: {}'.format(result1.message))
    t_star = float(result1.x[-1])
    utility_floor = UTILITY_OPTIMUM_FRACTION * t_star

    def _kl(q):
        positive = q > 0.0
        return float(np.sum(q[positive] * np.log(q[positive] / parent[positive])))

    constraints_stage2 = [
        {'type': 'eq', 'fun': lambda q: np.sum(q) - 1.0},
        {'type': 'ineq', 'fun': lambda q: exposure @ q - utility_floor},
        {'type': 'ineq', 'fun': lambda q: (
            distribution_entropy(q) - entropy_floor)},
    ]
    result2 = minimize(
        _kl, result1.x[:-1], method='SLSQP',
        bounds=[(0.0, float(cap)) for cap in caps],
        constraints=constraints_stage2,
        options={'ftol': 1e-13, 'maxiter': 2000, 'disp': False})
    if not result2.success:
        raise DesignError(
            'PARS stage-two KL projection failed: {}'.format(result2.message))
    q = np.maximum(result2.x, 0.0)
    q /= q.sum()
    parent_exposure = exposure @ parent
    designed_exposure = exposure @ q
    checks = {
        'simplex': bool(np.all(q >= 0.0) and abs(float(q.sum()) - 1.0) <= 1e-9),
        'density_cap': bool(np.all(q <= caps + 1e-9)),
        'entropy_floor': bool(
            distribution_entropy(q) >= entropy_floor - 1e-9),
        'utility_floor': bool(np.all(designed_exposure >= utility_floor - 1e-8)),
        'all_axial_thirds_active': bool(np.all(active.any(axis=0))),
    }
    if not all(checks.values()):
        raise DesignError(
            'PARS distribution violates locked constraints: {}'.format(checks))
    return {
        'probabilities': q,
        't_star': t_star,
        'utility_floor': utility_floor,
        'kl_to_parent': _kl(q),
        'stage1_iterations': int(result1.nit),
        'stage2_iterations': int(result2.nit),
        'checks': checks,
        'normalized_opportunity': normalized,
        'active_strata': active,
        'exposure_row_keys': row_keys,
        'parent_exposure': parent_exposure,
        'designed_exposure': designed_exposure,
        'worst_parent_exposure': float(np.min(parent_exposure)),
        'worst_designed_exposure': float(np.min(designed_exposure)),
        'entropy': distribution_entropy(q),
        'parent_entropy': distribution_entropy(parent),
        'max_density_ratio': float(np.max(q / parent)),
    }


def build_direct_pars_artifact(root_path, mpd_design, output_path,
                               protocol_path=None):
    """Design and atomically freeze PARS before the direct full run."""
    probabilities = np.asarray(mpd_design['probabilities'], dtype=np.float64)
    mpd_distribution_sha = mpd_design['distribution_sha256']
    statistics = collect_exact_design_statistics(root_path)
    names, list_path = _read_train_slice_names(root_path)
    manifest = build_sampling_manifest(names)
    if statistics['case_order'] != manifest['labeled_cases']:
        raise DesignError('MPD statistics and PARS labeled patient order differ')
    if statistics['train_slices_sha256'] != sha256_file(list_path):
        raise DesignError('training slice list changed during PARS design')

    active_mean = statistics['utilities'] @ probabilities
    # MPD utilities average only over slices with nonzero opportunity. Runtime
    # samples every slice in a stratum, so inactive slices contribute zero.
    per_slice_opportunity = active_mean * (
        statistics['active_slice_count'] /
        statistics['slice_count'].astype(np.float64))
    opportunity = per_slice_opportunity.reshape(
        len(statistics['case_order']), AXIAL_THIRDS)
    parent = manifest['parent_axial_probabilities']
    result = design_axial_distribution(opportunity, parent)
    q = result['probabilities']
    distribution_sha = _sampling_distribution_hash(
        q, statistics['train_slices_sha256'], mpd_distribution_sha)

    stratum_names = [
        '{}:index-third-{}'.format(case_name, third)
        for case_name in statistics['case_order']
        for third in range(AXIAL_THIRDS)]
    report = {
        'schema_version': SCHEMA_VERSION,
        'method': 'SliceEqOcc-OAAC-Strong-MPD-PARS',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'execution_mode': 'direct_full_training_single_sampling_intervention',
        'decision': 'exploratory_direct_design_ready',
        'training_authorized': True,
        'evidence_scope': (
            'all-seven labeled-training patients; exact occupancy under frozen '
            'MPD; no U label, model output, loss, validation or test used'),
        'data_firewall': {
            'labeled_slices_read': LABELED_SLICE_COUNT,
            'labeled_patients_read': len(statistics['case_order']),
            'unlabeled_labels_read': 0,
            'validation_or_test_read': False,
            'model_predictions_or_losses_read': False,
            'train_slices_sha256': statistics['train_slices_sha256'],
            'labeled_image_label_content_sha256':
                statistics['labeled_content_sha256'],
            'labeled_patient_ids': statistics['case_order'],
            'unlabeled_patient_ids': manifest['unlabeled_cases'],
        },
        'locked_design': {
            'axial_bins': 'three equal index-rank thirds within each patient',
            'patient_sampling': 'uniform cyclic random permutation per stream',
            'within_patient_third_sampling': 'uniform with replacement',
            'density_ratio_cap': DENSITY_RATIO_CAP,
            'entropy_fraction_min': ENTROPY_FRACTION_MIN,
            'utility_optimum_fraction': UTILITY_OPTIMUM_FRACTION,
            'sampler_seed': SAMPLER_SEED,
            'mpd_distribution_sha256': mpd_distribution_sha,
            'protocol_path': protocol_path,
            'protocol_sha256': sha256_file(protocol_path)
                if protocol_path and os.path.isfile(protocol_path) else None,
        },
        'sampling_design': {
            'probabilities': [float(value) for value in q],
            'parent_probabilities': [float(value) for value in parent],
            'sampling_distribution_sha256': distribution_sha,
            't_star': result['t_star'],
            'utility_floor': result['utility_floor'],
            'kl_to_parent': result['kl_to_parent'],
            'stage1_iterations': result['stage1_iterations'],
            'stage2_iterations': result['stage2_iterations'],
            'entropy': result['entropy'],
            'parent_entropy': result['parent_entropy'],
            'max_density_ratio': result['max_density_ratio'],
            'worst_parent_exposure': result['worst_parent_exposure'],
            'worst_designed_exposure': result['worst_designed_exposure'],
            'checks': result['checks'],
        },
        'patient_strata': {
            'names': stratum_names,
            'slice_count': statistics['slice_count'].tolist(),
            'active_slice_count': statistics['active_slice_count'].tolist(),
            'expected_mpd_rfi_per_sample': per_slice_opportunity.tolist(),
            'normalized_opportunity':
                result['normalized_opportunity'].reshape(-1).tolist(),
            'active_for_robust_exposure':
                result['active_strata'].reshape(-1).tolist(),
            'labeled_third_counts':
                manifest['labeled_third_counts'].tolist(),
        },
        'conditions': dict(result['checks']),
    }
    if not all(report['conditions'].values()):
        raise DesignError('PARS direct design failed a locked condition')
    atomic_json_dump(report, output_path)
    design = validate_pars_artifact(report)
    design['artifact_sha256'] = sha256_file(output_path)
    design['report'] = report
    design['manifest'] = manifest
    return design


def validate_pars_artifact(report):
    if report.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported H7.20 PARS artifact schema')
    if report.get('decision') != 'exploratory_direct_design_ready' or not \
            report.get('training_authorized', False):
        raise ValueError('PARS artifact does not authorize training')
    design = report.get('sampling_design', {})
    q = np.asarray(design.get('probabilities', []), dtype=np.float64)
    parent = np.asarray(
        design.get('parent_probabilities', []), dtype=np.float64)
    if q.shape != (AXIAL_THIRDS,) or parent.shape != (AXIAL_THIRDS,):
        raise ValueError('PARS artifact must contain three probabilities')
    if np.any(q < 0.0) or not np.isfinite(q).all() or not np.isclose(
            q.sum(), 1.0, atol=1e-9):
        raise ValueError('invalid PARS probability simplex')
    if not report.get('conditions') or not all(report['conditions'].values()):
        raise ValueError('PARS artifact has a failed condition')
    expected_hash = _sampling_distribution_hash(
        q, report['data_firewall']['train_slices_sha256'],
        report['locked_design']['mpd_distribution_sha256'])
    if design.get('sampling_distribution_sha256') != expected_hash:
        raise ValueError('PARS sampling distribution hash mismatch')
    return {
        'probabilities': q,
        'parent_probabilities': parent,
        'sampling_distribution_sha256': expected_hash,
    }


class PatientAxialAcquisitionBatchSampler:
    """Patient-balanced, frozen axial-risk two-stream batch sampler."""

    def __init__(self, primary_indices, secondary_indices, batch_size,
                 secondary_batch_size, manifest, axial_probabilities,
                 seed=SAMPLER_SEED, log_callback=None,
                 parent_sampler_class=None, warmup_batches=0):
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.secondary_batch_size = int(secondary_batch_size)
        self.primary_batch_size = int(batch_size) - self.secondary_batch_size
        if self.primary_batch_size <= 0 or self.secondary_batch_size <= 0:
            raise ValueError('both PARS streams must be nonempty')
        if self.primary_indices != manifest['labeled_indices']:
            raise ValueError('PARS primary indices differ from locked prefix')
        if self.secondary_indices != manifest['unlabeled_indices']:
            raise ValueError('PARS secondary indices differ from locked suffix')
        self.manifest = manifest
        self.axial_probabilities = np.asarray(
            axial_probabilities, dtype=np.float64)
        if self.axial_probabilities.shape != (AXIAL_THIRDS,) or not np.isclose(
                self.axial_probabilities.sum(), 1.0, atol=1e-9):
            raise ValueError('PARS runtime probabilities are invalid')
        self._rng = np.random.default_rng(int(seed))
        self._epoch = 0
        self._log_callback = log_callback
        self._parent_sampler_class = parent_sampler_class
        self._warmup_batches = int(warmup_batches)
        self._batches_seen = 0
        if self._warmup_batches < 0:
            raise ValueError('warmup_batches must be nonnegative')
        if self._warmup_batches and self._parent_sampler_class is None:
            raise ValueError(
                'a parent sampler class is required for warmup preservation')

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

    def _balanced_cases(self, cases, count):
        output = []
        while len(output) < count:
            output.extend(self._rng.permutation(cases).tolist())
        return output[:count]

    def _draw_stream(self, cases, count):
        selected = []
        case_counts = {case_name: 0 for case_name in cases}
        third_counts = np.zeros(AXIAL_THIRDS, dtype=np.int64)
        for case_name in self._balanced_cases(cases, count):
            third = int(self._rng.choice(
                AXIAL_THIRDS, p=self.axial_probabilities))
            candidates = self.manifest['case_groups'][case_name][third]
            choice = int(candidates[int(self._rng.integers(len(candidates)))])
            selected.append(choice)
            case_counts[case_name] += 1
            third_counts[third] += 1
        return selected, case_counts, third_counts

    def __iter__(self):
        self._epoch += 1
        batch_count = len(self)
        parent_batch_count = min(
            batch_count, max(0, self._warmup_batches - self._batches_seen))
        if parent_batch_count:
            parent_sampler = self._parent_sampler_class(
                self.primary_indices, self.secondary_indices,
                self.primary_batch_size + self.secondary_batch_size,
                self.secondary_batch_size)
            parent_iterator = iter(parent_sampler)
            for _ in range(parent_batch_count):
                self._batches_seen += 1
                yield tuple(next(parent_iterator))

        pars_batch_count = batch_count - parent_batch_count
        if pars_batch_count <= 0:
            return
        primary, primary_cases, primary_thirds = self._draw_stream(
            self.manifest['labeled_cases'],
            pars_batch_count * self.primary_batch_size)
        secondary, secondary_cases, secondary_thirds = self._draw_stream(
            self.manifest['unlabeled_cases'],
            pars_batch_count * self.secondary_batch_size)
        if self._log_callback is not None and (
                self._epoch == 1 or self._epoch % 25 == 0):
            self._log_callback({
                'epoch': self._epoch,
                'primary_patient_counts': primary_cases,
                'secondary_patient_counts': secondary_cases,
                'primary_third_counts': primary_thirds.tolist(),
                'secondary_third_counts': secondary_thirds.tolist(),
            })
        for batch_index in range(pars_batch_count):
            p0 = batch_index * self.primary_batch_size
            s0 = batch_index * self.secondary_batch_size
            self._batches_seen += 1
            yield tuple(
                primary[p0:p0 + self.primary_batch_size] +
                secondary[s0:s0 + self.secondary_batch_size])


class FrozenPARSBatchSamplerFactory:
    """Parent-compatible factory that permits exactly one sampler creation."""

    def __init__(self, manifest, probabilities, seed=SAMPLER_SEED,
                 log_callback=None, parent_sampler_class=None,
                 warmup_batches=1000):
        self.manifest = manifest
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.seed = int(seed)
        self.log_callback = log_callback
        self.parent_sampler_class = parent_sampler_class
        self.warmup_batches = int(warmup_batches)
        self.calls = 0
        self.instance = None

    def __call__(self, primary_indices, secondary_indices, batch_size,
                 secondary_batch_size):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError('PARS batch sampler factory called more than once')
        self.instance = PatientAxialAcquisitionBatchSampler(
            primary_indices, secondary_indices, batch_size,
            secondary_batch_size, self.manifest, self.probabilities,
            seed=self.seed, log_callback=self.log_callback,
            parent_sampler_class=self.parent_sampler_class,
            warmup_batches=self.warmup_batches)
        return self.instance
