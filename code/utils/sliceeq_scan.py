"""Scan-coherent continuous profile schedules for SliceEqOccSC.

The schedule treats one patient as one virtual acquisition domain within a
refresh window.  Randomized stratification preserves continuous marginal
coverage without introducing the fixed-node tail truncation of SliceEqSAQ.
"""

import torch


def _validate_range(value_range, name, positive_minimum=False):
    minimum, maximum = value_range
    if maximum < minimum:
        raise ValueError('{} maximum must be >= minimum'.format(name))
    if positive_minimum and minimum <= 0.0:
        raise ValueError('{} minimum must be positive'.format(name))


def _refresh_seed(base_seed, refresh_id):
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError('base_seed must be a nonnegative integer')
    if not isinstance(refresh_id, int) or refresh_id < 0:
        raise ValueError('refresh_id must be a nonnegative integer')
    # Explicit arithmetic is stable across Python processes and platforms and
    # is unaffected by Python's randomized string hashing.
    return base_seed + refresh_id * 1000003


def build_scan_protocol_table(case_names, sigma_range, phase_range,
                              refresh_id, base_seed):
    """Assign one continuous, stratified virtual protocol to every case.

    Each marginal receives exactly one jittered draw from each of ``N``
    strata. Independent permutations decorrelate sigma and phase.  Sorting the
    unique case names makes the table invariant to DataLoader/list traversal.
    """
    _validate_range(sigma_range, 'sigma', positive_minimum=True)
    _validate_range(phase_range, 'phase')
    cases = sorted(set(str(name) for name in case_names))
    if not cases:
        raise ValueError('case_names must contain at least one case')
    if len(cases) != len(case_names):
        raise ValueError('case_names passed to the protocol table must be unique')

    generator = torch.Generator(device='cpu')
    generator.manual_seed(_refresh_seed(base_seed, refresh_id))
    count = len(cases)
    sigma_order = torch.randperm(count, generator=generator)
    phase_order = torch.randperm(count, generator=generator)
    sigma_unit = (
        sigma_order.to(torch.float64) +
        torch.rand(count, generator=generator, dtype=torch.float64)
    ) / float(count)
    phase_unit = (
        phase_order.to(torch.float64) +
        torch.rand(count, generator=generator, dtype=torch.float64)
    ) / float(count)

    sigma_min, sigma_max = sigma_range
    phase_min, phase_max = phase_range
    sigma = sigma_unit * (sigma_max - sigma_min) + sigma_min
    phase = phase_unit * (phase_max - phase_min) + phase_min
    return {
        case_name: (float(sigma[index]), float(phase[index]))
        for index, case_name in enumerate(cases)
    }


def scan_coherent_slice_profiles(case_names, protocol_table, offsets, device,
                                 dtype=torch.float32):
    """Return Gaussian profile weights selected by stable patient identity."""
    if not case_names:
        raise ValueError('case_names must contain at least one batch element')
    missing = sorted(set(case_names) - set(protocol_table))
    if missing:
        raise KeyError(
            'scan protocol is missing case {}'.format(missing[0]))
    values = [protocol_table[case_name] for case_name in case_names]
    sigma = torch.tensor(
        [value[0] for value in values], device=device, dtype=dtype)
    phase = torch.tensor(
        [value[1] for value in values], device=device, dtype=dtype)
    offset_tensor = torch.as_tensor(
        offsets, dtype=dtype, device=device).view(1, -1)
    logits = -0.5 * (
        (offset_tensor - phase.view(-1, 1)) /
        sigma.view(-1, 1)).square()
    weights = torch.softmax(logits, dim=1)
    if not torch.isfinite(weights).all():
        raise FloatingPointError('scan-coherent weights contain non-finite values')
    if (weights < 0).any():
        raise ValueError('scan-coherent weights must be nonnegative')
    sums = weights.sum(dim=1)
    if not torch.allclose(
            sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError('scan-coherent weights must sum to one')
    return weights, sigma, phase


def scan_coherence_diagnostics(case_names, sigma, phase):
    """Measure within-batch case coherence without reading image or labels."""
    if len(case_names) != sigma.numel() or sigma.shape != phase.shape:
        raise ValueError('case/profile batch dimensions do not match')
    if not torch.isfinite(sigma).all() or not torch.isfinite(phase).all():
        raise FloatingPointError('profile parameters contain non-finite values')
    unique_cases = sorted(set(case_names))
    sigma_range_max = sigma.new_tensor(0.0)
    phase_range_max = phase.new_tensor(0.0)
    for case_name in unique_cases:
        indices = [
            index for index, name in enumerate(case_names)
            if name == case_name
        ]
        selected_sigma = sigma[indices]
        selected_phase = phase[indices]
        sigma_range_max = torch.maximum(
            sigma_range_max,
            selected_sigma.max() - selected_sigma.min())
        phase_range_max = torch.maximum(
            phase_range_max,
            selected_phase.max() - selected_phase.min())
    return {
        'unique_case_fraction': sigma.new_tensor(
            len(unique_cases) / float(len(case_names))),
        'within_case_sigma_range_max': sigma_range_max,
        'within_case_phase_range_max': phase_range_max,
    }
