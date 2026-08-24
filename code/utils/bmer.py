"""Boundary-Manifold Evidence Resynthesis utilities.

BMER v1 is deliberately an input-only augmentation.  It extracts standardized
two-sided intensity profiles from detached unlabeled pseudo-masks, stores them in a
frozen non-parametric bank, and renders sampled profiles on labeled recipient geometry.
It never changes a target, loss, network, EMA update, sampler, or inference path.
"""

import hashlib
import os

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _check_positive_integer(name, value):
    if not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))


def _check_image_mask_batch(images, masks):
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must have shape [B, 1, H, W]")
    if masks.ndim != 3:
        raise ValueError("masks must have shape [B, H, W]")
    if (images.shape[0], images.shape[2], images.shape[3]) != tuple(masks.shape):
        raise ValueError("images and masks must have matching batch/spatial shapes")
    if not images.is_floating_point():
        raise TypeError("images must be floating-point tensors")


def _robust_location_scale(image, eps=1e-6):
    median = float(np.median(image))
    mad = float(np.median(np.abs(image - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.std(image))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return median, scale


def _fill_profile(values, fallback):
    values = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(values)
    if valid.any():
        x = np.arange(values.size, dtype=np.float32)
        values[~valid] = np.interp(x[~valid], x[valid], values[valid])
        return values
    return np.asarray(fallback, dtype=np.float32).copy()


def _geometry(mask, radius, sectors, min_foreground_pixels):
    mask = np.asarray(mask) > 0
    foreground_pixels = int(mask.sum())
    if foreground_pixels < min_foreground_pixels or foreground_pixels == mask.size:
        return None

    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    signed_distance = inside - outside
    band = np.abs(signed_distance) <= float(radius)
    if not band.any():
        return None

    coordinates = np.argwhere(mask)
    center_y, center_x = coordinates.mean(axis=0)
    grid_y, grid_x = np.indices(mask.shape)
    angle = np.mod(np.arctan2(grid_y - center_y, grid_x - center_x),
                   2.0 * np.pi)
    sector_index = np.floor(angle * sectors / (2.0 * np.pi)).astype(np.int64)
    sector_index = np.clip(sector_index, 0, sectors - 1)

    rounded_distance = np.rint(signed_distance).astype(np.int64)
    rounded_distance = np.clip(rounded_distance, -radius, radius)
    distance_index = rounded_distance + radius

    if radius == 1:
        normalized_distance = np.zeros_like(signed_distance, dtype=np.float32)
    else:
        normalized_distance = np.clip(
            (np.abs(signed_distance) - 1.0) / float(radius - 1), 0.0, 1.0)
    taper = 0.5 * (1.0 + np.cos(np.pi * normalized_distance))
    taper = taper.astype(np.float32)
    taper[~band] = 0.0

    return {
        "band": band,
        "distance_index": distance_index,
        "sector_index": sector_index,
        "taper": taper,
    }


def extract_profile_field(image, mask, radius=8, sectors=16,
                          min_foreground_pixels=32):
    """Extract one standardized [sector, signed-distance] intensity field.

    Returns ``None`` for empty, full, or too-small masks. Missing sector/distance cells
    are interpolated along the normal coordinate, with the slice-global profile as the
    deterministic fallback.
    """
    _check_positive_integer("radius", radius)
    _check_positive_integer("sectors", sectors)
    _check_positive_integer("min_foreground_pixels", min_foreground_pixels)

    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask)
    if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
        raise ValueError("image and mask must be matching 2-D arrays")
    if not np.isfinite(image).all():
        raise ValueError("image contains NaN or Inf")

    geometry = _geometry(mask, radius, sectors, min_foreground_pixels)
    if geometry is None:
        return None

    median, scale = _robust_location_scale(image)
    standardized = (image - median) / scale
    profile_length = 2 * radius + 1

    global_profile = np.full(profile_length, np.nan, dtype=np.float32)
    for distance_bin in range(profile_length):
        selected = (geometry["distance_index"] == distance_bin) & geometry["band"]
        if selected.any():
            global_profile[distance_bin] = float(standardized[selected].mean())
    global_profile = _fill_profile(global_profile, np.zeros(profile_length,
                                                             dtype=np.float32))

    field = np.empty((sectors, profile_length), dtype=np.float32)
    for sector in range(sectors):
        values = np.full(profile_length, np.nan, dtype=np.float32)
        sector_mask = geometry["sector_index"] == sector
        for distance_bin in range(profile_length):
            selected = (sector_mask & geometry["band"] &
                        (geometry["distance_index"] == distance_bin))
            if selected.any():
                values[distance_bin] = float(standardized[selected].mean())
        field[sector] = _fill_profile(values, global_profile)

    if not np.isfinite(field).all():
        raise RuntimeError("profile extraction produced NaN or Inf")
    return {
        "field": torch.from_numpy(field),
        "median": median,
        "scale": scale,
        "geometry": geometry,
    }


def build_position_bin_map(sample_names, position_bins=3):
    """Map ordered ``case_slice_index`` names to normalized volume-position bins."""
    _check_positive_integer("position_bins", position_bins)
    grouped = {}
    unparsed = []
    for order, raw_name in enumerate(sample_names):
        name = str(raw_name).strip()
        if "_slice_" not in name:
            unparsed.append(name)
            continue
        case_name, slice_text = name.rsplit("_slice_", 1)
        try:
            slice_index = int(slice_text)
        except ValueError:
            unparsed.append(name)
            continue
        grouped.setdefault(case_name, []).append((slice_index, order, name))

    result = {}
    for entries in grouped.values():
        entries = sorted(entries, key=lambda item: (item[0], item[1]))
        count = len(entries)
        for rank, (_, _, name) in enumerate(entries):
            normalized = 0.5 if count == 1 else rank / float(count - 1)
            position_bin = min(int(normalized * position_bins), position_bins - 1)
            result[name] = position_bin

    center_bin = position_bins // 2
    for name in unparsed:
        result[name] = center_bin
    return result


def lookup_position_bins(case_names, position_bin_map):
    bins = []
    for case_name in case_names:
        key = str(case_name).strip()
        if key not in position_bin_map:
            raise KeyError("missing BMER position bin for {}".format(key))
        bins.append(int(position_bin_map[key]))
    return bins


class BoundaryProfileBank(object):
    """Frozen non-parametric bank of standardized boundary profile fields."""

    VERSION = 1

    def __init__(self, radius=8, sectors=16, position_bins=3,
                 min_foreground_pixels=32):
        _check_positive_integer("radius", radius)
        _check_positive_integer("sectors", sectors)
        _check_positive_integer("position_bins", position_bins)
        _check_positive_integer("min_foreground_pixels", min_foreground_pixels)
        self.radius = radius
        self.sectors = sectors
        self.position_bins = position_bins
        self.min_foreground_pixels = min_foreground_pixels
        self._profiles = {index: [] for index in range(position_bins)}
        self._source_ids = {index: [] for index in range(position_bins)}
        self._frozen = False
        self.skipped = 0

    def __len__(self):
        return sum(len(values) if isinstance(values, list) else values.shape[0]
                   for values in self._profiles.values())

    @property
    def frozen(self):
        return self._frozen

    def add(self, images, masks, position_bins, source_ids=None):
        if self._frozen:
            raise RuntimeError("cannot add to a frozen BMER bank")
        _check_image_mask_batch(images, masks)
        if len(position_bins) != images.shape[0]:
            raise ValueError("position_bins length must match batch size")
        if source_ids is None:
            source_ids = ["unknown"] * images.shape[0]
        if len(source_ids) != images.shape[0]:
            raise ValueError("source_ids length must match batch size")

        images_np = images.detach().cpu().float().numpy()
        masks_np = masks.detach().cpu().numpy()
        added = 0
        for index in range(images.shape[0]):
            position_bin = int(position_bins[index])
            if position_bin < 0 or position_bin >= self.position_bins:
                raise ValueError("position bin out of range: {}".format(position_bin))
            extracted = extract_profile_field(
                images_np[index, 0], masks_np[index], self.radius, self.sectors,
                self.min_foreground_pixels)
            if extracted is None:
                self.skipped += 1
                continue
            self._profiles[position_bin].append(extracted["field"].float())
            self._source_ids[position_bin].append(str(source_ids[index]))
            added += 1
        return added

    def freeze(self):
        if self._frozen:
            return self
        if len(self) == 0:
            raise RuntimeError("BMER bank has no valid donor profiles")
        for position_bin in range(self.position_bins):
            values = self._profiles[position_bin]
            if values:
                self._profiles[position_bin] = torch.stack(values, dim=0).contiguous()
            else:
                self._profiles[position_bin] = torch.empty(
                    (0, self.sectors, 2 * self.radius + 1), dtype=torch.float32)
        self._frozen = True
        return self

    def _resolved_bin(self, requested_bin):
        available = [index for index, values in self._profiles.items()
                     if values.shape[0] > 0]
        if not available:
            raise RuntimeError("BMER bank has no available position bin")
        return min(available, key=lambda index: (abs(index - requested_bin), index))

    def sample(self, position_bins, generator=None, device=None, dtype=None):
        if not self._frozen:
            raise RuntimeError("freeze the BMER bank before sampling")
        sampled = []
        resolved_bins = []
        donor_indices = []
        for requested in position_bins:
            requested = int(requested)
            if requested < 0 or requested >= self.position_bins:
                raise ValueError("position bin out of range: {}".format(requested))
            resolved = self._resolved_bin(requested)
            profiles = self._profiles[resolved]
            donor_index = int(torch.randint(
                profiles.shape[0], (1,), generator=generator, device="cpu").item())
            sampled.append(profiles[donor_index])
            resolved_bins.append(resolved)
            donor_indices.append(donor_index)
        result = torch.stack(sampled, dim=0)
        if device is not None or dtype is not None:
            result = result.to(device=device, dtype=dtype)
        return result, resolved_bins, donor_indices

    def summary(self):
        counts = {}
        for index, values in self._profiles.items():
            counts[index] = len(values) if isinstance(values, list) else int(values.shape[0])
        return {
            "version": self.VERSION,
            "radius": self.radius,
            "sectors": self.sectors,
            "position_bins": self.position_bins,
            "min_foreground_pixels": self.min_foreground_pixels,
            "counts": counts,
            "total": int(sum(counts.values())),
            "skipped": int(self.skipped),
            "frozen": bool(self._frozen),
        }

    def state_dict(self):
        if not self._frozen:
            raise RuntimeError("freeze the BMER bank before serialization")
        source_text = "\n".join(
            source for position_bin in range(self.position_bins)
            for source in self._source_ids[position_bin])
        return {
            "version": self.VERSION,
            "radius": self.radius,
            "sectors": self.sectors,
            "position_bins": self.position_bins,
            "min_foreground_pixels": self.min_foreground_pixels,
            "profiles": self._profiles,
            "source_ids": self._source_ids,
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "skipped": self.skipped,
        }

    def save(self, path):
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        if state.get("version") != cls.VERSION:
            raise ValueError("unsupported BMER bank version")
        bank = cls(state["radius"], state["sectors"], state["position_bins"],
                   state["min_foreground_pixels"])
        bank._profiles = {
            int(index): values.detach().cpu().float().contiguous()
            for index, values in state["profiles"].items()
        }
        bank._source_ids = {
            int(index): list(values)
            for index, values in state.get("source_ids", {}).items()
        }
        for index in range(bank.position_bins):
            bank._source_ids.setdefault(index, [])
        bank.skipped = int(state.get("skipped", 0))
        bank._frozen = True
        if len(bank) == 0:
            raise ValueError("serialized BMER bank is empty")
        return bank


@torch.no_grad()
def resynthesize_labeled_images(images, masks, bank, position_bins,
                                probability=0.5, strength=(0.5, 1.0),
                                generator=None):
    """Render sampled donor profiles on labeled recipient geometry.

    The result is detached, has the same shape/dtype/device as ``images``, and is
    bitwise identical outside the signed-distance ribbon.
    """
    _check_image_mask_batch(images, masks)
    if not isinstance(bank, BoundaryProfileBank) or not bank.frozen:
        raise ValueError("bank must be a frozen BoundaryProfileBank")
    if len(position_bins) != images.shape[0]:
        raise ValueError("position_bins length must match batch size")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be in [0, 1]")
    strength_min, strength_max = float(strength[0]), float(strength[1])
    if strength_min < 0.0 or strength_max < strength_min:
        raise ValueError("invalid BMER strength range")

    donor_fields, resolved_bins, donor_indices = bank.sample(
        position_bins, generator=generator, device="cpu", dtype=torch.float32)
    images_cpu = images.detach().cpu().float().numpy()
    masks_cpu = masks.detach().cpu().numpy()
    donor_cpu = donor_fields.numpy()
    augmented_cpu = images_cpu.copy()

    applied = []
    valid_masks = []
    strengths = []
    changed_fractions = []
    mean_absolute_changes = []

    for index in range(images.shape[0]):
        extracted = extract_profile_field(
            images_cpu[index, 0], masks_cpu[index], bank.radius, bank.sectors,
            bank.min_foreground_pixels)
        if extracted is None:
            applied.append(0.0)
            valid_masks.append(0.0)
            strengths.append(0.0)
            changed_fractions.append(0.0)
            mean_absolute_changes.append(0.0)
            continue

        apply_draw = float(torch.rand((1,), generator=generator,
                                      device="cpu").item())
        if apply_draw >= probability:
            applied.append(0.0)
            valid_masks.append(1.0)
            strengths.append(0.0)
            changed_fractions.append(0.0)
            mean_absolute_changes.append(0.0)
            continue

        if strength_min == strength_max:
            sampled_strength = strength_min
        else:
            draw = float(torch.rand((1,), generator=generator,
                                    device="cpu").item())
            sampled_strength = strength_min + (strength_max - strength_min) * draw

        geometry = extracted["geometry"]
        recipient_field = extracted["field"].numpy()
        sector_index = geometry["sector_index"]
        distance_index = geometry["distance_index"]
        recipient_lookup = recipient_field[sector_index, distance_index]
        donor_lookup = donor_cpu[index][sector_index, distance_index]
        delta = (sampled_strength * geometry["taper"] *
                 (donor_lookup - recipient_lookup) * extracted["scale"])

        original = images_cpu[index, 0]
        rendered = original + delta.astype(np.float32)
        rendered = np.clip(rendered, float(original.min()), float(original.max()))
        augmented_cpu[index, 0] = rendered

        actual_change = np.abs(rendered - original)
        applied.append(1.0)
        valid_masks.append(1.0)
        strengths.append(sampled_strength)
        changed_fractions.append(float((actual_change > 1e-8).mean()))
        mean_absolute_changes.append(float(actual_change.mean()))

    augmented = torch.from_numpy(augmented_cpu).to(
        device=images.device, dtype=images.dtype)
    metadata = {
        "applied_fraction": images.new_tensor(applied).mean(),
        "valid_mask_fraction": images.new_tensor(valid_masks).mean(),
        "strength_mean": images.new_tensor(strengths).mean(),
        "changed_fraction": images.new_tensor(changed_fractions).mean(),
        "mean_absolute_change": images.new_tensor(mean_absolute_changes).mean(),
        "resolved_position_bins": torch.as_tensor(resolved_bins, dtype=torch.long),
        "donor_indices": torch.as_tensor(donor_indices, dtype=torch.long),
    }
    return augmented.detach(), metadata
