"""Read-only validation for the locked PROMISE12 split and HDF5 assets."""

import hashlib
import os


EXPECTED_LIST_SHA256 = {
    "train.list": "282bd77adb9d57056625db536c87e10d01f5e58e9d26c064b5e0afa8889fba08",
    "train_slices.list": "2621ebb12b53f31a2899916acab7cbe344b4447b9e8d654fbe4e8faae3e6972b",
    "val.list": "080ce173502f51e9dad7e127e05dd0961e1343f05e2f5eeca1f1f9fa9f59cef9",
    "test.list": "ad8d4aba047a74eda7679f57208be8b38d742a08c05e57dfa2b1d58e54bda37b",
}
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
GIT_LFS_PREFIX = b"version https://git-lfs.github.com/spec/"


class DatasetPreflightError(RuntimeError):
    pass


def _canonical_list_sha256(lines):
    """Hash ordered entries while ignoring LF/CRLF and a UTF-8 BOM."""
    canonical_bytes = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _read_list(root_path, filename):
    path = os.path.join(root_path, filename)
    if not os.path.isfile(path):
        raise DatasetPreflightError("Missing PROMISE12 list: {}".format(path))
    with open(path, "r", encoding="utf-8-sig") as stream:
        return [line.strip() for line in stream if line.strip()]


def _asset_state(path):
    with open(path, "rb") as stream:
        prefix = stream.read(max(len(HDF5_SIGNATURE), len(GIT_LFS_PREFIX)))
    if prefix.startswith(HDF5_SIGNATURE):
        return "hdf5"
    if prefix.startswith(GIT_LFS_PREFIX):
        return "git_lfs_pointer"
    return "unknown"


def validate_promise12_root(root_path, strict_split=True, check_hdf5=True):
    """Validate without changing list files, data, or sampling order."""
    root_path = os.path.abspath(os.path.expanduser(root_path))
    if not os.path.isdir(root_path):
        raise DatasetPreflightError(
            "PROMISE12 root does not exist: {}".format(root_path))

    train_cases = _read_list(root_path, "train.list")
    train_slices = _read_list(root_path, "train_slices.list")
    val_cases = _read_list(root_path, "val.list")
    test_cases = _read_list(root_path, "test.list")

    if strict_split:
        list_contents = {
            "train.list": train_cases,
            "train_slices.list": train_slices,
            "val.list": val_cases,
            "test.list": test_cases,
        }
        for filename, expected_hash in EXPECTED_LIST_SHA256.items():
            actual_hash = _canonical_list_sha256(list_contents[filename])
            if actual_hash != expected_hash:
                raise DatasetPreflightError(
                    "Locked split entries changed: {} (expected {}, got {})".format(
                        filename, expected_hash, actual_hash))

    expected_counts = {
        "train cases": (len(train_cases), 35),
        "train slices": (len(train_slices), 940),
        "validation cases": (len(val_cases), 5),
        "test cases": (len(test_cases), 10),
    }
    for name, (actual, expected) in expected_counts.items():
        if actual != expected:
            raise DatasetPreflightError(
                "Locked {} count changed: expected {}, got {}".format(
                    name, expected, actual))

    labeled_cases = set(train_cases[:7])
    first_labeled_slices = train_slices[:191]
    first_labeled_case_names = {
        item.split("_slice_", 1)[0] for item in first_labeled_slices
    }
    if first_labeled_case_names != labeled_cases:
        raise DatasetPreflightError(
            "The first 191 slices no longer match the first 7 train cases")
    if any(item.split("_slice_", 1)[0] in labeled_cases
           for item in train_slices[191:]):
        raise DatasetPreflightError(
            "A labeled-case slice appears after the locked index 191 boundary")

    slice_paths = [
        os.path.join(root_path, "data", "slices", item + ".h5")
        for item in train_slices
    ]
    volume_paths = [
        os.path.join(root_path, "data", item + ".h5")
        for item in val_cases + test_cases
    ]
    missing = [path for path in slice_paths + volume_paths
               if not os.path.isfile(path)]
    if missing:
        raise DatasetPreflightError(
            "PROMISE12 list references {} missing files; first: {}".format(
                len(missing), missing[0]))

    states = {"hdf5": 0, "git_lfs_pointer": 0, "unknown": 0}
    if check_hdf5:
        for path in slice_paths + volume_paths:
            states[_asset_state(path)] += 1
        if states["git_lfs_pointer"]:
            raise DatasetPreflightError(
                "PROMISE12 contains {} Git LFS pointer files instead of HDF5 "
                "data. Download the LFS objects (normally `git lfs pull` in "
                "a full clone) or replace the pointers with the real H5 files."
                .format(states["git_lfs_pointer"]))
        if states["unknown"]:
            raise DatasetPreflightError(
                "PROMISE12 contains {} files without an HDF5 signature"
                .format(states["unknown"]))

    return {
        "root_path": root_path,
        "train_cases": len(train_cases),
        "labeled_cases": 7,
        "labeled_slices": 191,
        "train_slices": len(train_slices),
        "validation_cases": len(val_cases),
        "test_cases": len(test_cases),
        "asset_states": states,
    }
