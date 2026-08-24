import ast
import hashlib
import sys
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = BASELINE_ROOT / "code"
TRAIN_PATH = CODE_ROOT / "train_coda.py"
DATA_ROOT = Path("/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source")
sys.path.insert(0, str(CODE_ROOT))

from utils.promise12_preflight import (  # noqa: E402
    DatasetPreflightError,
    validate_promise12_root,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser_defaults(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    return defaults


class DesktopDeploymentContractTest(unittest.TestCase):
    def test_original_baseline_files_are_unchanged(self):
        expected = {
            "code/train_baseline.py": "54393fcb977a3e4f199420885b6f6acbd8b1d2b320c820979f355c003cd3eec8",
            "code/test_baseline.py": "31cc57d26fb3476f55a593445e10bbc4efa96e6dbfde778baa2d65c599491682",
            "code/dataloaders/dataset.py": "7a5b3c28ebeaf7aa2f64e5f111f88bd15f6075afc53238b477bd1a8511b2206a",
        }
        for relative_path, expected_hash in expected.items():
            self.assertEqual(
                sha256(BASELINE_ROOT / relative_path), expected_hash,
                relative_path)

    def test_coda_defaults_match_current_locked_path_without_changing_recipe(self):
        defaults = parser_defaults(TRAIN_PATH)
        expected = {
            "--root_path": "/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source",
            "--exp": "CoDA_MT_PROMISE12",
            "--model": "unet",
            "--pre_iterations": 10000,
            "--max_iterations": 30000,
            "--batch_size": 24,
            "--base_lr": 0.01,
            "--patch_size": [256, 256],
            "--seed": 1337,
            "--num_classes": 2,
            "--labeled_bs": 12,
            "--labelnum": 7,
            "--ema_decay": 0.99,
            "--consistency": 0.1,
            "--consistency_rampup": 200.0,
            "--pretrained_checkpoint":
                "/home/aiteam/zhengtaoma/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth",
        }
        for name, value in expected.items():
            self.assertEqual(defaults[name], value, name)

    def test_dataloader_worker_callback_is_windows_spawn_safe(self):
        tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"),
                         filename=str(TRAIN_PATH))
        module_functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("seed_data_worker", module_functions)

        training_functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"pre_train", "self_train"}
        }
        self.assertEqual(set(training_functions), {"pre_train", "self_train"})
        for function in training_functions.values():
            nested_names = {
                node.name for node in function.body
                if isinstance(node, ast.FunctionDef)
            }
            self.assertNotIn("worker_init_fn", nested_names)

            worker_callbacks = []
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name) or node.func.id != "DataLoader":
                    continue
                worker_callbacks.extend(
                    keyword.value for keyword in node.keywords
                    if keyword.arg == "worker_init_fn")
            self.assertEqual(len(worker_callbacks), 1)
            callback = worker_callbacks[0]
            self.assertIsInstance(callback, ast.Call)
            self.assertIsInstance(callback.func, ast.Name)
            self.assertEqual(callback.func.id, "partial")
            self.assertIsInstance(callback.args[0], ast.Name)
            self.assertEqual(callback.args[0].id, "seed_data_worker")

    def test_coda_reuses_fixed_pretrain_without_running_pretrain(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("load_pretrained_checkpoint(", source)
        self.assertIn("checkpoint_sha256(pretrained_checkpoint)", source)
        self.assertIn("reset_stage_rng(args.seed)", source)
        self.assertIn("model.load_state_dict(checkpoint['net'], strict=True)",
                      source)
        self.assertIn("optimizer.load_state_dict(checkpoint['opt'])", source)
        self.assertNotIn("    pre_train(args, pre_snapshot_path)", source)
        self.assertNotIn("glob.glob", source)

    def test_locked_split_and_191_slice_boundary(self):
        if not DATA_ROOT.is_dir():
            self.skipTest("Current locked data root is unavailable: {}".format(DATA_ROOT))
        report = validate_promise12_root(
            str(DATA_ROOT), strict_split=True, check_hdf5=False)
        self.assertEqual(report["train_cases"], 35)
        self.assertEqual(report["train_slices"], 940)
        self.assertEqual(report["labeled_slices"], 191)
        self.assertEqual(report["validation_cases"], 5)
        self.assertEqual(report["test_cases"], 10)

    def test_hdf5_assets_or_report_lfs_pointer_state(self):
        if not DATA_ROOT.is_dir():
            self.skipTest("Current locked data root is unavailable: {}".format(DATA_ROOT))
        try:
            validate_promise12_root(
                str(DATA_ROOT), strict_split=True, check_hdf5=True)
        except DatasetPreflightError as error:
            if "Git LFS pointer" in str(error):
                self.skipTest(str(error))
            raise


if __name__ == "__main__":
    unittest.main()
