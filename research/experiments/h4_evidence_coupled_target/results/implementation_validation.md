# CoDA-MT Implementation Validation

Date: 2026-08-10

Scope: code/infrastructure validation only

PROMISE12 training runs: **0**

## Isolation Result

- Source Baseline: `E:/Desktop/Baseline` (read-only).
- Implementation: `implementation/coda_mt_baseline`.
- The three audited source hashes remain unchanged after implementation.
- IDE files, Python caches, and unrelated `logs/window_ablation` files were not
  copied.
- Twenty-two copied Python files not on the intentional-change allowlist match
  their source SHA-256 hashes.

## Intentional Differences

1. `code/train_baseline.py`
   - imports CoDA helpers;
   - adds only `coda_`-prefixed method arguments;
   - creates a dedicated augmentation RNG from `args.seed`;
   - changes the copied `self_train` unlabeled view/target/loss path;
   - adds CoDA TensorBoard diagnostics.
2. `code/test_baseline.py`
   - removes only the terminal bare `vvvv...` expression that raises
     `NameError` after result output.
3. New helper, test, README, and provenance files.

No dataset, network, supervised pretraining, optimizer, EMA update, schedule,
validation, checkpoint, or inference logic was changed.

## Automated Checks

| Check | Result |
|---|---:|
| CoDA utility tests | 9 passed |
| Baseline contract tests | 4 passed |
| Total unit tests | 13 passed |
| Python AST parse | 27 files passed |
| `compileall` | passed |
| Unchanged-copy SHA-256 comparison | 22 files passed |
| Source Baseline core hashes after work | unchanged |

The utility tests cover deterministic augmentation, resolution evidence loss,
constant-background behavior, probability normalization, `gamma=0/1` limits,
LCC soft-probability retention, hard-CE equivalence, finite soft losses, and
student-logit backpropagation.

## Environment Limitation

`python code/train_baseline.py --help` reaches the original import section but
fails because the current Windows Python environment does not provide the
Baseline's existing `tensorboardX` dependency. The test environment also loads
duplicate Intel OpenMP runtimes; `KMP_DUPLICATE_LIB_OK=TRUE` was used only for
the isolated CPU test process. Neither condition was written into training
code or project configuration.

Full GPU/data-path validation must use the user's established Linux Baseline
environment and the unchanged list files. This report makes no segmentation
performance claim.
