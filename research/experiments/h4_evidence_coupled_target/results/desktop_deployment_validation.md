# Desktop Baseline Deployment Validation

Date: 2026-08-10

Scope: deploy the validated CoDA-MT implementation directly under the user's
desktop Baseline without overwriting original code, and make desktop PROMISE12
the new entry point's default data root.

## Deployed Files

| File | SHA-256 |
|---|---|
| `E:/Desktop/Baseline/code/train_coda.py` | `4512EC5737F183BA8CF5907DF4AF9D5B39AC25FAB030360CA3DAFE751189BC1C` |
| `E:/Desktop/Baseline/code/test_coda.py` | `CC8C1F68AC5883662F0E7A0B6D45E5F26F8881BCBAD318FBE83F15320E2FF66D` |
| `E:/Desktop/Baseline/code/utils/coda.py` | `9AF0D96DCC2542890D66BCEC8ED67251328C0C889188B206764C33655DACCD44` |
| `E:/Desktop/Baseline/code/utils/promise12_preflight.py` | `B5C473C3C3EA3EA422FA7FB54DDD15BF8E1CE5CB1811F9529961A713714B4498` |
| `E:/Desktop/Baseline/tests/test_coda.py` | `752A4BA76E216241D24CCE0C28397C1B6FF89F0CC8434EF50A76A21BA4BD4F16` |
| `E:/Desktop/Baseline/tests/test_deployment_contract.py` | `C18C05F52D6101361A9FBE645ABA05E1E627B5D9583ACE9F9E3FD0D08D412741` |
| `E:/Desktop/Baseline/CODA_README.md` | `06A06DDCBFE9611328A974E87497CC7C7FF9897E3234FC5AA388C6E3602C1D0F` |

## Original-Code Integrity

| Original file | SHA-256 after deployment | Result |
|---|---|---:|
| `code/train_baseline.py` | `54393FCB977A3E4F199420885B6F6ACBD8B1D2B320C820979F355C003CD3EEC8` | unchanged |
| `code/test_baseline.py` | `31CC57D26FB3476F55A593445E10BBC4EFA96E6DBFDE778BAA2D65C599491682` | unchanged |
| `code/dataloaders/dataset.py` | `7A5B3C28EBEAF7AA2F64E5F111F88BD15F6075AFC53238B477BD1A8511B2206A` | unchanged |

## Locked PROMISE12 Contract

| List | Count | Canonical ordered-entry SHA-256 |
|---|---:|---|
| `train.list` | 35 cases | `282BD77ADB9D57056625DB536C87E10D01F5E58E9D26C064B5E0AFA8889FBA08` |
| `train_slices.list` | 940 slices | `2621EBB12B53F31A2899916ACAB7CBE344B4447B9E8D654FBE4E8FAAE3E6972B` |
| `val.list` | 5 cases | `080CE173502F51E9DAD7E127E05DD0961E1343F05E2F5EECA1F1F9FA9F59CEF9` |
| `test.list` | 10 cases | `AD8D4ABA047A74EDA7679F57208BE8B38D742A08C05E57DFA2B1D58E54BDA37B` |

The first seven `train.list` cases contribute exactly the first 191 entries of
`train_slices.list`; no labeled-case slice appears after that boundary.

## Validation

- 9 CoDA numerical/gradient tests passed.
- 5 deployment/data-contract tests passed.
- The training worker callback is module-level and pickleable under Windows;
  a real `spawn` DataLoader with four workers read a `(2, 1, 256, 256)` batch.
- 955/955 referenced assets have valid HDF5 signatures.
- Six deployed Python files passed `py_compile`.
- Default training root: `E:/Desktop/PROMISE12`.
- Default experiment: `CoDA_MT_PROMISE12`.
- The PROMISE12 name is mapped to the unchanged prostate slice-count table, so
  `labelnum=7` still yields 191 labeled slices.

## Resolved Data Condition

The user replaced all Git LFS pointers with real H5 objects. The replacement
list files use CRLF instead of the original LF, changing raw file hashes but not
their entries. Preflight now hashes the canonical ordered non-empty line
sequence, ignoring LF/CRLF and a UTF-8 BOM while still rejecting any semantic
split change. All four canonical hashes match the locked values.

Read-only samples at labeled indices 0, 190, and 191 and from the validation and
test lists contain finite float32 images. Volume labels are binary int8, and all
expected `image`/`label` datasets are present.

This deployment produces no segmentation metric and does not count as a
confirmatory H4 run.
