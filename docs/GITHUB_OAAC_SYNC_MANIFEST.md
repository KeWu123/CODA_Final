# OAAC GitHub synchronization manifest

Prepared: 2026-08-14

This manifest defines the reviewed OAAC source and documentation set. It is
deliberately narrower than the local research workspace: medical data,
checkpoints, NIfTI/H5 files, external logs, performance dumps, credentials and
Python caches must not be committed.

## Proposed synchronization

- Repository: `XGS61/CoDA-MT-PROMISE12` (the sole accessible repository
  matching the local CoDA/PROMISE12 project, confirmed with administrator
  permission before synchronization).
- Target branch: `main`, as explicitly requested by the repository owner on
  2026-08-14. The synchronization is prepared in an isolated clone and is
  pushed only as a fast-forward update; force-push remains prohibited.
- Merge policy: clone/fetch the existing remote, apply only the reviewed files,
  inspect the diff, run tests, scan for secrets, then push normally. Never
  force-push or initialize the current snapshot as a replacement remote.
- Result provenance: publish the manifest and analysis, but not the external
  `log.txt` or `performance.txt` files.

## Reviewed source and documentation

| SHA-256 | Path |
|---|---|
| `6ba4738591a863a7b9ae9a4a8a051f07fdbdeb05937c60c80805a6851b82eafe` | `.gitignore` |
| `a4a12e54808eafb1a63761a7ee323e467f5ba18b98f868d27e7a3d00f1209e43` | `code/train_sliceeq_occ_oaac.py` |
| `4f7295a870c3529f2deefb6c9f2d1e4f3c6ec556bd3c5f2260a4d2553fe0c428` | `code/test_sliceeq_occ_oaac.py` |
| `1cc0fdf5397d0fbb4d754efb2844b4882c4a22d95db7bcba70ddbee8f9b0bc97` | `code/utils/sliceeq_oaac.py` |
| `633fef8b28269046765c553c7270bb72b1e2daa0fd2a4ffebcda9c60aaf74a4b` | `tests/test_sliceeq_oaac.py` |
| `1de6227a1abedcaa373635789152616a0d78908028d6aae78a581777ccfe8111` | `tests/test_sliceeq_oaac_contract.py` |
| `758e47ed9ce1c21ae95ad4e52f9343e00408f0a20583f608f9e0f92429d4d2ff` | `docs/SLICEEQ_OCC_OAAC_README.md` |
| `328c56645ef51fd191043e0583b7f112621701f6a6b79593325f7fd374874224` | `docs/SLICEEQ_OCC_OAAC_MODEL_AND_PAPER_GUIDE.md` |
| `146fecfa6e02b2f095f319fb03d8c827b25e26e0af6a7431277d7c35480827f8` | `research/experiments/h7_slice_profile_reacquisition/h7_13_oaac_protocol.md` |
| `7bef89e98d312e986a37ddeb48f3eec9c7eeb62222c3d2be65063f3e2aeda95e` | `research/experiments/h7_slice_profile_reacquisition/results/sliceeq_occ_oaac_external_run_2026-08-14/result_manifest.md` |
| `47c58bf02eee2eff181ad3ec794755446a6b155910367b5fdcb07abd3a922ac3` | `research/experiments/h7_slice_profile_reacquisition/results/sliceeq_occ_oaac_external_run_2026-08-14/analysis.md` |
| `5c5961b650a9d29be096d275190003a78695f16a608049a00adfd558ff4660eb` | `research/paper/sliceeq_occ_cvpr_outline_2026-08-13.md` |

The hashes above describe the files at the time this manifest was generated.
Recompute them immediately before synchronization and investigate any mismatch.
The root `README.md` is intentionally omitted from the immutable table because
the existing remote version must be merged rather than overwritten wholesale.

## Pre-push checks

```bash
python -m unittest tests.test_sliceeq_oaac tests.test_sliceeq_oaac_contract -v
```

The tensor suite must execute in a PyTorch/CUDA environment; a skipped CUDA
isolation test is not a successful release check. Also verify that the staged
diff contains no patient identifiers, data files, weights, machine credentials
or raw experiment artifacts.

## Known release limitations

- The training entry is artifact-locked to a private-server recipe and pretrain
  hash. The detailed guide states this explicitly; a portable configuration
  entry can be added later without changing the algorithm.
- The repository contains inherited code with more than one upstream license.
  Do not apply a new blanket repository license until all inherited files have
  been audited.
- OAAC's current `0.849538` Dice is the maximum selected after multiple
  checkpoints were inspected on the local test split. It is a test-selected
  development oracle; evaluating validation-best later cannot restore an
  untouched primary test. Fresh hidden/external evaluation is still required.
