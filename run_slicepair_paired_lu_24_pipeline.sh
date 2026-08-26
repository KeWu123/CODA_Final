#!/usr/bin/env bash
set -Eeuo pipefail

# Paper A3: 6 native-L + 6 paired-L + 12 paired-U student views.
# The shared runner preserves the 1k identity warm-up, trains Self30000,
# evaluates only the validation-selected checkpoint, and writes performance.txt.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export STAGES="paired_lu_24"
export PIPELINE_NAME="slicepair_paired_lu_24"
export APPEARANCE_MODE="oaac_strong"

(cd "${ROOT}" && python -m unittest discover -s tests \
  -p 'test_sliceeq_occ_ablation_contract.py')

exec bash "${ROOT}/run_sliceeq_occ_ablation_pipeline.sh"
