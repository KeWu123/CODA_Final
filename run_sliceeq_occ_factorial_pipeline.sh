#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_NAME="factorial_lu"
export STAGES="occ_l_only occ_u_only"
exec bash "${ROOT}/run_sliceeq_occ_ablation_pipeline.sh"
