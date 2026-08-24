#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_NAME="m2_m3"
export STAGES="aligned_occ full"
exec bash "${ROOT}/run_sliceeq_occ_ablation_pipeline.sh"
