#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_NAME="m0_m1"
export STAGES="baseline image_only"
exec bash "${ROOT}/run_sliceeq_occ_ablation_pipeline.sh"
