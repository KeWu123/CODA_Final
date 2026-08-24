#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_NAME="paper_main"
export STAGES="baseline_36 image_only_36 hard_targets full"
exec bash "${ROOT}/run_sliceeq_occ_ablation_pipeline.sh"
