#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Current paper table: A0--A3 use 24 student views, A4--A5 use 36. All rows
# share OAAC-Strong, the Pre10000 state, seed, schedule, and validation-best
# selection. hard_targets remains a supplementary mechanism control.
export STAGES="${STAGES:-paper_a0 paper_a1 paper_a2 paper_a3 paper_a4 paper_a5}"
export SUMMARY_PRESET="${SUMMARY_PRESET:-paper_mpd}"
export RESULT_BASENAME="${RESULT_BASENAME:-slicepair_mpd_ablation}"
export RUN_FIGURES="${RUN_FIGURES:-0}"
export RUN_AUDIT="${RUN_AUDIT:-0}"
exec bash "${ROOT}/run_coda_final_paper_suite.sh"
