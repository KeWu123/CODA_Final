#!/usr/bin/env bash
set -Eeuo pipefail

# Launcher for SliceEqOcc-OAAC-Strong-MPD training and testing.
# Usage:
#   DATA_ROOT=/path/to/PROMISE12_h5 PRETRAINED_CHECKPOINT=/path/to/unet_best_model.pth \
#     bash run_sliceeq_occ_oaac_strong_mpd.sh
#   # Or to test a specific checkpoint:
#   DATA_ROOT=... PRETRAINED_CHECKPOINT=... TEST_CHECKPOINT=/path/to/iter_27000.pth \
#     bash run_sliceeq_occ_oaac_strong_mpd.sh

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
TEST_CHECKPOINT="${TEST_CHECKPOINT:-}"
GPU="${GPU:-0}"
DETACH="${DETACH:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PIPELINE_WORKER="${PIPELINE_WORKER:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="SliceEqOccOAACStrongMPD_PROMISE12"
LOG_DIR="${ROOT}/server_logs"
PIPELINE_LOG="${LOG_DIR}/sliceeq_occ_oaac_strong_mpd_${RUN_TAG}.log"
SNAPSHOT="${ROOT}/model/${EXP_NAME}_7_labeled/self_train/unet"
BEST_CHECKPOINT="${SNAPSHOT}/unet_best_model.pth"

validate_inputs() {
  [[ -f "${DATA_ROOT}/train_slices.list" ]] || {
    echo "Missing PROMISE12 data: ${DATA_ROOT}" >&2
    exit 2
  }
  [[ -n "${PRETRAINED_CHECKPOINT}" && -f "${PRETRAINED_CHECKPOINT}" ]] || {
    echo "Missing shared label7 Pre10000 checkpoint." >&2
    echo "Set PRETRAINED_CHECKPOINT=/path/to/unet_best_model.pth" >&2
    echo "Expected SHA-256: 49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9" >&2
    exit 2
  }
  case "${ALLOW_EXISTING}" in
    0|1) ;;
    *) echo "ALLOW_EXISTING must be 0 or 1." >&2; exit 2 ;;
  esac
  if [[ "${ALLOW_EXISTING}" != "1" && -d "${SNAPSHOT}" ]]; then
    echo "Refusing existing experiment directory: ${SNAPSHOT}" >&2
    echo "Set ALLOW_EXISTING=1 to overwrite." >&2
    exit 2
  fi
}

run_contract_tests() {
  echo "[$(date '+%F %T')] Running MPD contract tests..."
  (
    cd "${ROOT}"
    python -m unittest tests.test_sliceeq_mpd tests.test_sliceeq_mpd_contract -v
  )
}

run_training() {
  echo "[$(date '+%F %T')] Training ${EXP_NAME}"
  (
    cd "${ROOT}/code"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u train_sliceeq_occ_oaac_strong_mpd_portable.py \
      --root_path "${DATA_ROOT}" \
      --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  )
}

run_test() {
  local checkpoint="${1:-${BEST_CHECKPOINT}}"
  local test_log="${LOG_DIR}/sliceeq_occ_oaac_strong_mpd_test_${RUN_TAG}.log"
  echo "[$(date '+%F %T')] Testing checkpoint: ${checkpoint}"
  (
    cd "${ROOT}/code"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u test_sliceeq_occ_oaac_strong_mpd.py \
      --root_path "${DATA_ROOT}" --exp "${EXP_NAME}" \
      --checkpoint_path "${checkpoint}" --labelnum 7 --gpu "${GPU}" \
      --save_result False --nms 0 --auto_find_checkpoint False
  ) 2>&1 | tee "${test_log}"
  echo "[$(date '+%F %T')] MPD training and strict test completed."
}

run_pipeline() {
  run_contract_tests
  run_training
  if [[ -f "${BEST_CHECKPOINT}" ]]; then
    run_test "${BEST_CHECKPOINT}"
  else
    echo "Training finished without validation-best checkpoint." >&2
    exit 1
  fi
}

validate_inputs
mkdir -p "${LOG_DIR}"

if [[ -n "${TEST_CHECKPOINT}" ]]; then
  run_test "${TEST_CHECKPOINT}"
  exit 0
fi

if [[ "${DETACH}" == "1" && "${PIPELINE_WORKER}" != "1" ]]; then
  nohup env DATA_ROOT="${DATA_ROOT}" \
    PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
    GPU="${GPU}" DETACH=0 ALLOW_EXISTING="${ALLOW_EXISTING}" \
    PIPELINE_WORKER=1 RUN_TAG="${RUN_TAG}" \
    bash "${ROOT}/run_sliceeq_occ_oaac_strong_mpd.sh" \
      >"${PIPELINE_LOG}" 2>&1 </dev/null &
  echo "Started MPD pipeline: PID=$!"
  echo "Log: ${PIPELINE_LOG}"
else
  run_pipeline
fi
