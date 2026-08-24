#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
GPU="${GPU:-0}"
DETACH="${DETACH:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PIPELINE_WORKER="${PIPELINE_WORKER:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="SliceEqOccOAACStrong_PROMISE12"
LOG_DIR="${ROOT}/server_logs"
PIPELINE_LOG="${LOG_DIR}/sliceeq_occ_oaac_strong_${RUN_TAG}.log"
SNAPSHOT="${ROOT}/model/${EXP_NAME}_7_labeled/self_train/unet"
CHECKPOINT="${SNAPSHOT}/unet_best_model.pth"

validate_inputs() {
  [[ -f "${DATA_ROOT}/train_slices.list" ]] || {
    echo "Missing PROMISE12 data: ${DATA_ROOT}" >&2
    exit 2
  }
  [[ -f "${PRETRAINED_CHECKPOINT}" ]] || {
    echo "Missing shared label7 Pre10000 checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
    exit 2
  }
  case "${PRETRAINED_CHECKPOINT}" in
    *7_labeled*|*label7*|*7label*) ;;
    *) echo "Checkpoint path has no label7 marker." >&2; exit 2 ;;
  esac
  case "${ALLOW_EXISTING}" in
    0|1) ;;
    *) echo "ALLOW_EXISTING must be 0 or 1." >&2; exit 2 ;;
  esac
  if [[ "${ALLOW_EXISTING}" != "1" && -d "${SNAPSHOT}" ]]; then
    echo "Refusing existing experiment directory: ${SNAPSHOT}" >&2
    exit 2
  fi
}

run_pipeline() {
  local test_log="${LOG_DIR}/sliceeq_occ_oaac_strong_test_${RUN_TAG}.log"
  echo "[$(date '+%F %T')] Training ${EXP_NAME}"
  (
    cd "${ROOT}/code"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u train_sliceeq_occ_oaac_strong_portable.py \
      --root_path "${DATA_ROOT}" \
      --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
      --exp "${EXP_NAME}" --max_iterations 30000 --batch_size 24 \
      --labeled_bs 12 --labelnum 7 --seed 1337
  )

  [[ -f "${CHECKPOINT}" ]] || {
    echo "Training finished without validation-best checkpoint: ${CHECKPOINT}" >&2
    exit 1
  }

  echo "[$(date '+%F %T')] Testing validation-selected ${EXP_NAME}"
  (
    cd "${ROOT}/code"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u test_sliceeq_occ_oaac_strong.py \
      --root_path "${DATA_ROOT}" --exp "${EXP_NAME}" \
      --checkpoint_path "${CHECKPOINT}" --labelnum 7 --gpu "${GPU}" \
      --save_result False --nms 0 --auto_find_checkpoint False
  ) 2>&1 | tee "${test_log}"
  echo "[$(date '+%F %T')] OAAC Strong 1.25x training and strict test completed."
}

validate_inputs
mkdir -p "${LOG_DIR}"

if [[ "${DETACH}" == "1" && "${PIPELINE_WORKER}" != "1" ]]; then
  nohup env DATA_ROOT="${DATA_ROOT}" \
    PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
    GPU="${GPU}" DETACH=0 ALLOW_EXISTING="${ALLOW_EXISTING}" \
    PIPELINE_WORKER=1 RUN_TAG="${RUN_TAG}" \
    bash "${ROOT}/run_sliceeq_occ_oaac_strong.sh" \
      >"${PIPELINE_LOG}" 2>&1 </dev/null &
  echo "Started OAAC Strong 1.25x pipeline: PID=$!"
  echo "Log: ${PIPELINE_LOG}"
else
  run_pipeline
fi
