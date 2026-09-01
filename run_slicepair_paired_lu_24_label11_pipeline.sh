#!/usr/bin/env bash
set -Eeuo pipefail

# Label-11 counterpart of paper A3. The method stays 6+6+12; only the
# labeled data pool and its matching Pre10000 checkpoint change.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
GPU="${GPU:-0}"
DETACH="${DETACH:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PIPELINE_WORKER="${PIPELINE_WORKER:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="SlicePairPairedLU24_35_5_10_Pre10000_Self30000_label11_seed1337"
SNAPSHOT="${ROOT}/model/${EXP_NAME}_11_labeled/self_train/unet"
CHECKPOINT="${SNAPSHOT}/unet_best_model.pth"
PERFORMANCE="${SNAPSHOT}/performance.txt"
LOG_DIR="${ROOT}/server_logs"
PIPELINE_LOG="${LOG_DIR}/slicepair_paired_lu_24_label11_${RUN_TAG}.log"

validate_inputs() {
  [[ -f "${DATA_ROOT}/train_slices.list" ]] || {
    echo "Missing PROMISE12 data: ${DATA_ROOT}" >&2
    exit 2
  }
  [[ -n "${PRETRAINED_CHECKPOINT}" ]] || {
    echo "Set PRETRAINED_CHECKPOINT to the matching label-11 Pre10000 net+opt checkpoint." >&2
    exit 2
  }
  [[ -f "${PRETRAINED_CHECKPOINT}" ]] || {
    echo "Missing label-11 Pre10000 checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
    exit 2
  }
  case "${PRETRAINED_CHECKPOINT,,}" in
    *label11*|*11_labeled*|*11label*) ;;
    *)
      echo "Refusing checkpoint path without a label-11 marker: ${PRETRAINED_CHECKPOINT}" >&2
      exit 2
      ;;
  esac
  case "${ALLOW_EXISTING}" in
    0|1) ;;
    *) echo "ALLOW_EXISTING must be 0 or 1." >&2; exit 2 ;;
  esac
  if [[ -d "${SNAPSHOT}" && "${ALLOW_EXISTING}" != "1" ]]; then
    echo "Refusing to reuse existing experiment directory: ${SNAPSHOT}" >&2
    echo "Move it aside, or set ALLOW_EXISTING=1 only for an intentional retest." >&2
    exit 2
  fi
}

run_pipeline() {
  echo "[$(date '+%F %T')] Starting label-11 paired-L/U-24 training"
  (
    cd "${ROOT}/code"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u \
      train_slicepair_paired_lu_24_label11.py \
      --root_path "${DATA_ROOT}" \
      --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
      --exp "${EXP_NAME}" --labelnum 11 \
      --occ_ablation paired_lu_24 --appearance_mode oaac_strong \
      --max_iterations 30000 --batch_size 24 --labeled_bs 12 --seed 1337
  )

  [[ -f "${CHECKPOINT}" ]] || {
    echo "Training finished without validation-best checkpoint: ${CHECKPOINT}" >&2
    exit 1
  }

  echo "[$(date '+%F %T')] Testing validation-selected label-11 checkpoint"
  (
    cd "${ROOT}/code"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u test_sliceeq_occ.py \
      --root_path "${DATA_ROOT}" --exp "${EXP_NAME}" \
      --checkpoint_path "${CHECKPOINT}" --labelnum 11 --gpu "${GPU}" \
      --save_result False --nms 0 --auto_find_checkpoint False
  )

  [[ -f "${PERFORMANCE}" ]] || {
    echo "Test finished without performance file: ${PERFORMANCE}" >&2
    exit 1
  }
  echo "[$(date '+%F %T')] Label-11 paired-L/U-24 completed"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Performance: ${PERFORMANCE}"
  cat "${PERFORMANCE}"
}

(cd "${ROOT}" && python -m unittest discover -s tests \
  -p 'test_slicepair_paired_lu_24_label11_contract.py')
validate_inputs
mkdir -p "${LOG_DIR}"

if [[ "${DETACH}" == "1" && "${PIPELINE_WORKER}" != "1" ]]; then
  nohup env DATA_ROOT="${DATA_ROOT}" \
    PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" GPU="${GPU}" \
    DETACH=0 ALLOW_EXISTING="${ALLOW_EXISTING}" PIPELINE_WORKER=1 \
    RUN_TAG="${RUN_TAG}" \
    bash "${ROOT}/run_slicepair_paired_lu_24_label11_pipeline.sh" \
      >"${PIPELINE_LOG}" 2>&1 </dev/null &
  echo "Started label-11 paired-L/U-24 pipeline: PID=$!"
  echo "Log: ${PIPELINE_LOG}"
else
  run_pipeline
fi
