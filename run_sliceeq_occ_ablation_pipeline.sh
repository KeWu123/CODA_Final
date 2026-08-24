#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
GPU="${GPU:-0}"
DETACH="${DETACH:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PIPELINE_WORKER="${PIPELINE_WORKER:-0}"
PIPELINE_NAME="${PIPELINE_NAME:-ablation}"
STAGES="${STAGES:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/server_logs"
PIPELINE_LOG="${LOG_DIR}/sliceeq_occ_${PIPELINE_NAME}_pipeline_${RUN_TAG}.log"

read -r -a STAGE_LIST <<< "${STAGES}"

experiment_name() {
  local stage="$1"
  echo "SliceEqOccIncremental_${stage}_35_5_10_Pre10000_Self30000_label7_seed1337"
}

validate_inputs() {
  local stage exp snapshot
  [[ ${#STAGE_LIST[@]} -gt 0 ]] || {
    echo "No ablation stages were configured." >&2
    exit 2
  }
  for stage in "${STAGE_LIST[@]}"; do
    case "${stage}" in
      baseline|baseline_36|image_only|image_only_36|aligned_occ|hard_targets|occ_l_only|occ_u_only|full) ;;
      *) echo "Unsupported ablation stage: ${stage}" >&2; exit 2 ;;
    esac
  done
  [[ -f "${DATA_ROOT}/train_slices.list" ]] || {
    echo "Missing PROMISE12 data: ${DATA_ROOT}" >&2
    exit 2
  }
  [[ -n "${PRETRAINED_CHECKPOINT}" ]] || {
    echo "Set PRETRAINED_CHECKPOINT to the shared label-7 Pre10000 net+opt checkpoint." >&2
    exit 2
  }
  [[ -f "${PRETRAINED_CHECKPOINT}" ]] || {
    echo "Missing Pre10000 checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
    exit 2
  }
  case "${PRETRAINED_CHECKPOINT}" in
    *7_labeled*|*label7*|*7label*) ;;
    *)
      echo "Refusing ambiguous checkpoint without a label-7 marker: ${PRETRAINED_CHECKPOINT}" >&2
      exit 2
      ;;
  esac
  case "${ALLOW_EXISTING}" in
    0|1) ;;
    *) echo "ALLOW_EXISTING must be 0 or 1." >&2; exit 2 ;;
  esac

  if [[ "${ALLOW_EXISTING}" != "1" ]]; then
    for stage in "${STAGE_LIST[@]}"; do
      exp="$(experiment_name "${stage}")"
      snapshot="${ROOT}/model/${exp}_7_labeled/self_train/unet"
      [[ ! -d "${snapshot}" ]] || {
        echo "Refusing to reuse existing experiment directory: ${snapshot}" >&2
        echo "Move it aside, or set ALLOW_EXISTING=1 to intentionally reuse it." >&2
        exit 2
      }
    done
  fi
}

run_pipeline() {
  local stage exp snapshot checkpoint test_log performance

  for stage in "${STAGE_LIST[@]}"; do
    exp="$(experiment_name "${stage}")"
    snapshot="${ROOT}/model/${exp}_7_labeled/self_train/unet"
    checkpoint="${snapshot}/unet_best_model.pth"
    test_log="${LOG_DIR}/sliceeq_occ_test_${stage}_${RUN_TAG}.log"
    performance="${snapshot}/performance.txt"

    echo "[$(date '+%F %T')] Starting ${stage} training (${exp})"
    DATA_ROOT="${DATA_ROOT}" \
    PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
    OCC_ABLATION="${stage}" EXP_NAME="${exp}" GPU="${GPU}" \
    DETACH=0 RUN_TAG="${RUN_TAG}" \
      bash "${ROOT}/run_sliceeq_occ_ablation.sh"

    [[ -f "${checkpoint}" ]] || {
      echo "Training finished without a best checkpoint: ${checkpoint}" >&2
      exit 1
    }

    echo "[$(date '+%F %T')] Starting ${stage} test"
    (
      cd "${ROOT}/code"
      CUDA_VISIBLE_DEVICES="${GPU}" python -u test_sliceeq_occ.py \
        --root_path "${DATA_ROOT}" \
        --exp "${exp}" \
        --checkpoint_path "${checkpoint}" \
        --labelnum 7 --gpu "${GPU}" --save_result False --nms 0 \
        --auto_find_checkpoint False
    ) 2>&1 | tee "${test_log}"

    [[ -f "${performance}" ]] || {
      echo "Test finished without performance file: ${performance}" >&2
      exit 1
    }
    echo "[$(date '+%F %T')] Completed ${stage}: ${performance}"
  done

  echo
  echo "================ ${PIPELINE_NAME} performance files ================"
  for stage in "${STAGE_LIST[@]}"; do
    exp="$(experiment_name "${stage}")"
    performance="${ROOT}/model/${exp}_7_labeled/self_train/unet/performance.txt"
    echo "--- ${stage}: ${performance}"
    cat "${performance}"
  done
  echo "[$(date '+%F %T')] ${PIPELINE_NAME} training and testing completed."
}

validate_inputs
mkdir -p "${LOG_DIR}"

if [[ "${DETACH}" == "1" && "${PIPELINE_WORKER}" != "1" ]]; then
  nohup env \
    DATA_ROOT="${DATA_ROOT}" \
    PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
    GPU="${GPU}" DETACH=0 ALLOW_EXISTING="${ALLOW_EXISTING}" \
    PIPELINE_WORKER=1 PIPELINE_NAME="${PIPELINE_NAME}" STAGES="${STAGES}" \
    RUN_TAG="${RUN_TAG}" \
    bash "${ROOT}/run_sliceeq_occ_ablation_pipeline.sh" \
      >"${PIPELINE_LOG}" 2>&1 </dev/null &
  echo "Started SliceEqOcc ${PIPELINE_NAME} pipeline: PID=$!"
  echo "Log: ${PIPELINE_LOG}"
else
  run_pipeline
fi
