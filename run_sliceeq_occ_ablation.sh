#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
OCC_ABLATION="${OCC_ABLATION:-image_only}"
APPEARANCE_MODE="${APPEARANCE_MODE:-none}"
EXP_NAME="${EXP_NAME:-SliceEqOccIncremental_${OCC_ABLATION}_35_5_10_Pre10000_Self30000_label7_seed1337}"
GPU="${GPU:-0}"
DETACH="${DETACH:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/server_logs"
LOG_FILE="${LOG_DIR}/sliceeq_occ_ablation_${OCC_ABLATION}_${RUN_TAG}.log"

case "${OCC_ABLATION}" in
  baseline|baseline_36|image_only|image_only_36|aligned_occ|paired_lu_24|hard_targets|occ_l_only|occ_u_only|full|no_labeled_reacq) ;;
  *) echo "Unsupported OCC_ABLATION=${OCC_ABLATION}" >&2; exit 2 ;;
esac
case "${APPEARANCE_MODE}" in
  none|oaac_strong) ;;
  *) echo "Unsupported APPEARANCE_MODE=${APPEARANCE_MODE}" >&2; exit 2 ;;
esac
[[ -f "${DATA_ROOT}/train_slices.list" ]] || { echo "Missing PROMISE12 data: ${DATA_ROOT}" >&2; exit 2; }
[[ -n "${PRETRAINED_CHECKPOINT}" ]] || { echo "Set PRETRAINED_CHECKPOINT to the shared label-7 Pre10000 net+opt checkpoint." >&2; exit 2; }
[[ -f "${PRETRAINED_CHECKPOINT}" ]] || { echo "Missing Pre10000 checkpoint: ${PRETRAINED_CHECKPOINT}" >&2; exit 2; }
case "${PRETRAINED_CHECKPOINT}" in
  *7_labeled*|*label7*|*7label*) ;;
  *) echo "Refusing ambiguous checkpoint without a label-7 marker: ${PRETRAINED_CHECKPOINT}" >&2; exit 2 ;;
esac

mkdir -p "${LOG_DIR}"
cd "${ROOT}/code"
ARGS=(python -u "${ROOT}/code/train_sliceeq_occ_ablation.py" --root_path "${DATA_ROOT}" --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" --exp "${EXP_NAME}" --occ_ablation "${OCC_ABLATION}" --appearance_mode "${APPEARANCE_MODE}" --max_iterations 30000 --batch_size 24 --labeled_bs 12 --labelnum 7 --seed 1337)
if [[ "${DETACH}" == "1" ]]; then
  nohup env CUDA_VISIBLE_DEVICES="${GPU}" "${ARGS[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
  echo "Started SliceEqOcc ablation ${OCC_ABLATION}: PID=$!"
  echo "Log: ${LOG_FILE}"
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi
