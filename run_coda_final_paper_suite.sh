#!/usr/bin/env bash
set -Eeuo pipefail

# Complete label-7 PROMISE12 paper suite. All trainable stages share the same
# Pre10000 checkpoint, split, seed, optimizer schedule and validation-selected
# test rule. Override STAGES to run a subset.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
GPU="${GPU:-0}"
DETACH="${DETACH:-1}"
PIPELINE_WORKER="${PIPELINE_WORKER:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_FIGURES="${RUN_FIGURES:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
STAGES="${STAGES:-baseline baseline_36 image_only_36 hard_targets occ_l_only occ_u_only full oaac_strong mpd}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/server_logs"
RESULT_DIR="${ROOT}/paper_results"
FIGURE_DIR="${ROOT}/paper_figures/sliceeq_demo"
AUDIT_DIR="${ROOT}/mpd_offline_audit"
PIPELINE_LOG="${LOG_DIR}/coda_final_paper_suite_${RUN_TAG}.log"
EXPECTED_PRETRAIN_SHA256="49e8883039a5712102dc17c5277009504b55c232a10a0af1de4d26fbb414b9b9"
read -r -a STAGE_LIST <<< "${STAGES}"

experiment_name() {
  case "$1" in
    baseline|baseline_36|image_only_36|hard_targets|occ_l_only|occ_u_only|full)
      echo "SliceEqOccIncremental_$1_35_5_10_Pre10000_Self30000_label7_seed1337" ;;
    oaac_strong) echo "SliceEqOccOAACStrong_PROMISE12" ;;
    mpd) echo "SliceEqOccOAACStrongMPD_PROMISE12" ;;
    *) echo "Unsupported stage: $1" >&2; return 2 ;;
  esac
}

stage_label() {
  case "$1" in
    baseline) echo "B0 MT-24" ;;
    baseline_36) echo "C0 ViewMatch-36" ;;
    image_only_36) echo "C1 SRA-Image-36" ;;
    hard_targets) echo "C2 SRA-Hard-36" ;;
    occ_l_only) echo "F10 AFO-L-only" ;;
    occ_u_only) echo "F01 AFO-U-only" ;;
    full) echo "C3 SliceEqOcc" ;;
    oaac_strong) echo "C4 SliceEqOcc+OAAC-S1.25" ;;
    mpd) echo "C5 SliceEqOcc+OAAC-S1.25+MPD" ;;
  esac
}

nonempty_lines() {
  awk 'NF{n++} END{print n+0}' "$1"
}

validate_inputs() {
  [[ ${#STAGE_LIST[@]} -gt 0 ]] || { echo "STAGES is empty" >&2; exit 2; }
  local stage
  for stage in "${STAGE_LIST[@]}"; do experiment_name "${stage}" >/dev/null; done
  for pair in "train.list:35" "val.list:5" "test.list:10" "train_slices.list:940"; do
    local file="${pair%%:*}" expected="${pair##*:}" actual
    [[ -f "${DATA_ROOT}/${file}" ]] || { echo "Missing ${DATA_ROOT}/${file}" >&2; exit 2; }
    actual="$(nonempty_lines "${DATA_ROOT}/${file}")"
    [[ "${actual}" == "${expected}" ]] || {
      echo "Split mismatch: ${file} has ${actual}, expected ${expected}" >&2; exit 2;
    }
  done
  [[ -f "${PRETRAINED_CHECKPOINT}" ]] || {
    echo "Set PRETRAINED_CHECKPOINT to the shared label-7 Pre10000 checkpoint." >&2
    exit 2
  }
  local actual_hash
  actual_hash="$(sha256sum "${PRETRAINED_CHECKPOINT}" | awk '{print $1}')"
  [[ "${actual_hash}" == "${EXPECTED_PRETRAIN_SHA256}" ]] || {
    echo "Pretrain SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_PRETRAIN_SHA256}" >&2
    echo "Actual:   ${actual_hash}" >&2
    exit 2
  }
  case "${DETACH}:${ALLOW_EXISTING}:${SKIP_COMPLETED}:${RUN_FIGURES}:${RUN_AUDIT}:${RUN_TESTS}" in
    *[!01:]*) echo "Boolean controls must be 0 or 1." >&2; exit 2 ;;
  esac
}

run_contract_tests() {
  [[ "${RUN_TESTS}" == "1" ]] || return 0
  echo "[$(date '+%F %T')] Running focused contracts"
  (
    cd "${ROOT}"
    python -m unittest discover -s tests -p 'test_sliceeq_occ_ablation_contract.py' -v
    python -m unittest discover -s tests -p 'test_sliceeq_ablation_summary.py' -v
    python -m unittest discover -s tests -p 'test_sliceeq_oaac_strong_contract.py' -v
    python -m unittest discover -s tests -p 'test_sliceeq_mpd_contract.py' -v
    python -m unittest discover -s tests -p 'test_sliceeq_mpd_audit.py' -v
    python -m unittest discover -s tests -p 'test_sliceeq_portable_entries.py' -v
  )
}

run_figures() {
  [[ "${RUN_FIGURES}" == "1" ]] || return 0
  echo "[$(date '+%F %T')] Rendering paired re-acquisition examples"
  python "${ROOT}/code/visualize_sliceeq_reacquisition.py" \
    --root_path "${DATA_ROOT}" --output_dir "${FIGURE_DIR}"
}

run_audit() {
  [[ "${RUN_AUDIT}" == "1" ]] || return 0
  echo "[$(date '+%F %T')] Running train-only MPD robustness audit"
  python "${ROOT}/code/analyze_sliceeq_mpd_robustness.py" \
    --root_path "${DATA_ROOT}" --output_dir "${AUDIT_DIR}"
}

train_stage() {
  local stage="$1" exp="$2"
  echo "[$(date '+%F %T')] Training $(stage_label "${stage}")"
  case "${stage}" in
    baseline|baseline_36|image_only_36|hard_targets|occ_l_only|occ_u_only|full)
      (
        cd "${ROOT}/code"
        CUDA_VISIBLE_DEVICES="${GPU}" python -u train_sliceeq_occ_ablation.py \
          --root_path "${DATA_ROOT}" \
          --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
          --exp "${exp}" --occ_ablation "${stage}" \
          --max_iterations 30000 --batch_size 24 --labeled_bs 12 \
          --labelnum 7 --seed 1337
      ) ;;
    oaac_strong)
      (
        cd "${ROOT}/code"
        CUDA_VISIBLE_DEVICES="${GPU}" python -u \
          train_sliceeq_occ_oaac_strong_portable.py \
          --root_path "${DATA_ROOT}" \
          --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
          --exp "${exp}" --max_iterations 30000 --batch_size 24 \
          --labeled_bs 12 --labelnum 7 --seed 1337
      ) ;;
    mpd)
      (
        cd "${ROOT}/code"
        CUDA_VISIBLE_DEVICES="${GPU}" python -u \
          train_sliceeq_occ_oaac_strong_mpd_portable.py \
          --root_path "${DATA_ROOT}" \
          --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
          --exp "${exp}" --max_iterations 30000 --batch_size 24 \
          --labeled_bs 12 --labelnum 7 --seed 1337
      ) ;;
  esac
}

test_stage() {
  local stage="$1" exp="$2" checkpoint="$3" log_file="$4" entry
  case "${stage}" in
    oaac_strong) entry="test_sliceeq_occ_oaac_strong.py" ;;
    mpd) entry="test_sliceeq_occ_oaac_strong_mpd.py" ;;
    *) entry="test_sliceeq_occ.py" ;;
  esac
  echo "[$(date '+%F %T')] Testing validation-selected ${checkpoint}"
  (
    cd "${ROOT}/code"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u "${entry}" \
      --root_path "${DATA_ROOT}" --exp "${exp}" \
      --checkpoint_path "${checkpoint}" --labelnum 7 --gpu "${GPU}" \
      --save_result False --nms 0 --auto_find_checkpoint False
  ) 2>&1 | tee "${log_file}"
}

summarize_results() {
  mkdir -p "${RESULT_DIR}"
  (
    cd "${ROOT}/code"
    python summarize_sliceeq_ablation.py \
      --model_root "${ROOT}/model" \
      --output_prefix "${RESULT_DIR}/coda_final_paper_results"
  )
}

run_stages() {
  local stage exp snapshot checkpoint performance stage_log
  for stage in "${STAGE_LIST[@]}"; do
    exp="$(experiment_name "${stage}")"
    snapshot="${ROOT}/model/${exp}_7_labeled/self_train/unet"
    checkpoint="${snapshot}/unet_best_model.pth"
    performance="${snapshot}/performance.txt"
    stage_log="${LOG_DIR}/test_${stage}_${RUN_TAG}.log"

    if [[ "${SKIP_COMPLETED}" == "1" && -s "${performance}" ]]; then
      echo "[$(date '+%F %T')] Skip completed $(stage_label "${stage}"): ${performance}"
      continue
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      if [[ -d "${snapshot}" && "${ALLOW_EXISTING}" != "1" ]]; then
        echo "Incomplete existing stage: ${snapshot}" >&2
        echo "Move it aside or set ALLOW_EXISTING=1 to restart that stage." >&2
        exit 2
      fi
      train_stage "${stage}" "${exp}"
    else
      echo "[$(date '+%F %T')] Reusing existing validation-best checkpoint"
    fi
    [[ -f "${checkpoint}" ]] || {
      echo "No validation-best checkpoint after ${stage}: ${checkpoint}" >&2; exit 1;
    }
    test_stage "${stage}" "${exp}" "${checkpoint}" "${stage_log}"
    [[ -s "${performance}" ]] || {
      echo "Test did not produce ${performance}" >&2; exit 1;
    }
    summarize_results
  done
}

run_pipeline() {
  run_contract_tests
  run_figures
  run_audit
  run_stages
  summarize_results
  echo "[$(date '+%F %T')] CODA Final paper suite completed"
  echo "Results: ${RESULT_DIR}/coda_final_paper_results.md"
  echo "Figures: ${FIGURE_DIR}"
  echo "Audit:   ${AUDIT_DIR}/audit_summary.md"
}

validate_inputs
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
if [[ "${DETACH}" == "1" && "${PIPELINE_WORKER}" != "1" ]]; then
  nohup env DATA_ROOT="${DATA_ROOT}" \
    PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" GPU="${GPU}" \
    DETACH=0 PIPELINE_WORKER=1 ALLOW_EXISTING="${ALLOW_EXISTING}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" RUN_FIGURES="${RUN_FIGURES}" \
    RUN_AUDIT="${RUN_AUDIT}" RUN_TESTS="${RUN_TESTS}" STAGES="${STAGES}" \
    RUN_TAG="${RUN_TAG}" bash "${ROOT}/run_coda_final_paper_suite.sh" \
      >"${PIPELINE_LOG}" 2>&1 </dev/null &
  echo "Started CODA Final paper suite: PID=$!"
  echo "Log: ${PIPELINE_LOG}"
else
  if [[ "${PIPELINE_WORKER}" == "1" ]]; then
    run_pipeline
  else
    run_pipeline 2>&1 | tee "${PIPELINE_LOG}"
  fi
fi
