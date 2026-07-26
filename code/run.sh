#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

CLIP_ARCH="${CLIP_ARCH:-/home/zqhe/HDD/datasets/ViT-L-14.pt}"
DATASET_ROOT="${DATASET_ROOT:-/home/zqhe/HDD/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zqhe/HDD/FlowComposer}"

MIT_DATASET_PATH="${MIT_DATASET_PATH:-${DATASET_ROOT}/mit-states}"
UT_DATASET_PATH="${UT_DATASET_PATH:-${DATASET_ROOT}/data/ut-zappos}"
CGQA_DATASET_PATH="${CGQA_DATASET_PATH:-${DATASET_ROOT}/data/cgqa}"

MIT_CHECKPOINT_EPOCH="${MIT_CHECKPOINT_EPOCH:-9}"
UT_CHECKPOINT_EPOCH="${UT_CHECKPOINT_EPOCH:-5}"
GATE_LR="1e-7"

run_s1() {
  local dataset="$1"
  local dataset_path="$2"
  local epoch_pt_end="$3"
  local epoch_fm_start="$4"
  local save_path="${OUTPUT_ROOT}/${dataset}"
  local yml_path="${SCRIPT_DIR}/config/troika/${dataset}.yml"

  echo "===== Stage 1: ${dataset} ====="
  mkdir -p "${save_path}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -u "${SCRIPT_DIR}/s1_flow.py" \
    --clip_arch "${CLIP_ARCH}" \
    --dataset_path "${dataset_path}" \
    --save_path "${save_path}" \
    --yml_path "${yml_path}" \
    --num_workers 10 \
    --save_final_model \
    --seed 0 \
    --velocity_loss_weight 1 \
    --logit_weight 1 \
    --path_2 \
    --vis_reg_weight 0 \
    --text_reg_weight 0 \
    --epoch_pt_end "${epoch_pt_end}" \
    --epoch_fm_start "${epoch_fm_start}" \
    --leak_augmentation \
    --vel_consistency_weight 0 \
    --model_name troika_cfm
}

run_s2() {
  local dataset="$1"
  local dataset_path="$2"
  local checkpoint_epoch="$3"
  local epoch_pt_end="$4"
  local epoch_fm_start="$5"
  local save_path="${OUTPUT_ROOT}/${dataset}"
  local yml_path="${SCRIPT_DIR}/config/troika/${dataset}.yml"

  echo "===== Stage 2: ${dataset} (gate lr=${GATE_LR}) ====="
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -u "${SCRIPT_DIR}/s2_gate.py" \
    --clip_arch "${CLIP_ARCH}" \
    --dataset_path "${dataset_path}" \
    --save_path "${save_path}" \
    --yml_path "${yml_path}" \
    --num_workers 10 \
    --save_final_model \
    --seed 0 \
    --velocity_loss_weight 1 \
    --logit_weight 1 \
    --path_2 \
    --vis_reg_weight 0 \
    --text_reg_weight 0 \
    --epoch_pt_end "${epoch_pt_end}" \
    --epoch_fm_start "${epoch_fm_start}" \
    --vel_consistency_weight 0 \
    --lr_fm 2e-4 \
    --save_model_path "${save_path}/troika_${checkpoint_epoch}.pt" \
    --save_cfm1_path "${save_path}/cfm_attribute_${checkpoint_epoch}.pt" \
    --save_cfm2_path "${save_path}/cfm_obj_${checkpoint_epoch}.pt" \
    --lr_gate "${GATE_LR}" \
    --model_name troika_cfm
}

# Finish Stage 1 for all datasets before starting any gate training.
# run_s1 "mit-states" "${MIT_DATASET_PATH}" 4 0
# run_s1 "ut-zappos" "${UT_DATASET_PATH}" 1 0
# run_s1 "cgqa" "${CGQA_DATASET_PATH}" 2 0

# run_s2 "mit-states" "${MIT_DATASET_PATH}" "${MIT_CHECKPOINT_EPOCH}" 4 0
# run_s2 "ut-zappos" "${UT_DATASET_PATH}" "${UT_CHECKPOINT_EPOCH}" 1 0
