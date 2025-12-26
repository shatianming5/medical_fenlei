#!/usr/bin/env bash
set -euo pipefail

# Run long (200-epoch) Dual 3D experiments with different backbones:
#   - pure ResNet3D (MONAI ResNet)
#   - ViT3D (MONAI ViT)
#   - UNet3D (MONAI UNet, pooled for classification)
#
# Designed to maximize throughput on a single GPU using cached volumes and auto-batch.
#
# Usage (inside conda env):
#   bash scripts/run_dual_models_200ep_max.sh
#
# Customize:
#   MF_PCTS="100" \
#   MF_MODELS="dual_resnet200_3d,dual_vit_3d,dual_unet_3d" \
#   MF_EPOCHS=200 \
#   bash scripts/run_dual_models_200ep_max.sh

: "${MF_SPLITS_ROOT:=artifacts/splits_dual_patient_clustered_v1}"
: "${MF_DICOM_BASE:=/home/ubuntu/tim/medical_data_2}"
: "${MF_PCTS:=100}"
: "${MF_LABEL_TASK:=six_class}"
# Optional: comma-separated list of tasks to run (overrides MF_LABEL_TASK).
# Example:
#   MF_LABEL_TASKS="six_class,normal_vs_diseased,normal_vs_csoma,normal_vs_cholesteatoma,normal_vs_cholesterol_granuloma,normal_vs_ome"
# Notes:
#   - Tasks are defined in `src/medical_fenlei/tasks.py`.
#   - Some tasks filter the dataset; if no samples remain, training will exit with code 2.
: "${MF_LABEL_TASKS:=}"

: "${MF_MODELS:=dual_resnet200_3d,dual_vit_3d,dual_unet_3d}"

: "${MF_NUM_SLICES:=32}"
: "${MF_IMAGE_SIZE:=224}"
: "${MF_CROP_SIZE:=192}"
: "${MF_WINDOW_WL:=700}"
: "${MF_WINDOW_WW:=4000}"
: "${MF_WINDOW2_WL:=0}"
: "${MF_WINDOW2_WW:=0}"
: "${MF_PAIR_FEATURES:=none}"
: "${MF_SAMPLING:=air_block}"
: "${MF_BLOCK_LEN:=64}"
: "${MF_TARGET_SPACING:=0.7}"
: "${MF_TARGET_Z_SPACING:=0.8}"
: "${MF_NUM_WORKERS:=32}"
: "${MF_MAX_BATCH_SIZE:=32}"
: "${MF_AUTO_BATCH:=1}"
: "${MF_BATCH_SIZE:=1}"

: "${MF_EPOCHS:=200}"
: "${MF_LR:=3e-4}"
: "${MF_LR_SCHEDULE:=cosine}"
: "${MF_MIN_LR:=1e-6}"
: "${MF_WARMUP_EPOCHS:=5}"
: "${MF_WARMUP_RATIO:=0.1}"
: "${MF_GRAD_CLIP_NORM:=1.0}"
: "${MF_GRAD_ACCUM_STEPS:=1}"
: "${MF_EVAL_EVERY:=1}"
: "${MF_WEIGHT_DECAY:=0.05}"
: "${MF_LABEL_SMOOTHING:=0.10}"
: "${MF_CLASS_WEIGHTS:=1}"
: "${MF_BALANCED_SAMPLER:=0}"
: "${MF_BALANCED_SAMPLER_MODE:=max}"
: "${MF_LOSS:=ce}"
: "${MF_FOCAL_GAMMA:=2.0}"
: "${MF_INIT_BIAS:=1}"

: "${MF_AUGMENT:=1}"
: "${MF_AMP:=1}"
: "${MF_TF32:=1}"
: "${MF_CUDNN_BENCHMARK:=1}"
: "${MF_EMPTY_CACHE:=1}"

# torch.compile needs Triton on this machine; leave disabled unless you know it's installed.
: "${MF_COMPILE:=0}"

# Optional W&B logging (do NOT hardcode keys; use env: WANDB_API_KEY).
: "${MF_WANDB:=0}"
: "${MF_WANDB_PROJECT:=medical_fenlei}"
: "${MF_WANDB_ENTITY:=}"
: "${MF_WANDB_GROUP:=}"
: "${MF_WANDB_TAGS:=dual3d}"
: "${MF_WANDB_MODE:=online}"
: "${MF_WANDB_DIR:=wandb}"

# ViT config (applies only to dual_vit_3d).
# NOTE: default patch_size (4,16,16) can be extremely heavy at (32,224,224).
: "${MF_VIT_PATCH_SIZE:=8,32,32}"
: "${MF_VIT_POOL:=mean}"
: "${MF_VIT_HIDDEN_SIZE:=512}"
: "${MF_VIT_MLP_DIM:=2048}"
: "${MF_VIT_NUM_LAYERS:=8}"
: "${MF_VIT_NUM_HEADS:=8}"

# ViT training overrides (helps avoid \"always-one-class\" collapse on this dataset).
# You can override per-run by exporting these env vars.
: "${MF_VIT_CLASS_WEIGHTS:=0}"
: "${MF_VIT_BALANCED_SAMPLER:=1}"
: "${MF_VIT_LABEL_SMOOTHING:=0.0}"
: "${MF_VIT_WEIGHT_DECAY:=0.02}"

# UNet config (applies only to dual_unet_3d).
: "${MF_UNET_CHANNELS:=16,32,64,128,256}"
: "${MF_UNET_STRIDES:=2,2,2,2}"
: "${MF_UNET_NUM_RES_UNITS:=2}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs outputs
ts="$(date +%Y%m%d_%H%M%S)"

to_bool_flag() {
  local v="${1}"
  if [[ "${v}" == "1" || "${v,,}" == "true" ]]; then
    echo "true"
  else
    echo "false"
  fi
}

run_one() {
  local pct="$1"
  local model="$2"
  local label_task="$3"

  local out="outputs/dual_${model}__${label_task}_${pct}pct_e${MF_EPOCHS}_${ts}"
  echo "==== [$(date +%F_%T)] START dual model=${model} task=${label_task} pct=${pct} epochs=${MF_EPOCHS} out=${out} ===="

  local weight_decay="${MF_WEIGHT_DECAY}"
  local label_smoothing="${MF_LABEL_SMOOTHING}"
  local class_weights="${MF_CLASS_WEIGHTS}"
  local balanced_sampler="${MF_BALANCED_SAMPLER}"
  if [[ "${model}" == "dual_vit_3d" ]]; then
    weight_decay="${MF_VIT_WEIGHT_DECAY}"
    label_smoothing="${MF_VIT_LABEL_SMOOTHING}"
    class_weights="${MF_VIT_CLASS_WEIGHTS}"
    balanced_sampler="${MF_VIT_BALANCED_SAMPLER}"
  fi

  local amp_flag="--no-amp"
  if [[ "$(to_bool_flag "${MF_AMP}")" == "true" ]]; then amp_flag="--amp"; fi

  local augment_flag="--no-augment"
  if [[ "$(to_bool_flag "${MF_AUGMENT}")" == "true" ]]; then augment_flag="--augment"; fi

  local tf32_flag="--no-tf32"
  if [[ "$(to_bool_flag "${MF_TF32}")" == "true" ]]; then tf32_flag="--tf32"; fi

  local cudnn_flag="--no-cudnn-benchmark"
  if [[ "$(to_bool_flag "${MF_CUDNN_BENCHMARK}")" == "true" ]]; then cudnn_flag="--cudnn-benchmark"; fi

  local empty_cache_flag="--no-empty-cache"
  if [[ "$(to_bool_flag "${MF_EMPTY_CACHE}")" == "true" ]]; then empty_cache_flag="--empty-cache"; fi

  local wandb_args=()
  if [[ "$(to_bool_flag "${MF_WANDB}")" == "true" ]]; then
    wandb_args+=(--wandb --wandb-project "${MF_WANDB_PROJECT}" --wandb-mode "${MF_WANDB_MODE}" --wandb-dir "${MF_WANDB_DIR}")
    if [[ -n "${MF_WANDB_ENTITY}" ]]; then wandb_args+=(--wandb-entity "${MF_WANDB_ENTITY}"); fi
    if [[ -n "${MF_WANDB_GROUP}" ]]; then wandb_args+=(--wandb-group "${MF_WANDB_GROUP}"); fi
    if [[ -n "${MF_WANDB_TAGS}" ]]; then wandb_args+=(--wandb-tags "${MF_WANDB_TAGS}"); fi
  fi

  local compile_args=()
  if [[ "$(to_bool_flag "${MF_COMPILE}")" == "true" ]]; then
    compile_args+=(--compile)
  fi

  local batch_args=()
  if [[ "$(to_bool_flag "${MF_AUTO_BATCH}")" == "true" ]]; then
    batch_args+=(--auto-batch --max-batch-size "${MF_MAX_BATCH_SIZE}")
  else
    batch_args+=(--batch-size "${MF_BATCH_SIZE}")
  fi

  set +e
  /home/ubuntu/miniconda3/envs/medical_fenlei/bin/python scripts/train_dual.py \
    --splits-root "${MF_SPLITS_ROOT}" \
    --pct "${pct}" \
    --dicom-base "${MF_DICOM_BASE}" \
    --model "${model}" \
    --label-task "${label_task}" \
    --epochs "${MF_EPOCHS}" \
    "${batch_args[@]}" \
    --num-workers "${MF_NUM_WORKERS}" \
    --num-slices "${MF_NUM_SLICES}" \
    --image-size "${MF_IMAGE_SIZE}" \
    --crop-size "${MF_CROP_SIZE}" \
    --window-wl "${MF_WINDOW_WL}" \
    --window-ww "${MF_WINDOW_WW}" \
    --window2-wl "${MF_WINDOW2_WL}" \
    --window2-ww "${MF_WINDOW2_WW}" \
    --pair-features "${MF_PAIR_FEATURES}" \
    --sampling "${MF_SAMPLING}" \
    --block-len "${MF_BLOCK_LEN}" \
    --target-spacing "${MF_TARGET_SPACING}" \
    --target-z-spacing "${MF_TARGET_Z_SPACING}" \
    --cache \
    --cache-dir cache/dual_volumes \
    --cache-dtype float16 \
    --vit-patch-size "${MF_VIT_PATCH_SIZE}" \
    --vit-pool "${MF_VIT_POOL}" \
    --vit-hidden-size "${MF_VIT_HIDDEN_SIZE}" \
    --vit-mlp-dim "${MF_VIT_MLP_DIM}" \
    --vit-num-layers "${MF_VIT_NUM_LAYERS}" \
    --vit-num-heads "${MF_VIT_NUM_HEADS}" \
    --unet-channels "${MF_UNET_CHANNELS}" \
    --unet-strides "${MF_UNET_STRIDES}" \
    --unet-num-res-units "${MF_UNET_NUM_RES_UNITS}" \
    --lr "${MF_LR}" \
    --lr-schedule "${MF_LR_SCHEDULE}" \
    --min-lr "${MF_MIN_LR}" \
    --warmup-epochs "${MF_WARMUP_EPOCHS}" \
    --warmup-ratio "${MF_WARMUP_RATIO}" \
    --grad-clip-norm "${MF_GRAD_CLIP_NORM}" \
    --grad-accum-steps "${MF_GRAD_ACCUM_STEPS}" \
    --weight-decay "${weight_decay}" \
    --label-smoothing "${label_smoothing}" \
    --loss "${MF_LOSS}" \
    --focal-gamma "${MF_FOCAL_GAMMA}" \
    ${amp_flag} \
    ${augment_flag} \
    ${tf32_flag} \
    ${cudnn_flag} \
    ${empty_cache_flag} \
    "${compile_args[@]}" \
    --eval-every "${MF_EVAL_EVERY}" \
    "${wandb_args[@]}" \
    $( [[ "$(to_bool_flag "${class_weights}")" == "true" ]] && echo "--class-weights" || echo "--no-class-weights" ) \
    $( [[ "$(to_bool_flag "${balanced_sampler}")" == "true" ]] && echo "--balanced-sampler" || echo "--no-balanced-sampler" ) \
    --balanced-sampler-mode "${MF_BALANCED_SAMPLER_MODE}" \
    $( [[ "$(to_bool_flag "${MF_INIT_BIAS}")" == "true" ]] && echo "--init-bias" || echo "--no-init-bias" ) \
    --early-stop-patience 0 \
    --early-stop-metric val_acc \
    --output-dir "${out}" \
    2>&1 | stdbuf -oL -eL tee -a "logs/train_dual_${model}__${label_task}_${pct}pct_e${MF_EPOCHS}_${ts}.log"
  local rc=$?
  set -e
  if [[ "${rc}" -eq 2 ]]; then
    echo "==== [$(date +%F_%T)] SKIP dual model=${model} task=${label_task} pct=${pct} (no data after filtering; exit=2) ===="
    return 0
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "==== [$(date +%F_%T)] FAIL dual model=${model} task=${label_task} pct=${pct} exit=${rc} ===="
    return "${rc}"
  fi

  echo "==== [$(date +%F_%T)] DONE dual model=${model} task=${label_task} pct=${pct} out=${out} ===="
}

IFS=',' read -r -a pcts <<< "${MF_PCTS}"
IFS=',' read -r -a models <<< "${MF_MODELS}"
if [[ -n "${MF_LABEL_TASKS}" ]]; then
  IFS=',' read -r -a tasks <<< "${MF_LABEL_TASKS}"
else
  tasks=("${MF_LABEL_TASK}")
fi

for pct in "${pcts[@]}"; do
  pct="$(echo "${pct}" | xargs)"
  [[ -z "${pct}" ]] && continue
  for model in "${models[@]}"; do
    model="$(echo "${model}" | xargs)"
    [[ -z "${model}" ]] && continue
    for task in "${tasks[@]}"; do
      task="$(echo "${task}" | xargs)"
      [[ -z "${task}" ]] && continue
      run_one "${pct}" "${model}" "${task}"
    done
  done
done

# Summarize all dual runs (fast).
/home/ubuntu/miniconda3/envs/medical_fenlei/bin/python scripts/summarize_experiments_dual.py --outputs-dir outputs 2>/dev/null || true

echo "==== [$(date +%F_%T)] all done ===="
