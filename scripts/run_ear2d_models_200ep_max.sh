#!/usr/bin/env bash
set -euo pipefail

# Run long (200-epoch) Ear2D experiments with different backbones:
#   - resnet: SliceAttentionResNet (torchvision)
#   - vit:    SliceAttentionViT (lightweight ViT encoder per slice)
#   - unet:   SliceAttentionUNet (MONAI UNet per slice, pooled for classification)
#
# Usage (inside conda env):
#   bash scripts/run_ear2d_models_200ep_max.sh
#
# Customize:
#   MF_PCTS="20" MF_MODELS="resnet,vit,unet" MF_EPOCHS=200 bash scripts/run_ear2d_models_200ep_max.sh

: "${MF_SPLITS_ROOT:=artifacts/splits_dual_patient_clustered_v1}"
: "${MF_MANIFEST_CSV:=artifacts/manifest_ears.csv}"
: "${MF_DICOM_BASE:=/home/ubuntu/tim/medical_data_2}"

: "${MF_PCTS:=20}"
# Optional: comma-separated list of tasks to run (overrides MF_LABEL_TASK).
# Example:
#   MF_LABEL_TASKS="normal_vs_abnormal,normal_vs_csoma,normal_vs_cholesteatoma,normal_vs_cholesterol_granuloma,normal_vs_ome,cholesteatoma_vs_other_abnormal"
: "${MF_LABEL_TASK:=normal_vs_abnormal}"
: "${MF_LABEL_TASKS:=}"

: "${MF_MODELS:=resnet,vit,unet}"

: "${MF_NUM_SLICES:=32}"
: "${MF_IMAGE_SIZE:=224}"
: "${MF_CROP_SIZE:=192}"
: "${MF_CROP_MODE:=temporal_patch}"
: "${MF_CROP_LATERAL_BAND_FRAC:=0.6}"
: "${MF_CROP_LATERAL_BIAS:=0.25}"
: "${MF_CROP_MIN_AREA:=300}"
: "${MF_SAMPLING:=even}"
: "${MF_BLOCK_LEN:=64}"
: "${MF_TARGET_SPACING:=0.7}"
: "${MF_TARGET_Z_SPACING:=0.8}"
: "${MF_CACHE_DIR:=cache/ears_hu}"

: "${MF_EPOCHS:=200}"
: "${MF_LR:=3e-4}"
: "${MF_LR_RESNET:=${MF_LR}}"
: "${MF_LR_VIT:=${MF_LR}}"
: "${MF_LR_UNET:=${MF_LR}}"
: "${MF_WEIGHT_DECAY:=1e-4}"
: "${MF_LABEL_SMOOTHING:=0.05}"

: "${MF_NUM_WORKERS:=32}"
: "${MF_BATCH_SIZE_RESNET:=64}"
: "${MF_BATCH_SIZE_VIT:=16}"
: "${MF_BATCH_SIZE_UNET:=16}"

: "${MF_AGGREGATOR:=transformer}"
: "${MF_ATTN_HIDDEN:=128}"
: "${MF_DROPOUT:=0.2}"
: "${MF_TRANSFORMER_LAYERS:=2}"
: "${MF_TRANSFORMER_HEADS:=8}"
: "${MF_TRANSFORMER_FF_DIM:=0}"
: "${MF_TRANSFORMER_DROPOUT:=0.1}"
: "${MF_TRANSFORMER_MAX_LEN:=256}"

: "${MF_RESNET_BACKBONE:=resnet50}"

: "${MF_VIT_PATCH_SIZE:=16}"
: "${MF_VIT_HIDDEN_SIZE:=512}"
: "${MF_VIT_MLP_DIM:=2048}"
: "${MF_VIT_NUM_LAYERS:=8}"
: "${MF_VIT_NUM_HEADS:=8}"
: "${MF_VIT_DROPOUT:=0.1}"

: "${MF_UNET_EMBED_DIM:=128}"
: "${MF_UNET_CHANNELS:=16,32,64,128,256}"
: "${MF_UNET_STRIDES:=2,2,2,2}"
: "${MF_UNET_NUM_RES_UNITS:=2}"

: "${MF_AUGMENT:=1}"
: "${MF_AMP:=1}"
: "${MF_TF32:=1}"
: "${MF_CUDNN_BENCHMARK:=1}"

# Augmentation knobs (passed to scripts/train_ear2d.py).
: "${MF_AUG_FLIP_PROB:=0.2}"
: "${MF_AUG_INTENSITY_PROB:=0.6}"
: "${MF_AUG_NOISE_PROB:=0.2}"
: "${MF_AUG_GAMMA_PROB:=0.2}"
: "${MF_AUG_AFFINE_PROB:=0.0}"
: "${MF_AUG_AFFINE_DEGREES:=12.0}"
: "${MF_AUG_AFFINE_TRANSLATE:=0.08}"
: "${MF_AUG_AFFINE_SCALE_LOW:=0.85}"
: "${MF_AUG_AFFINE_SCALE_HIGH:=1.15}"

# Optional W&B logging (do NOT hardcode keys; use env: WANDB_API_KEY).
: "${MF_WANDB:=0}"
: "${MF_WANDB_PROJECT:=medical_fenlei}"
: "${MF_WANDB_ENTITY:=}"
: "${MF_WANDB_GROUP:=}"
: "${MF_WANDB_TAGS:=ear2d}"
: "${MF_WANDB_MODE:=online}"
: "${MF_WANDB_DIR:=wandb}"

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

batch_for_model() {
  local model="${1}"
  case "${model}" in
    resnet) echo "${MF_BATCH_SIZE_RESNET}" ;;
    vit) echo "${MF_BATCH_SIZE_VIT}" ;;
    unet) echo "${MF_BATCH_SIZE_UNET}" ;;
    *) echo "${MF_BATCH_SIZE_RESNET}" ;;
  esac
}

lr_for_model() {
  local model="${1}"
  case "${model}" in
    resnet) echo "${MF_LR_RESNET}" ;;
    vit) echo "${MF_LR_VIT}" ;;
    unet) echo "${MF_LR_UNET}" ;;
    *) echo "${MF_LR}" ;;
  esac
}

run_one() {
  local pct="$1"
  local model="$2"
  local task="$3"
  local bs
  bs="$(batch_for_model "${model}")"
  local lr_model
  lr_model="$(lr_for_model "${model}")"

  local out="outputs/ear2d_${model}__${task}_${pct}pct_e${MF_EPOCHS}_${ts}"
  echo "==== [$(date +%F_%T)] START ear2d model=${model} task=${task} pct=${pct} epochs=${MF_EPOCHS} bs=${bs} lr=${lr_model} out=${out} ===="

  local amp_flag="--no-amp"
  if [[ "$(to_bool_flag "${MF_AMP}")" == "true" ]]; then amp_flag="--amp"; fi

  local augment_flag="--no-augment"
  if [[ "$(to_bool_flag "${MF_AUGMENT}")" == "true" ]]; then augment_flag="--augment"; fi

  local tf32_flag="--no-tf32"
  if [[ "$(to_bool_flag "${MF_TF32}")" == "true" ]]; then tf32_flag="--tf32"; fi

  local cudnn_flag="--no-cudnn-benchmark"
  if [[ "$(to_bool_flag "${MF_CUDNN_BENCHMARK}")" == "true" ]]; then cudnn_flag="--cudnn-benchmark"; fi

  local wandb_args=()
  if [[ "$(to_bool_flag "${MF_WANDB}")" == "true" ]]; then
    wandb_args+=(--wandb --wandb-project "${MF_WANDB_PROJECT}" --wandb-mode "${MF_WANDB_MODE}" --wandb-dir "${MF_WANDB_DIR}")
    if [[ -n "${MF_WANDB_ENTITY}" ]]; then wandb_args+=(--wandb-entity "${MF_WANDB_ENTITY}"); fi
    if [[ -n "${MF_WANDB_GROUP}" ]]; then wandb_args+=(--wandb-group "${MF_WANDB_GROUP}"); fi
    if [[ -n "${MF_WANDB_TAGS}" ]]; then wandb_args+=(--wandb-tags "${MF_WANDB_TAGS}"); fi
  fi

  set +e
  /home/ubuntu/miniconda3/envs/medical_fenlei/bin/python scripts/train_ear2d.py \
    --splits-root "${MF_SPLITS_ROOT}" \
    --pct "${pct}" \
    --manifest-csv "${MF_MANIFEST_CSV}" \
    --dicom-base "${MF_DICOM_BASE}" \
    --label-task "${task}" \
    --model "${model}" \
    --backbone "${MF_RESNET_BACKBONE}" \
    --vit-patch-size "${MF_VIT_PATCH_SIZE}" \
    --vit-hidden-size "${MF_VIT_HIDDEN_SIZE}" \
    --vit-mlp-dim "${MF_VIT_MLP_DIM}" \
    --vit-num-layers "${MF_VIT_NUM_LAYERS}" \
    --vit-num-heads "${MF_VIT_NUM_HEADS}" \
    --vit-dropout "${MF_VIT_DROPOUT}" \
    --unet-embed-dim "${MF_UNET_EMBED_DIM}" \
    --unet-channels "${MF_UNET_CHANNELS}" \
    --unet-strides "${MF_UNET_STRIDES}" \
    --unet-num-res-units "${MF_UNET_NUM_RES_UNITS}" \
    --aggregator "${MF_AGGREGATOR}" \
    --attn-hidden "${MF_ATTN_HIDDEN}" \
    --dropout "${MF_DROPOUT}" \
    --transformer-layers "${MF_TRANSFORMER_LAYERS}" \
    --transformer-heads "${MF_TRANSFORMER_HEADS}" \
    --transformer-ff-dim "${MF_TRANSFORMER_FF_DIM}" \
    --transformer-dropout "${MF_TRANSFORMER_DROPOUT}" \
    --transformer-max-len "${MF_TRANSFORMER_MAX_LEN}" \
    --epochs "${MF_EPOCHS}" \
    --batch-size "${bs}" \
    --num-workers "${MF_NUM_WORKERS}" \
    --num-slices "${MF_NUM_SLICES}" \
    --image-size "${MF_IMAGE_SIZE}" \
    --crop-size "${MF_CROP_SIZE}" \
    --crop-mode "${MF_CROP_MODE}" \
    --crop-lateral-band-frac "${MF_CROP_LATERAL_BAND_FRAC}" \
    --crop-lateral-bias "${MF_CROP_LATERAL_BIAS}" \
    --crop-min-area "${MF_CROP_MIN_AREA}" \
    --sampling "${MF_SAMPLING}" \
    --block-len "${MF_BLOCK_LEN}" \
    --target-spacing "${MF_TARGET_SPACING}" \
    --target-z-spacing "${MF_TARGET_Z_SPACING}" \
    --cache \
    --cache-dir "${MF_CACHE_DIR}" \
    --lr "${lr_model}" \
    --weight-decay "${MF_WEIGHT_DECAY}" \
    --label-smoothing "${MF_LABEL_SMOOTHING}" \
    ${amp_flag} \
    ${tf32_flag} \
    ${cudnn_flag} \
    ${augment_flag} \
    --aug-flip-prob "${MF_AUG_FLIP_PROB}" \
    --aug-intensity-prob "${MF_AUG_INTENSITY_PROB}" \
    --aug-noise-prob "${MF_AUG_NOISE_PROB}" \
    --aug-gamma-prob "${MF_AUG_GAMMA_PROB}" \
    --aug-affine-prob "${MF_AUG_AFFINE_PROB}" \
    --aug-affine-degrees "${MF_AUG_AFFINE_DEGREES}" \
    --aug-affine-translate "${MF_AUG_AFFINE_TRANSLATE}" \
    --aug-affine-scale-low "${MF_AUG_AFFINE_SCALE_LOW}" \
    --aug-affine-scale-high "${MF_AUG_AFFINE_SCALE_HIGH}" \
    --early-stop-metric acc \
    --early-stop-patience 0 \
    "${wandb_args[@]}" \
    --output-dir "${out}" \
    2>&1 | stdbuf -oL -eL tee -a "logs/train_ear2d_${model}__${task}_${pct}pct_e${MF_EPOCHS}_${ts}.log"
  local rc=$?
  set -e

  if [[ "${rc}" -eq 2 ]]; then
    echo "==== [$(date +%F_%T)] SKIP ear2d model=${model} task=${task} pct=${pct} (no data after filtering; exit=2) ===="
    return 0
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "==== [$(date +%F_%T)] FAIL ear2d model=${model} task=${task} pct=${pct} exit=${rc} ===="
    return "${rc}"
  fi

  # Optional: eval best checkpoint (fast-ish, but includes bootstrap by exam_id).
  local ckpt="${out}/checkpoints/best.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "==== [$(date +%F_%T)] eval ear2d model=${model} task=${task} pct=${pct} ckpt=${ckpt} ===="
    /home/ubuntu/miniconda3/envs/medical_fenlei/bin/python scripts/eval_ear2d.py \
      --checkpoint "${ckpt}" \
      --splits-root "${MF_SPLITS_ROOT}" \
      --pct "${pct}" \
      --manifest-csv "${MF_MANIFEST_CSV}" \
      --dicom-base "${MF_DICOM_BASE}" \
      --cache-dir "${MF_CACHE_DIR}" \
      --num-workers 16 \
      --batch-size "${bs}" \
      --topk-slices 3 \
      --n-boot 500 \
      --seed 42 \
      2>&1 | stdbuf -oL -eL tee -a "logs/eval_ear2d_${model}__${task}_${pct}pct_e${MF_EPOCHS}_${ts}.log"
  fi

  echo "==== [$(date +%F_%T)] DONE ear2d model=${model} task=${task} pct=${pct} out=${out} ===="
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

# Summarize all ear2d runs (fast).
/home/ubuntu/miniconda3/envs/medical_fenlei/bin/python scripts/summarize_experiments_ear2d.py --runs-root outputs 2>/dev/null || true

echo "==== [$(date +%F_%T)] all done ===="
