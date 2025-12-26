#!/usr/bin/env bash
set -euo pipefail

# Runs 1% / 20% / 100% training using prebuilt caches to maximize throughput.
# Intended to be launched inside the `medical_fenlei` conda env.

: "${MF_SPLITS_ROOT:=artifacts/splits_dual_patient}"
: "${MF_MANIFEST_CSV:=artifacts/manifest_ears.csv}"
: "${MF_DUAL_CACHE_DIR:=cache/dual_volumes}"
: "${MF_EAR_CACHE_DIR:=cache/ears_hu}"

: "${MF_SEED:=42}"

: "${MF_DUAL_MODEL:=dual_resnet50_3d}"
: "${MF_DUAL_LABEL_TASK:=six_class}"
: "${MF_DUAL_EPOCHS_1:=10}"
: "${MF_DUAL_EPOCHS_20:=10}"
: "${MF_DUAL_EPOCHS_100:=10}"
: "${MF_DUAL_NUM_WORKERS:=16}"
: "${MF_DUAL_MAX_BATCH_SIZE:=32}"
: "${MF_DUAL_COMPILE:=0}"

: "${MF_EAR_LABEL_TASK:=normal_vs_abnormal}"
: "${MF_EAR_BACKBONE:=resnet50}"
: "${MF_EAR_AGG:=transformer}"
: "${MF_EAR_TR_LAYERS:=2}"
: "${MF_EAR_BATCH_SIZE:=64}"
: "${MF_EAR_NUM_WORKERS:=32}"
: "${MF_EAR_CROP_MODE:=temporal_patch}"
: "${MF_EAR_CROP_LATERAL_BAND_FRAC:=0.6}"
: "${MF_EAR_CROP_LATERAL_BIAS:=0.25}"
: "${MF_EAR_CROP_MIN_AREA:=300}"
: "${MF_EAR_TARGET_SPACING:=0.7}"
: "${MF_EAR_TARGET_Z_SPACING:=0.8}"
: "${MF_EAR_EPOCHS_1:=10}"
: "${MF_EAR_EPOCHS_20:=30}"
: "${MF_EAR_EPOCHS_100:=50}"
: "${MF_EAR_EVAL_BOOT:=500}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs outputs

ts="$(date +%Y%m%d_%H%M%S)"

run_dual() {
  local pct="$1"
  local epochs="$2"
  local out="outputs/dual_${MF_DUAL_MODEL}__${MF_DUAL_LABEL_TASK}_${pct}pct_${ts}"
  local compile_args=()
  if [[ "${MF_DUAL_COMPILE}" == "1" ]]; then
    compile_args+=(--compile)
  fi
  echo "==== [$(date +%F_%T)] dual pct=${pct} epochs=${epochs} out=${out} ===="
  python scripts/train_dual.py \
    --splits-root "${MF_SPLITS_ROOT}" \
    --pct "${pct}" \
    --label-task "${MF_DUAL_LABEL_TASK}" \
    --model "${MF_DUAL_MODEL}" \
    --epochs "${epochs}" \
    --auto-batch \
    --max-batch-size "${MF_DUAL_MAX_BATCH_SIZE}" \
    --num-workers "${MF_DUAL_NUM_WORKERS}" \
    --num-slices 32 \
    --image-size 224 \
    --cache-dir "${MF_DUAL_CACHE_DIR}" \
    --cache-dtype float16 \
    "${compile_args[@]}" \
    --seed "${MF_SEED}" \
    --output-dir "${out}" \
    2>&1 | stdbuf -oL -eL tee -a "logs/train_dual_${MF_DUAL_MODEL}__${MF_DUAL_LABEL_TASK}_${pct}pct_${ts}.log"
}

run_ear() {
  local pct="$1"
  local epochs="$2"
  local out="outputs/ear2d_${MF_EAR_BACKBONE}__${MF_EAR_LABEL_TASK}_${pct}pct_${ts}"
  echo "==== [$(date +%F_%T)] ear2d pct=${pct} epochs=${epochs} out=${out} ===="
  python scripts/train_ear2d.py \
    --splits-root "${MF_SPLITS_ROOT}" \
    --pct "${pct}" \
    --manifest-csv "${MF_MANIFEST_CSV}" \
    --label-task "${MF_EAR_LABEL_TASK}" \
    --backbone "${MF_EAR_BACKBONE}" \
    --aggregator "${MF_EAR_AGG}" \
    --transformer-layers "${MF_EAR_TR_LAYERS}" \
    --epochs "${epochs}" \
    --batch-size "${MF_EAR_BATCH_SIZE}" \
    --num-workers "${MF_EAR_NUM_WORKERS}" \
    --num-slices 32 \
    --image-size 224 \
    --crop-size 192 \
    --crop-mode "${MF_EAR_CROP_MODE}" \
    --crop-lateral-band-frac "${MF_EAR_CROP_LATERAL_BAND_FRAC}" \
    --crop-lateral-bias "${MF_EAR_CROP_LATERAL_BIAS}" \
    --crop-min-area "${MF_EAR_CROP_MIN_AREA}" \
    --sampling even \
    --cache-dir "${MF_EAR_CACHE_DIR}" \
    --target-spacing "${MF_EAR_TARGET_SPACING}" \
    --target-z-spacing "${MF_EAR_TARGET_Z_SPACING}" \
    --early-stop-patience 10 \
    --early-stop-metric acc \
    --seed "${MF_SEED}" \
    --output-dir "${out}" \
    2>&1 | stdbuf -oL -eL tee -a "logs/train_ear2d_${MF_EAR_BACKBONE}__${MF_EAR_LABEL_TASK}_${pct}pct_${ts}.log"

  local ckpt="${out}/checkpoints/best.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "==== [$(date +%F_%T)] eval ear2d pct=${pct} ckpt=${ckpt} ===="
    python scripts/eval_ear2d.py \
      --checkpoint "${ckpt}" \
      --splits-root "${MF_SPLITS_ROOT}" \
      --pct "${pct}" \
      --manifest-csv "${MF_MANIFEST_CSV}" \
      --cache-dir "${MF_EAR_CACHE_DIR}" \
      --num-workers 16 \
      --batch-size "${MF_EAR_BATCH_SIZE}" \
      --topk-slices 3 \
      --n-boot "${MF_EAR_EVAL_BOOT}" \
      --seed "${MF_SEED}" \
      2>&1 | stdbuf -oL -eL tee -a "logs/eval_ear2d_${MF_EAR_BACKBONE}__${MF_EAR_LABEL_TASK}_${pct}pct_${ts}.log"

    if [[ -f "${out}/reports/predictions_val.csv" ]]; then
      python scripts/calibrate_threshold_binary.py \
        --pred-csv "${out}/reports/predictions_val.csv" \
        --objective f1 \
        2>&1 | stdbuf -oL -eL tee -a "logs/calib_thr_ear2d_${MF_EAR_BACKBONE}__${MF_EAR_LABEL_TASK}_${pct}pct_${ts}.log"
    fi
  else
    echo "[warn] missing checkpoint: ${ckpt}"
  fi
}

run_dual 1 "${MF_DUAL_EPOCHS_1}"
run_dual 20 "${MF_DUAL_EPOCHS_20}"
run_dual 100 "${MF_DUAL_EPOCHS_100}"

run_ear 1 "${MF_EAR_EPOCHS_1}"
run_ear 20 "${MF_EAR_EPOCHS_20}"
run_ear 100 "${MF_EAR_EPOCHS_100}"

# Optional summaries (fast, CPU-only).
python scripts/summarize_experiments_dual.py --outputs-dir outputs 2>/dev/null || true
python scripts/summarize_experiments_ear2d.py --runs-root outputs 2>/dev/null || true

echo "==== [$(date +%F_%T)] all done ===="
