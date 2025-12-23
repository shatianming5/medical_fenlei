#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CORES="$(nproc)"
JOBS="${JOBS:-$CORES}"
THREADS="${THREADS:-$CORES}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/extract_all.log}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE" >/dev/null
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing command: $1" >&2; exit 1; }
}

outer_zip="${1:-$PROJECT_ROOT/archives/medical_data_2/medical_data_2.zip}"
splits_dir="${SPLITS_DIR:-$PROJECT_ROOT/archives/medical_data_2/splits}"
data_root="${DATA_ROOT:-$PROJECT_ROOT/data/medical_data_2}"

require_cmd find

mkdir -p "$(dirname "$LOG_FILE")" "$splits_dir" "$data_root"

log "CPU cores=$CORES jobs=$JOBS threads=$THREADS"
log "outer_zip=$outer_zip"
log "splits_dir=$splits_dir"
log "data_root=$data_root"

if [[ ! -f "$outer_zip" ]]; then
  echo "not found: $outer_zip" >&2
  exit 1
fi

extract_outer_zip() {
  require_cmd 7z
  if find "$splits_dir" -maxdepth 1 -type f -name '*.zip.001' -print -quit | grep -q .; then
    log "stage1: splits already exist in $splits_dir; skipping"
    return 0
  fi

  log "stage1: extracting $outer_zip -> $splits_dir"
  # -spe drops the single top-level folder inside the zip (so files land directly in splits_dir/).
  7z x -y -aos "-mmt=$THREADS" -bb0 -bsp0 -scsUTF-8 -spd -spe "-o$splits_dir" "$outer_zip" >>"$LOG_FILE" 2>&1
  log "stage1: done"
}

extract_zip_volumes() {
  require_cmd 7z
  require_cmd split
  require_cmd xargs
  require_cmd awk

  log "stage2: finding split zip first volumes (*.zip.001)"
  mapfile -d '' volumes < <(find "$splits_dir" -type f -name '*.zip.001' -print0)

  if (( ${#volumes[@]} == 0 )); then
    log "stage2: no *.zip.001 found; skipping"
    return 0
  fi

  log "stage2: found ${#volumes[@]} volume(s)"

  local volume
  for volume in "${volumes[@]}"; do
    parent="$(dirname "$volume")"
    name="$(basename "$volume")"
    base1="${name%.001}"
    base2="${base1%.zip}"
    outdir="$data_root/$base2"

    if [[ -d "$outdir" ]] && find "$outdir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      log "stage2: skip (already extracted): $volume -> $outdir"
      continue
    fi

    mkdir -p "$outdir"
    log "stage2: listing paths in $volume (for parallel extraction)"

    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' RETURN

    # Extract "Path = ..." records, skipping the first archive-properties "Path = <archive>" line.
    7z l -slt -ba -bd "$volume" \
      | awk '
          $0 ~ /^Path = / {
            p = substr($0, 8);
            if (++n > 1) print p;
          }
        ' >"$tmpdir/paths.txt"

    path_count="$(wc -l <"$tmpdir/paths.txt" | tr -d ' ')"
    if [[ "$path_count" == "0" ]]; then
      log "stage2: no paths found in $volume; skipping"
      trap - RETURN
      rm -rf "$tmpdir"
      continue
    fi

    workers="$JOBS"
    if (( workers > path_count )); then
      workers="$path_count"
    fi
    if (( workers < 1 )); then
      workers=1
    fi

    log "stage2: paths=$path_count workers=$workers outdir=$outdir"

    split -n "l/$workers" -d --additional-suffix=.lst "$tmpdir/paths.txt" "$tmpdir/part_"

    export LOG_FILE outdir volume
    find "$tmpdir" -maxdepth 1 -type f -name 'part_*.lst' -print0 | xargs -0 -n 1 -P "$workers" bash -lc '
      set -euo pipefail
      part="$1"
      # -spd disables wildcard matching so list entries are treated literally.
      7z x -y -aos -mmt=1 -bb0 -bsp0 -scsUTF-8 -spd "-o$outdir" "-i@$part" "$volume" >>"$LOG_FILE" 2>&1
    ' bash

    trap - RETURN
    rm -rf "$tmpdir"
    log "stage2: done: $volume"
  done

  log "stage2: done"
}

extract_outer_zip
extract_zip_volumes

log "all done"
