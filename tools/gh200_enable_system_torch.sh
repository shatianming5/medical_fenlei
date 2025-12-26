#!/usr/bin/env bash
set -euo pipefail

# Expose system-installed CUDA-enabled torch/torchvision (apt: python3-torch-cuda, python3-torchvision-cuda)
# inside a conda env on GH200 (linux-aarch64).
#
# Usage:
#   tools/gh200_enable_system_torch.sh [env_name]
#
# This script symlinks the system torch packages into the conda env's site-packages.
# (Avoids adding the entire system dist-packages directory to sys.path, which can
# confuse pip due to distro package version strings.)

ENV_NAME="${1:-medical_fenlei}"

if [[ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  echo "not found: $HOME/miniconda3/etc/profile.d/conda.sh (install Miniconda first)" >&2
  exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" >/dev/null

SITE_PACKAGES="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"

if [[ -z "$SITE_PACKAGES" ]] || [[ ! -d "$SITE_PACKAGES" ]]; then
  echo "could not locate site-packages for env: $ENV_NAME" >&2
  exit 1
fi

SYSTEM_DIST="/usr/lib/python3/dist-packages"
if [[ ! -d "$SYSTEM_DIST" ]]; then
  echo "not found: $SYSTEM_DIST" >&2
  exit 1
fi

# If a previous version of this script added dist-packages via .pth, remove it.
PTH_FILE="$SITE_PACKAGES/zzz_system_dist-packages.pth"
rm -f "$PTH_FILE"

to_link=(
  "torch"
  "torchgen"
  "functorch"
  "torch-2.7.0.egg-info"
  "torchvision"
  "torchvision-0.22.0.egg-info"
)

for name in "${to_link[@]}"; do
  src="$SYSTEM_DIST/$name"
  dst="$SITE_PACKAGES/$name"
  if [[ -e "$dst" ]]; then
    continue
  fi
  if [[ ! -e "$src" ]]; then
    echo "not found: $src" >&2
    exit 1
  fi
  ln -s "$src" "$dst"
done

python - <<'PY'
import importlib

torch = importlib.import_module("torch")
torchvision = importlib.import_module("torchvision")

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("torchvision", getattr(torchvision, "__version__", None))
PY

echo "Linked system torch packages into: $SITE_PACKAGES"
