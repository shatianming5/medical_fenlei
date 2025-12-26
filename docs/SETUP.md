# Setup

## 1) 安装 Miniconda（本机）

建议安装到 `~/miniconda3`：

```bash
ARCH="$(uname -m)"
if [[ "$ARCH" == "x86_64" ]]; then
  URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
elif [[ "$ARCH" == "aarch64" ]]; then
  URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
else
  echo "unsupported arch: $ARCH" >&2
  exit 1
fi
curl -fsSL -o /tmp/miniconda.sh "$URL"
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

## 2) 创建 conda 环境

```bash
# x86_64（推荐）：使用 CUDA conda 包
conda env create -f environment.yml

# GH200 / linux-aarch64：使用 environment.gh200.yml，并链接系统 CUDA torch
# conda env create -f environment.gh200.yml
# conda activate medical_fenlei
# tools/gh200_enable_system_torch.sh medical_fenlei
# python -m pip install --no-deps monai

conda activate medical_fenlei
pip install -e .
```

如果已经创建过环境，更新依赖：

```bash
conda env update -f environment.yml --prune
```

验证 GPU：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 3) （可选）启用 Weights & Biases（wandb）

不要把 `WANDB_API_KEY` 写进代码或提交到仓库；用环境变量即可：

```bash
export WANDB_API_KEY="...你的key..."
```

Dual 3D 训练启用上传：

```bash
MF_WANDB=1 MF_WANDB_PROJECT=medical_fenlei bash scripts/run_dual_models_200ep_max.sh
```

或单独跑脚本：

```bash
python scripts/train_dual.py --wandb --wandb-project medical_fenlei ...
```
