# Setup

## 1) 安装 Miniconda（本机）

建议安装到 `~/miniconda3`：

```bash
curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

## 2) 创建 conda 环境

```bash
conda env create -f environment.yml
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
