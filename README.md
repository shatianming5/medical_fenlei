# medical_fenlei（DICOM 医学影像诊断分类）

IMPORTANT（每一轮都要执行）
- 每一步改动都必须：`git add -A && git commit -m "<message>" && git push origin main`
- 严禁把 `data/`、`archives/`、`metadata/`、`logs/`、以及任何 checkpoint/输出目录提交到 Git；仓库已通过 `.gitignore` 强制忽略
- 远端仓库：`https://github.com/shatianming5/medical_fenlei`

## 当前数据结构（本地，不入库）
- `data/`：解压后的 DICOM（非常大）
- `metadata/`：标注表（可能含敏感信息）
- `archives/`：原始压缩包与分卷（非常大）
- `logs/`：解压/运行日志
- `tools/`：本地脚本工具（可入库）

## 可选：重新解压（本地）
```bash
JOBS=30 THREADS=30 tools/extract_all.sh
tail -f logs/extract_all.log
```

## 训练/推理（本地）

环境安装见 `docs/SETUP.md`。

1) 生成索引（会匹配 `metadata/*.xlsx` 的检查号/检查时间到本地 DICOM 目录）：

```bash
python scripts/build_index.py --out-csv artifacts/dataset_index.csv
```

2) 训练（输出到 `outputs/`，默认不入库）：

```bash
python scripts/train.py --index-csv artifacts/dataset_index.csv
```

3) 推理（输出到 `artifacts/`，默认不入库）：

```bash
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt --out-csv artifacts/predictions.csv
```
