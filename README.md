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

1) 生成索引（仅按检查号 `exam_id` 匹配到本地 DICOM 目录；检查时间在该数据中大量不一致，仅作为 `label_date` 保留）：

```bash
python scripts/build_index.py --out-csv artifacts/dataset_index.csv
```

2) 生成 1% / 20% / 100% 三层训练集（按类别分层等比例抽样，输出到 `artifacts/splits/`，不入库）：

```bash
python scripts/make_splits_dual.py --index-csv artifacts/dataset_index.csv --out-dir artifacts/splits_dual --ratios 0.01,0.2,1.0
```

3) 一次检查 -> 左/右双输出 6 分类训练（MONAI 3D ResNet10；输出到 `outputs/`，不入库）：

```bash
python scripts/train_dual.py --pct 1
python scripts/train_dual.py --pct 20
python scripts/train_dual.py --pct 100
```

4) 推理（输出到 `artifacts/`，不入库）：

```bash
python scripts/predict_dual.py --checkpoint outputs/<run>/checkpoints/best.pt --index-csv artifacts/splits_dual/100pct/val.csv --out-csv artifacts/predictions_dual.csv
```

更多说明见 `docs/TASK.md`。
