# 任务定义与模型选择

## 任务是什么？

这是一个 **医学影像诊断分类** 任务：输入为颞骨/耳部 CT 的 DICOM 序列（3D 体数据），输出为 **同一次检查的左/右两只耳朵的诊断类别（双输出）**。

本项目把一次检查的序列按图像宽度一分为二得到左右耳区域（右耳默认做水平翻转以对齐方向），然后做 **双输出** 的 6 分类（左耳一个分类头、右耳一个分类头；共享 backbone）：

| 类别 ID | 标注代码 | 名称 |
|---:|---:|---|
| 0 | 1 | 慢性化脓性中耳炎 |
| 1 | 2 | 中耳胆脂瘤 |
| 2 | 3 | 分泌性中耳炎 |
| 3 | 4 | 胆固醇肉芽肿 |
| 4 | 5 | 正常 |
| 5 | 6 | 其他 |

标注来源：`metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx`（仅本地使用，不入库）。

## 二分类任务（当 6 分类太难时）

如果直接做 6 分类效果不佳，可以拆成多组二分类任务（仍然是“一次检查 -> 左/右双输出”）：

- `normal_vs_diseased`：正常(5) vs 患病(1..4)（默认不含“其他(6)”）
- `normal_vs_abnormal`：正常(5) vs 非正常(1,2,3,4,6)（推荐：表征学习主任务）
- `normal_vs_csoma`：正常(5) vs 慢性化脓性中耳炎(1)
- `normal_vs_cholesteatoma`：正常(5) vs 中耳胆脂瘤(2)
- `normal_vs_cholesterol_granuloma`：正常(5) vs 胆固醇肉芽肿(4)
- `normal_vs_ome`：正常(5) vs 分泌性中耳炎(3)
- `ome_vs_cholesterol_granuloma`：分泌性中耳炎(3) vs 胆固醇肉芽肿(4)
- `cholesteatoma_vs_csoma`：中耳胆脂瘤(2) vs 慢性化脓性中耳炎(1)
- `cholesteatoma_vs_other_abnormal`：胆脂瘤(2) vs 其他异常(1,3,4,6)（不含正常）

说明：
- 二分类标签固定为：`0=neg(任务名左侧)`，`1=pos(任务名右侧/患病)`
- 训练时会自动过滤 split CSV，仅保留属于该任务相关代码的检查；同一检查中“另一侧”若不属于该任务，会被 mask 掉不参与 loss
- 任务定义集中在 `src/medical_fenlei/tasks.py`，可按需要调整正/负样本的代码集合

示例：

```bash
python scripts/train_dual.py --pct 20 --model dual_resnet50_3d --label-task normal_vs_csoma --auto-batch
python scripts/run_experiments_dual.py --label-task normal_vs_csoma
```

## 推荐主路线：耳朵级 2D + Attention（更稳/更省显存/可解释）

`check.md` 推荐把“耳朵”作为样本单位，并优先用 2D slice encoder + attention pooling 作为主干：

- 模型：`scripts/train_ear2d.py`（ResNet18/34/50 2D 编码器 + z attention 聚合）
- 输入：每耳 `K=32` 张 slice（缓存为 HU），训练时做 WL/WW jitter + 轻量增强
- 训练：balanced sampler + BCE(pos_weight) + cosine(warmup) + grad clip
- 评估：AUROC/AUPRC + sensitivity/specificity + sensitivity@95%spec + bootstrap CI（按 exam_id）
- 解释：attention top-k 切片 + `scripts/gradcam_ear2d.py` 生成 CAM/overlay

建议流程（本地，不入库）：

1) 先生成 manifest（含 patient_key_hash，用于泄漏检查与 patient split）：

```bash
python scripts/build_manifest_ears.py --index-csv artifacts/dataset_index.csv --out-csv artifacts/manifest_ears.csv
python scripts/check_patient_leakage.py --manifest-csv artifacts/manifest_ears.csv
```

2) 生成 patient-level split（推荐单独目录）：

```bash
python scripts/make_splits_dual.py --index-csv artifacts/dataset_index.csv --out-dir artifacts/splits_dual_patient --manifest-csv artifacts/manifest_ears.csv --patient-split
```

3) 构建耳朵级 HU cache（否则训练/评估会频繁解码 DICOM）：

```bash
python scripts/build_cache_ears.py --splits-root artifacts/splits_dual_patient --pct 100 --num-slices 32 --image-size 224 --crop-size 192 --sampling even --num-workers 16
```

4) 训练（推荐从 `normal_vs_abnormal` 开始）：

```bash
python scripts/train_ear2d.py --splits-root artifacts/splits_dual_patient --pct 20 --label-task normal_vs_abnormal --backbone resnet18
```

5) 多 seed + CI：

```bash
python scripts/run_seeds_ear2d.py --splits-root artifacts/splits_dual_patient --pct 20 --label-task normal_vs_abnormal --seeds 0,1,2
```

6) 评估与解释：

```bash
python scripts/eval_ear2d.py --checkpoint outputs/<run>/checkpoints/best.pt --splits-root artifacts/splits_dual_patient --pct 20
python scripts/gradcam_ear2d.py --checkpoint outputs/<run>/checkpoints/best.pt --exam-id <id> --side left
```

## code4（胆固醇肉芽肿）few-shot（推荐单独报告）

由于 code4 极端稀有（训练正例 28、验证正例 8），更合理的做法是：

1) 先用 `normal_vs_abnormal` 训练一个 embedding 网络
2) 用 embedding 做 prototype / kNN / 强正则线性模型，并强制报告 bootstrap CI

本仓库提供：

```bash
python scripts/extract_embeddings_ear2d.py --checkpoint outputs/<run>/checkpoints/best.pt --split train --pct 100 --out-npz artifacts/emb_train.npz
python scripts/extract_embeddings_ear2d.py --checkpoint outputs/<run>/checkpoints/best.pt --split val   --pct 100 --out-npz artifacts/emb_val.npz
python scripts/fewshot_code4.py --train-npz artifacts/emb_train.npz --val-npz artifacts/emb_val.npz --task normal_vs_cholesterol_granuloma
```

## 为什么要做 1% / 20% / 100% 三层？

用于观察 **数据量缩放** 对模型效果的影响，并快速验证 pipeline：

- 1%：最小可跑通/快速迭代
- 20%：中等规模，用于模型对比
- 100%：最终训练

本项目使用 `scripts/make_splits_dual.py` 对训练集做 **按类别分层等比例抽样**（按“耳朵标签”计数；选择 exam_id 子集），验证集按 exam_id 切分固定不变，避免左右耳泄漏，便于公平对比。

补充：`artifacts/dataset_index.csv` 一共可匹配 `4012` 个 `exam_id`，其中 `153` 条左右耳均无标注，因此 split 会自动剔除，实际可用的“有至少一侧标注”的检查数为 `3859`。

## 模型选什么？

数据是 CT 体数据（3D），但体数据训练开销大；因此建议从轻到重：

### 1) ResNet（推荐从这里开始）

支持：`dual_resnet10_3d` / `dual_resnet18_3d` / `dual_resnet34_3d` / `dual_resnet50_3d` / `dual_resnet101_3d` / `dual_resnet152_3d` / `dual_resnet200_3d`

- 做法：左右耳分别裁剪成 (C=1, D=K, H, W) 体数据，用同一个 MONAI 3D ResNet 前向（batch 维度拼接），输出 (left,right) 两个 logits
- 优点：速度/效果均衡，适合从小到大逐步榨干显存

### 2) UNet（当作“超重”特征提取器）

模型：`dual_unet_3d`

- 做法：用 MONAI 3D UNet 得到 (C=num_classes) 的体素 logits，再做全局平均得到分类 logits
- 说明：对显存需求更高；要求 `num_slices` 和 `image_size` 能被 `prod(strides)` 整除（默认 strides=2,2,2,2 -> 要求能被 16 整除）

### 3) ViT（Transformer）

模型：`dual_vit_3d`

- 做法：MONAI ViT（3D patch embedding）做分类头
- 说明：依赖 `einops`（已加入 `environment.yml`）；要求 `img_size` 能被 `patch_size` 整除（默认 patch=4,16,16）

训练入口：`scripts/train_dual.py --model <name>`

说明：仓库仍保留单耳样本的 `scripts/train_side.py`/`scripts/predict_side.py` 作为对照/回退，但默认以双输出为主。

## 指标（敏感度/Recall 等）

训练时会在每个 epoch 评估验证集，并输出：
- `outputs/<run>/metrics.jsonl`：每个 epoch 的 loss/acc + `macro_recall`(=敏感度)/`macro_specificity`/`macro_f1` 等
- `outputs/<run>/reports/epoch_<n>.json`：完整明细（每类 precision/recall/specificity/F1 + confusion matrix）

## 性能（榨干 GPU/显存）

建议流程：

1) 先跑一次缓存构建（CPU 密集，但只需一次；之后训练/评估会快很多）：

```bash
python scripts/build_cache_dual.py --pct 100 --num-slices 32 --image-size 224 --num-workers 16
```

2) 训练时开启：
- `--auto-batch`：自动找最大 batch_size
- `--amp`：默认开启，可显著省显存/提速
- `--compile`：可选（PyTorch2），可能提速也可能变慢，建议单独对比

3) 如果想按“数据量优先 -> 模型其次”完整扫一遍：

```bash
python scripts/run_experiments_dual.py
```

默认 max epochs（可改）：1%/20%/100% 分别是 200/80/40；并默认开启 early stopping（见 `scripts/run_experiments_dual.py` 和 `scripts/train_dual.py`）。

4) 跑完后汇总对比（生成 CSV + Markdown 排行）：

```bash
python scripts/summarize_experiments_dual.py --metric macro_f1
```
