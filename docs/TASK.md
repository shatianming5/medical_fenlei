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

## 为什么要做 1% / 20% / 100% 三层？

用于观察 **数据量缩放** 对模型效果的影响，并快速验证 pipeline：

- 1%：最小可跑通/快速迭代
- 20%：中等规模，用于模型对比
- 100%：最终训练

本项目使用 `scripts/make_splits_dual.py` 对训练集做 **按类别分层等比例抽样**（按“耳朵标签”计数；选择 exam_id 子集），验证集按 exam_id 切分固定不变，避免左右耳泄漏，便于公平对比。

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
