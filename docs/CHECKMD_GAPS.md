# check.md 与当前项目差距清单（对照表）

本文件用于把 `check.md` 的“落地要求”逐条映射到当前代码实现，方便按优先级补齐与验收。

> 说明：本仓库默认不把 `data/`、`metadata/`、`artifacts/`、`cache/`、`outputs/` 等包含隐私或大文件的目录提交到 Git（见 `.gitignore`）。

---

## 0) 成功标准（稳定/可解释/可复现）

**已有**
- 固定验证集（dual split：`val=772 exams / 1538 ears`）
- 每次训练输出 `metrics.jsonl` 与每 epoch 的分类报告（dual 6 类/二分类）
- 多 seed 跑法（耳朵级 2D 路线）：`scripts/run_seeds_ear2d.py`
- bootstrap 置信区间（按 exam_id 抽样，耳朵级 2D 路线）：`scripts/eval_ear2d.py`
- 解释性：attention top-k 切片（`scripts/eval_ear2d.py`），Grad-CAM（`scripts/gradcam_ear2d.py`）
- 可复现配置：每次 ear2d 训练保存 `run_config.json`（含 git sha）

**仍可增强（可选）**
- 汇总 mean±std 的统一排行榜（目前 `scripts/run_seeds_ear2d.py` 会输出 per-run CSV，可继续做更漂亮的 Markdown 汇总）

---

## 1) 数据治理：manifest / 切分审计 / QA

**已有**
- `artifacts/dataset_index.csv`（exam 级：`exam_id/date/series_relpath/n_instances/left_code/right_code`）
- exam 级切分（`scripts/make_splits_dual.py`）避免左右耳泄漏
- 耳朵级 manifest：`scripts/build_manifest_ears.py`
- 病人级泄漏检查：`scripts/check_patient_leakage.py`
- patient-level split（可选开关）：`scripts/make_splits_dual.py --patient-split`
- QA 汇总：`scripts/qa_manifest_ears.py`

**仍可增强（可选）**
- 如果 patient_id 缺失，可加入更强的弱标识策略（目前已支持 patient_id/name hash + study_uid fallback）

---

## 2) 任务设计：二分类任务集（Pairwise + One-vs-Rest）

**已有**
- 二分类 task spec：`src/medical_fenlei/tasks.py`（dual 训练/推理已接入 `--label-task`）
- One-vs-Rest 代表任务：`normal_vs_abnormal`（包含 code6）
- 鉴别诊断代表任务：`cholesteatoma_vs_other_abnormal`
- code4 的 few-shot 流水线：`scripts/extract_embeddings_ear2d.py` + `scripts/fewshot_code4.py`

**仍可增强（可选）**
- 为更多任务补齐 pairwise/ovr 的双口径版本（当前已覆盖关键推荐任务）

---

## 3) 输入与预处理：ROI / 采样 / 方向一致性

**已有**
- 中线一刀切左右分割 + 均匀 32 切片采样（dual/side dataset）
- 右耳翻转对齐方向
- ear2d 路线的改进预处理（`src/medical_fenlei/data/ear_dataset.py`）：
  - slice 排序：`ImagePositionPatient/InstanceNumber`（`list_dicom_files_ipp`）
  - WL/WW jitter（训练时）：`scripts/train_ear2d.py`
  - 骨结构 midline（粗）：`_bone_midline_x`
  - z 连续块采样（启发式，`--sampling air_block`）
  - 颞骨粗 crop（骨 bbox + lateral bias，`crop_size`）

**仍可增强（可选）**
- 方向一致性（基于 ImageOrientationPatient 的更严格对齐）
- 统一物理尺度 resample spacing（离线预处理）

---

## 4) 模型路线：2D slice encoder + z 聚合（attention）

**已有**
- 3D dual backbone：ResNet/UNet/ViT（MONAI）
- 2D baseline：`SliceMeanResNet`（mean pooling across slices）
- 2D + attention pooling：`src/medical_fenlei/models/slice_attention_resnet.py`
- ear-level 训练入口：`scripts/train_ear2d.py`

**缺失/待补**
- （可选）轻量 transformer 聚合（当前已提供 attention pooling baseline）

---

## 5) 训练策略：不均衡/噪声/小样本

**已有**
- AdamW、weight_decay、label_smoothing、基础增强、AMP、auto-batch（dual）
- ear2d 默认配方（`scripts/train_ear2d.py`）：
  - Balanced sampler（WeightedRandomSampler）
  - BCEWithLogits + pos_weight（clip）
  - cosine decay + warmup（step 级）
  - grad clip（1.0）
  - early stop（默认 AUPRC）

**缺失/待补**
- （可选）Focal loss
- （可选）更严格的“linear probe → fine-tune”自动化（目前提供 `--freeze-backbone-epochs`）

---

## 6) 6 分类保留路线：分层两阶段

**缺失/待补**
- Stage1：正常 vs 异常（包含 code6）
- Stage2：异常细分（并将 code4 合并/或拆出 few-shot）

---

## 7) code4 few-shot（专用）

**已补齐**
- embedding 抽取：`scripts/extract_embeddings_ear2d.py`
- prototype + kNN 检索式解释：`scripts/fewshot_code4.py`
- 评估 bootstrap CI（按 exam_id）：`medical_fenlei.metrics.bootstrap_binary_metrics_by_exam`

---

## 8) 噪声与域差闭环

**已补齐（基础版）**
- 可疑样本列表（高置信错分 + top loss）：`scripts/eval_ear2d.py` 输出到 `reports/`

**仍可增强（可选）**
- 训练中逐步降权（小损失优先/robust reweighting）

---

## 9) 评估协议（可比较、可写报告）

**已有**
- dual：macro_recall/specificity/f1 等（多类）
- ear2d（二分类）：AUROC/AUPRC、Sensitivity/Specificity、Sensitivity@95%Spec + bootstrap CI：`scripts/eval_ear2d.py`

**仍可增强（可选）**
- 阈值校准（train 内 calibration/temperature scaling）

---

## 10) 可解释性

**已补齐（基础版）**
- attention top-k 切片：`scripts/eval_ear2d.py`（写入 `predictions_val.csv`）
- slice-level Grad-CAM：`scripts/gradcam_ear2d.py`
- case-based 检索（few-shot/kNN）：`scripts/fewshot_code4.py`

---

## 11) 工程吞吐（GPU 不空转）

**已有**
- dual volume cache（`cache/dual_volumes`）
- ear-level HU cache（`cache/ears_hu`）：`scripts/build_cache_ears.py`

**仍可增强（可选）**
- 将缓存产物压缩到 uint8 或 zarr/memmap 以进一步提速/省盘
