# check.md 与当前项目差距清单（对照表）

本文件用于把 `check.md` 的“落地要求”逐条映射到当前代码实现，方便按优先级补齐与验收。

> 说明：本仓库默认不把 `data/`、`metadata/`、`artifacts/`、`cache/`、`outputs/` 等包含隐私或大文件的目录提交到 Git（见 `.gitignore`）。
>
> 兼容路径：如果你需要用绝对路径对照，本机已建立软链接：`/home/ubuntu/tim_download/check.md -> /home/ubuntu/medical_fenlei/check.md`。

---

## 0) 成功标准（稳定/可解释/可复现）

**已有**
- 固定验证集（dual split：`val=772 exams / 1538 ears`）
- 每次训练输出 `metrics.jsonl` 与每 epoch 的分类报告（dual 6 类/二分类）
- 多 seed 跑法（耳朵级 2D 路线）：`scripts/run_seeds_ear2d.py`
- bootstrap 置信区间（按 exam_id 抽样，耳朵级 2D 路线）：`scripts/eval_ear2d.py`
- 解释性：attention top-k 切片（`scripts/eval_ear2d.py`），Grad-CAM（`scripts/gradcam_ear2d.py`）
- 可复现配置：每次 ear2d 训练保存 `run_config.json`（含 git sha）

**已补齐**
- 多 seed 汇总（mean±std）：`scripts/summarize_experiments_ear2d.py`

---

## 1) 数据治理：manifest / 切分审计 / QA

**已有**
- `artifacts/dataset_index.csv`（exam 级：`exam_id/date/series_relpath/n_instances/left_code/right_code`）
- exam 级切分（`scripts/make_splits_dual.py`）避免左右耳泄漏
- 耳朵级 manifest：`scripts/build_manifest_ears.py`
- 病人级泄漏检查：`scripts/check_patient_leakage.py`
- patient-level split（可选开关）：`scripts/make_splits_dual.py --patient-split`
- QA 汇总：`scripts/qa_manifest_ears.py`

**已补齐**
- 病人弱标识（manifest）：`patient_key_hash = patient_id_hash || patient_name_hash || study_uid hash`（见 `scripts/build_manifest_ears.py`）

---

## 2) 任务设计：二分类任务集（Pairwise + One-vs-Rest）

**已有**
- 二分类 task spec：`src/medical_fenlei/tasks.py`（dual 训练/推理已接入 `--label-task`）
- One-vs-Rest 代表任务：`normal_vs_abnormal`（包含 code6）
- 鉴别诊断代表任务：`cholesteatoma_vs_other_abnormal`
- code4 的 few-shot 流水线：`scripts/extract_embeddings_ear2d.py` + `scripts/fewshot_code4.py`

**已覆盖（关键推荐任务）**
- 二分类任务集已覆盖：`normal_vs_abnormal / normal_vs_cholesteatoma / normal_vs_csoma / normal_vs_ome / cholesteatoma_vs_other_abnormal / ...`（见 `src/medical_fenlei/tasks.py`）

---

## 3) 输入与预处理：ROI / 采样 / 方向一致性

**已有**
- 强制 slice 排序：优先 `ImagePositionPatient/InstanceNumber`（`list_dicom_files_ipp`）
- 强制方向统一：基于 `ImageOrientationPatient` 做 in-plane 规范化（见 `src/medical_fenlei/data/dicom.py`）
- 中线一刀切左右分割 + 均匀 32 切片采样（dual/side dataset）
- 右耳翻转对齐方向
- ear2d 路线的改进预处理（`src/medical_fenlei/data/ear_dataset.py`）：
  - slice 排序：`ImagePositionPatient/InstanceNumber`（`list_dicom_files_ipp`）
  - WL/WW jitter（训练时）：`scripts/train_ear2d.py`
  - 骨结构 midline（粗）：`_bone_midline_x`
  - z 连续块采样（启发式，`--sampling air_block`）
  - 颞骨粗 crop（骨 bbox + lateral bias，`crop_size`）

**已补齐（可选）**
- 统一物理尺度（in-plane spacing）：`EarPreprocessSpec.target_spacing`（训练/缓存 CLI：`--target-spacing`；默认关闭）

---

## 4) 模型路线：2D slice encoder + z 聚合（attention）

**已有**
- 3D dual backbone：ResNet/UNet/ViT（MONAI）
- 2D baseline：`SliceMeanResNet`（mean pooling across slices）
- 2D + attention pooling：`src/medical_fenlei/models/slice_attention_resnet.py`
- ear-level 训练入口：`scripts/train_ear2d.py`

**已补齐**
- 轻量 transformer 聚合：`src/medical_fenlei/models/slice_attention_resnet.py`

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

**已补齐**
- Focal loss（可选）：`scripts/train_ear2d.py --loss focal`
- linear probe → fine-tune：`scripts/train_ear2d.py --freeze-backbone-epochs ...`

---

## 6) 6 分类保留路线：分层两阶段

**已补齐**
- Stage1（binary）：`scripts/train_ear2d.py --label-task normal_vs_abnormal`
- Stage2（abnormal subtype）：`scripts/train_ear2d_multiclass.py`（支持 `--merge-code4`）
- 分层评估：`scripts/eval_hierarchical_ear2d.py`

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

**已补齐（可选）**
- robust：batch 内丢弃 top-loss（逐步 warmup）：`scripts/train_ear2d.py --drop-high-loss-frac ... --drop-high-loss-warmup-epochs ...`

---

## 9) 评估协议（可比较、可写报告）

**已有**
- dual：macro_recall/specificity/f1 等（多类）
- ear2d（二分类）：AUROC/AUPRC、Sensitivity/Specificity、Sensitivity@95%Spec + bootstrap CI：`scripts/eval_ear2d.py`

**已补齐（基础版）**
- 阈值选择/校准（基于 val 预测，输出推荐阈值）：`scripts/calibrate_threshold_binary.py`

---

## 10) 可解释性

**已补齐（基础版）**
- attention top-k 切片：`scripts/eval_ear2d.py`（写入 `predictions_val.csv`）
- slice-level Grad-CAM：`scripts/gradcam_ear2d.py`
- case-based 检索（few-shot/kNN）：`scripts/fewshot_code4.py`
- 通用检索（任意 task / 任意 label_code 过滤）：`scripts/retrieve_neighbors_ear2d.py`

---

## 11) 工程吞吐（GPU 不空转）

**已有**
- dual volume cache（`cache/dual_volumes`）
- ear-level HU cache（`cache/ears_hu`）：`scripts/build_cache_ears.py`

**已补齐（可选，npz 压缩版）**
- `.npy -> .npz` 压缩脚本：`scripts/compress_cache_npz.py`
- 读取兼容：ear/dual dataset 会自动回退读取同名 `.npz`（优先 `.npy`）

---

## 12) 实验设计：check.md 5.1 长尾设置（Setting A / B）

**已补齐**
- 三段切分：train/val/test = 70%/10%/20%（`scripts/make_splits_dual.py` 默认 `--val-ratio 0.1 --test-ratio 0.2`）
- 设置A（仅 Label 1,2,3,5）：`scripts/make_splits_dual.py --keep-codes 1,2,3,5 --keep-mode drop_exam`
- 设置B（训练移除 Label2，测试保留）：`scripts/make_splits_dual.py --drop-train-codes 2 --drop-train-mode drop_exam`

---

## 13) 文本语义模块：CMBERT（check.md 4.2.1 / 4.3.2 / 5.2.3）

**已补齐（可选）**
- HF 文本编码（可替换为 CMBERT）：`src/medical_fenlei/text_encoder.py`
- *_proto 的类别原型初始化支持 `prompt_hf`：`scripts/train_dual.py --proto-init prompt_hf --proto-text-model <hf_id_or_path>`
- 单开关覆盖（推荐）：`scripts/train_dual.py --cmbert-model <hf_id_or_path>`（也可用环境变量 `MEDICAL_FENLEI_CMBERT_MODEL`）
- *_proto 的视觉-文本 InfoNCE 支持 HF：`scripts/train_dual.py --vl-contrastive-lambda ... --vl-text-encoder hf --vl-text-model <hf_id_or_path>`
- 语义一致性检索评估支持 `--text-encoder hf`：`scripts/eval_dual_retrieval.py`

---

## 14) 文本局部对齐：实体-属性三元组 / Local Alignment（check.md 4.2.2）

**已补齐（可选）**
- 实体-属性 triplet 抽取（轻量规则版，无额外 NLP 依赖）：`src/medical_fenlei/text_triplets.py`
- dual *_proto 训练接入局部对齐 loss（InfoNCE；ear-level）：`scripts/train_dual.py --vl-local-lambda ... --vl-local-max-triplets-per-ear ...`
- 训练日志与 `metrics.jsonl` 增加：`vl_local_loss` / `vl_local_queries`

**已补齐（task1 二分类原型对齐）**
- 二分类也使用“文本 prompts → prompt_hf 原型”（不再默认 rand）：`src/medical_fenlei/text_prompts.py` + `scripts/train_dual.py`（binary + *_proto 时 `proto_init=auto` 自动切到 `prompt_hf`）
