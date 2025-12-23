# check.md 与当前项目差距清单（对照表）

本文件用于把 `check.md` 的“落地要求”逐条映射到当前代码实现，方便按优先级补齐与验收。

> 说明：本仓库默认不把 `data/`、`metadata/`、`artifacts/`、`cache/`、`outputs/` 等包含隐私或大文件的目录提交到 Git（见 `.gitignore`）。

---

## 0) 成功标准（稳定/可解释/可复现）

**已有**
- 固定验证集（dual split：`val=772 exams / 1538 ears`）
- 每次训练输出 `metrics.jsonl` 与每 epoch 的分类报告（dual 6 类/二分类）

**缺失/待补**
- 多 seed（≥3）稳定性报告（mean±std）
- bootstrap 置信区间（按 exam_id 抽样）
- 解释性产出（关键切片/Grad-CAM/相似病例检索）
- “数据清单/预处理/训练配置/代码版本”一一对应的可审计资产（manifest + run config + git sha）

---

## 1) 数据治理：manifest / 切分审计 / QA

**已有**
- `artifacts/dataset_index.csv`（exam 级：`exam_id/date/series_relpath/n_instances/left_code/right_code`）
- exam 级切分（`scripts/make_splits_dual.py`）避免左右耳泄漏

**缺失/待补**
- 统一“耳朵级” manifest（每耳一行：side/label/路径/spacing/thickness/vendor/kernel 等）
- 病人级泄漏检查（若能从 DICOM 提取 PatientID；否则用弱标识 hash）
- 每个二分类 task 的 val 正例数/占比与审计报表
- QA 三张表：扫描参数分布、切片覆盖/范围分布、可疑样本列表

---

## 2) 任务设计：二分类任务集（Pairwise + One-vs-Rest）

**已有**
- 二分类 task spec：`src/medical_fenlei/tasks.py`（dual 训练/推理已接入 `--label-task`）

**缺失/待补**
- 每个 task 同时提供 `pairwise` 与 `one-vs-rest` 两种口径（并在 manifest 中形成 `task_valid_mask`）
- 补齐推荐任务：`正常 vs 非正常(1,2,3,4,6)`、`胆脂瘤(2) vs 其他异常(1,3,4,6)` 等
- 明确 code4（胆固醇肉芽肿）走 few-shot/embedding 路线，而不是常规端到端二分类

---

## 3) 输入与预处理：ROI / 采样 / 方向一致性

**已有**
- 中线一刀切左右分割 + 均匀 32 切片采样（dual/side dataset）
- 右耳翻转对齐方向

**缺失/待补**
- WL/WW jitter（CT 专用增强）
- slice 排序：使用 `ImagePositionPatient` / `InstanceNumber` 而非文件名启发式
- 更稳的左右分割（骨结构对称轴/骨投影中线）
- z 采样从“全序列均匀”升级为“颞骨区域连续块”（骨+气房启发式）
- 更聚焦的 patch crop（减少无关结构）
- （可选）统一物理尺度 resample spacing（离线预处理）

---

## 4) 模型路线：2D slice encoder + z 聚合（attention）

**已有**
- 3D dual backbone：ResNet/UNet/ViT（MONAI）
- 2D baseline：`SliceMeanResNet`（mean pooling across slices）

**缺失/待补**
- attention pooling / 轻量 transformer 聚合（输出关键切片权重）
- ear-level batch 的训练主流程（按耳为样本，便于不均衡采样/解释）

---

## 5) 训练策略：不均衡/噪声/小样本

**已有**
- AdamW、weight_decay、label_smoothing、基础增强、AMP、auto-batch（dual）

**缺失/待补**
- Balanced batch（WeightedRandomSampler / 自定义 batch sampler）
- 二分类 loss：BCEWithLogits + pos_weight（并 clip）
- （可选）Focal loss
- cosine decay + warmup、梯度裁剪
- 早停指标以 AUPRC 或 sensitivity@high-specificity 为主（而非 accuracy）
- 1%/20%：linear probe → fine-tune 的公平对比策略

---

## 6) 6 分类保留路线：分层两阶段

**缺失/待补**
- Stage1：正常 vs 异常（包含 code6）
- Stage2：异常细分（并将 code4 合并/或拆出 few-shot）

---

## 7) code4 few-shot（专用）

**缺失/待补**
- 先训练“通用表征”（推荐：正常 vs 异常）
- embedding 上做 prototype / 正则化 logistic / kNN
- 评估强制 bootstrap CI（按 exam_id）

---

## 8) 噪声与域差闭环

**缺失/待补**
- 输出可疑样本列表：高置信错分、长期高 loss、按 date_match=False 分组
- （可选）小损失优先/样本降权策略

---

## 9) 评估协议（可比较、可写报告）

**已有**
- dual：macro_recall/specificity/f1 等（多类）

**缺失/待补**
- 二分类：AUROC/AUPRC、Sensitivity/Specificity、Sensitivity@95%Specificity
- 阈值策略固定（0.5 或训练内 calibration）
- bootstrap CI（按 exam_id）
- 3-seed 稳定性汇总（mean±std + CI）

---

## 10) 可解释性

**缺失/待补**
- attention 权重 top-k 切片输出
- slice-level Grad-CAM（对 top slice）
- embedding 最近邻检索（相似病例）

---

## 11) 工程吞吐（GPU 不空转）

**已有**
- dual volume cache（`cache/dual_volumes`）

**缺失/待补**
- 离线预处理“耳朵级”产物（crop + 对齐 + 采样 + WL/WW + normalize）
- DataLoader prefetch/persistent_workers 等进一步优化（针对离线产物）

