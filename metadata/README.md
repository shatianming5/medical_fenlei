# metadata/

本目录存放标注表等元数据文件（可能包含敏感信息），默认不提交到 Git。

示例（本地路径）：
- `metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx`

CLI 脚本会按优先级自动选择标注表路径：
1) `MEDICAL_FENLEI_LABELS_XLSX`
2) `/home/ubuntu/tim/导出数据第1~4017条数据20240329-To模型训练团队.xlsx`（若存在）
3) `metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx`
