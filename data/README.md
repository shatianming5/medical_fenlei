# data/

本目录存放本地解压后的 DICOM 数据（非常大），仅用于训练/推理，不会提交到 Git。

默认数据根目录示例：
- `data/medical_data_2/`

CLI 脚本会按优先级自动选择 DICOM 数据目录：
1) `MEDICAL_FENLEI_DICOM_BASE`
2) `/home/ubuntu/tim/medical_data_2`（若存在）
3) `data/medical_data_2/`
