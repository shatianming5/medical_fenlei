from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _clean_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        try:
            if path.exists():
                return path
        except Exception:
            continue
    return None


def default_dicom_base() -> Path:
    """
    Default DICOM base directory used by CLI scripts.

    Priority:
      1) env: MEDICAL_FENLEI_DICOM_BASE
      2) this workspace machine default: /home/ubuntu/tim/medical_data_2
      3) repo default: data/medical_data_2
    """
    env = _clean_path(os.environ.get("MEDICAL_FENLEI_DICOM_BASE"))
    candidates = [
        env,
        Path("/home/ubuntu/tim/medical_data_2"),
        Path("data/medical_data_2"),
    ]
    existing = _first_existing([p for p in candidates if p is not None])
    return existing or Path("data/medical_data_2")


def default_labels_xlsx() -> Path:
    """
    Default XLSX labels file used by CLI scripts.

    Priority:
      1) env: MEDICAL_FENLEI_LABELS_XLSX
      2) this workspace machine default: /home/ubuntu/tim/导出数据第1~4017条数据20240329-To模型训练团队.xlsx
      3) repo default: metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx
    """
    env = _clean_path(os.environ.get("MEDICAL_FENLEI_LABELS_XLSX"))
    candidates = [
        env,
        Path("/home/ubuntu/tim/导出数据第1~4017条数据20240329-To模型训练团队.xlsx"),
        Path("metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx"),
    ]
    existing = _first_existing([p for p in candidates if p is not None])
    return existing or Path("metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx")

