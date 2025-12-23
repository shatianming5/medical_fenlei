from __future__ import annotations

from pathlib import Path

import pandas as pd


def _count_dcm_files(series_dir: Path) -> int:
    # Fast-ish counting without recursion.
    try:
        return sum(1 for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    except FileNotFoundError:
        return 0


def _pick_series_dir(exam_dir: Path) -> tuple[Path | None, int]:
    series_dirs = [p for p in exam_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not series_dirs:
        return None, 0
    best = None
    best_n = -1
    for s in series_dirs:
        n = _count_dcm_files(s)
        if n > best_n:
            best = s
            best_n = n
    return best, max(best_n, 0)


def build_dataset_index(labels: pd.DataFrame, *, dicom_root: Path) -> pd.DataFrame:
    """
    Match labels to local DICOM folders and pick one Series folder per exam.

    Output columns:
      - exam_id, date
      - exam_relpath, series_relpath
      - n_instances
      - left_code, right_code
    """
    rows: list[dict] = []

    for r in labels.itertuples(index=False):
        exam_dir = dicom_root / r.date / str(r.exam_id)
        if not exam_dir.is_dir():
            continue

        series_dir, n_instances = _pick_series_dir(exam_dir)
        if series_dir is None or n_instances <= 0:
            continue

        rows.append(
            {
                "exam_id": int(r.exam_id),
                "date": r.date,
                "exam_relpath": str(exam_dir.relative_to(dicom_root)),
                "series_relpath": str(series_dir.relative_to(dicom_root)),
                "n_instances": int(n_instances),
                "left_code": r.left_code,
                "right_code": r.right_code,
            }
        )

    return pd.DataFrame(rows)
