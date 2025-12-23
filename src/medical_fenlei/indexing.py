from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date_dir(name: str) -> bool:
    return bool(_DATE_RE.match(name))


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


def _build_exam_dir_lookup(dicom_root: Path) -> dict[int, list[Path]]:
    """
    Build a lookup from exam_id -> list[exam_dir] by scanning dicom_root/*/*.

    The dataset layout is assumed to be:
      dicom_root/<YYYY-MM-DD>/<exam_id>/<series_id>/*.dcm
    """
    lookup: dict[int, list[Path]] = defaultdict(list)

    for d in dicom_root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        if not _is_date_dir(d.name):
            continue
        for e in d.iterdir():
            if not e.is_dir() or e.name.startswith("."):
                continue
            try:
                exam_id = int(e.name)
            except Exception:
                continue
            lookup[exam_id].append(e)

    return dict(lookup)


def _resolve_exam_dir(exam_dirs: list[Path]) -> tuple[Path, bool]:
    if len(exam_dirs) == 1:
        return exam_dirs[0], False

    # When an exam_id appears under multiple dates, pick the one with the most
    # DICOM instances in its best series. (We intentionally do NOT use label
    # date; matching is by exam_id only.)
    best_exam_dir = None
    best_instances = -1
    for cand in exam_dirs:
        _, n_instances = _pick_series_dir(cand)
        if n_instances > best_instances:
            best_exam_dir = cand
            best_instances = n_instances
    if best_exam_dir is not None:
        return best_exam_dir, True

    # Fallback: deterministic pick.
    return sorted(exam_dirs, key=lambda p: p.as_posix())[0], True


def build_dataset_index(labels: pd.DataFrame, *, dicom_root: Path) -> pd.DataFrame:
    """
    Match labels to local DICOM folders and pick one Series folder per exam.

    Matching strategy: by exam_id only (ignores label date for path selection).

    Output columns:
      - exam_id
      - date (folder date on disk)
      - label_date (date from XLSX, may differ)
      - exam_relpath, series_relpath
      - n_instances
      - left_code, right_code
    """
    rows: list[dict] = []

    exam_lookup = _build_exam_dir_lookup(dicom_root)

    for r in labels.itertuples(index=False):
        exam_dirs = exam_lookup.get(int(r.exam_id))
        if not exam_dirs:
            continue

        exam_dir, ambiguous = _resolve_exam_dir(exam_dirs)

        series_dir, n_instances = _pick_series_dir(exam_dir)
        if series_dir is None or n_instances <= 0:
            continue

        folder_date = exam_dir.parent.name
        label_date = getattr(r, "date", None)
        rows.append(
            {
                "exam_id": int(r.exam_id),
                "date": folder_date,
                "label_date": label_date,
                "folder_date": folder_date,
                "date_match": bool(label_date == folder_date) if label_date is not None else False,
                "ambiguous_match": bool(ambiguous),
                "exam_relpath": str(exam_dir.relative_to(dicom_root)),
                "series_relpath": str(series_dir.relative_to(dicom_root)),
                "n_instances": int(n_instances),
                "left_code": r.left_code,
                "right_code": r.right_code,
            }
        )

    return pd.DataFrame(rows)
