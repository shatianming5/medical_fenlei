from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from medical_fenlei.data.dicom import list_dicom_files, read_dicom_image


def _safe_int(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _code_to_class(code: int | None) -> int:
    # Label codes in XLSX are 1..6. Map to 0..5. Missing -> -1.
    if code is None:
        return -1
    if 1 <= code <= 6:
        return code - 1
    return -1


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if n >= k:
        return np.linspace(0, n - 1, num=k, dtype=int).tolist()
    # pad by repeating end slices
    idx = np.linspace(0, n - 1, num=n, dtype=int).tolist()
    while len(idx) < k:
        idx.append(n - 1)
    return idx


def _split_left_right(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    mid = w // 2
    return img[:, :mid], img[:, mid:]


class EarCTDataset(Dataset):
    def __init__(
        self,
        *,
        index_df: pd.DataFrame,
        dicom_root: Path,
        num_slices: int = 32,
        image_size: int = 224,
    ) -> None:
        self.index = index_df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.num_slices = int(num_slices)
        self.image_size = int(image_size)

    def __len__(self) -> int:  # noqa: D401
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        series_dir = self.dicom_root / str(row["series_relpath"])
        files = list_dicom_files(series_dir)
        indices = _evenly_spaced_indices(len(files), self.num_slices)

        left_slices: list[torch.Tensor] = []
        right_slices: list[torch.Tensor] = []

        for j in indices:
            img = read_dicom_image(files[j])
            left, right = _split_left_right(img)

            # [1, H, W]
            left_t = torch.from_numpy(left[None, ...])
            right_t = torch.from_numpy(right[None, ...])

            left_t = F.interpolate(left_t[None, ...], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            right_t = F.interpolate(right_t[None, ...], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]

            left_slices.append(left_t)
            right_slices.append(right_t)

        left_x = torch.stack(left_slices, dim=0)  # [K, 1, H, W]
        right_x = torch.stack(right_slices, dim=0)

        left_label = _code_to_class(_safe_int(row.get("left_code")))
        right_label = _code_to_class(_safe_int(row.get("right_code")))

        return {
            "left": left_x,
            "right": right_x,
            "left_label": torch.tensor(left_label, dtype=torch.long),
            "right_label": torch.tensor(right_label, dtype=torch.long),
            "meta": {
                "exam_id": int(row["exam_id"]),
                "date": str(row["date"]),
                "series_relpath": str(row["series_relpath"]),
            },
        }
