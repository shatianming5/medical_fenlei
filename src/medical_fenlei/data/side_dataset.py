from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from medical_fenlei.data.dicom import list_dicom_files, read_dicom_image


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if n >= k:
        return np.linspace(0, n - 1, num=k, dtype=int).tolist()
    idx = np.linspace(0, n - 1, num=n, dtype=int).tolist()
    while len(idx) < k:
        idx.append(n - 1)
    return idx


def _crop_side(img: np.ndarray, side: str, *, flip_right: bool) -> np.ndarray:
    _, w = img.shape
    mid = w // 2
    if side == "left":
        return img[:, :mid]
    if side == "right":
        out = img[:, mid:]
        if flip_right:
            out = np.fliplr(out).copy()
        return out
    raise ValueError(f"invalid side: {side}")


class EarCTSideDataset(Dataset):
    """
    One-ear (left/right) 6-class classification dataset.

    Expected columns in index_df:
      - exam_id, date, series_relpath
      - side in {"left","right"}
      - label in [0..5]
    """

    def __init__(
        self,
        *,
        index_df: pd.DataFrame,
        dicom_root: Path,
        num_slices: int = 32,
        image_size: int = 224,
        flip_right: bool = True,
    ) -> None:
        self.index = index_df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.num_slices = int(num_slices)
        self.image_size = int(image_size)
        self.flip_right = bool(flip_right)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        series_dir = self.dicom_root / str(row["series_relpath"])
        side = str(row["side"])
        label = int(row["label"])

        files = list_dicom_files(series_dir)
        indices = _evenly_spaced_indices(len(files), self.num_slices)
        if not indices:
            raise RuntimeError(f"no dicom files in: {series_dir}")

        slices: list[torch.Tensor] = []
        for j in indices:
            img = read_dicom_image(files[j])
            img = _crop_side(img, side, flip_right=self.flip_right)
            t = torch.from_numpy(img[None, ...])  # [1, H, W]
            t = F.interpolate(
                t[None, ...],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            slices.append(t)

        x = torch.stack(slices, dim=0)  # [K, 1, H, W]

        meta = {
            "exam_id": int(row["exam_id"]),
            "date": str(row["date"]),
            "series_relpath": str(row["series_relpath"]),
            "side": side,
        }

        return {"x": x, "label": torch.tensor(label, dtype=torch.long), "meta": meta}
