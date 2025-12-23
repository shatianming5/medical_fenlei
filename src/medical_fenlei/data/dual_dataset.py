from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from medical_fenlei.constants import CLASS_ID_TO_NAME
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


def _crop_left_right(img: np.ndarray, *, flip_right: bool) -> tuple[np.ndarray, np.ndarray]:
    _, w = img.shape
    mid = w // 2
    left = img[:, :mid]
    right = img[:, mid:]
    if flip_right:
        right = np.fliplr(right).copy()
    return left, right


def _code_to_label(code) -> tuple[int, bool]:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return -1, False
    try:
        code_int = int(code)
    except Exception:
        return -1, False
    if not (1 <= code_int <= len(CLASS_ID_TO_NAME)):
        return -1, False
    return code_int - 1, True


class EarCTDualDataset(Dataset):
    """
    One-exam dual-output dataset (left + right ear).

    Expected columns in index_df:
      - exam_id, date, series_relpath
      - left_code, right_code (1..6)

    Returns:
      - image: (2, 1, D, H, W)  # 0=left, 1=right
      - label: (2,) with -1 for missing
      - label_mask: (2,) bool indicating present labels
    """

    def __init__(
        self,
        *,
        index_df: pd.DataFrame,
        dicom_root: Path,
        num_slices: int = 32,
        image_size: int = 224,
        flip_right: bool = True,
        cache_dir: Path | None = None,
        cache_dtype: str = "float16",
        return_image: bool = True,
    ) -> None:
        self.index = index_df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.num_slices = int(num_slices)
        self.image_size = int(image_size)
        self.flip_right = bool(flip_right)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_dtype = str(cache_dtype)
        self.return_image = bool(return_image)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.index)

    def _cache_path(self, *, exam_id: int, series_relpath: str) -> Path:
        key = f"{series_relpath}|d={self.num_slices}|s={self.image_size}|flip={int(self.flip_right)}|v=2"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{exam_id}_{h}.npy"

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        series_dir = self.dicom_root / str(row["series_relpath"])
        exam_id = int(row["exam_id"])
        series_relpath = str(row["series_relpath"])

        left_label, left_present = _code_to_label(row.get("left_code"))
        right_label, right_present = _code_to_label(row.get("right_code"))

        if self.cache_dir is not None:
            cache_path = self._cache_path(exam_id=exam_id, series_relpath=series_relpath)
            if cache_path.exists():
                meta = {"exam_id": exam_id, "date": str(row["date"]), "series_relpath": series_relpath}
                label = torch.tensor([left_label, right_label], dtype=torch.long)
                label_mask = torch.tensor([left_present, right_present], dtype=torch.bool)
                if self.return_image:
                    arr = np.load(cache_path)
                    image = torch.from_numpy(arr).to(torch.float32)
                    return {"image": image, "label": label, "label_mask": label_mask, "meta": meta}
                return {"label": label, "label_mask": label_mask, "meta": meta}

        files = list_dicom_files(series_dir)
        indices = _evenly_spaced_indices(len(files), self.num_slices)
        if not indices:
            raise RuntimeError(f"no dicom files in: {series_dir}")

        left_slices: list[torch.Tensor] = []
        right_slices: list[torch.Tensor] = []
        for j in indices:
            img = read_dicom_image(files[j])
            left, right = _crop_left_right(img, flip_right=self.flip_right)
            left_t = torch.from_numpy(left[None, None, ...])  # (1,1,H,W)
            right_t = torch.from_numpy(right[None, None, ...])
            left_t = F.interpolate(left_t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            right_t = F.interpolate(right_t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            left_slices.append(left_t[0])  # (1,H,W)
            right_slices.append(right_t[0])

        left_t = torch.stack(left_slices, dim=0).permute(1, 0, 2, 3).contiguous()  # (1,D,H,W)
        right_t = torch.stack(right_slices, dim=0).permute(1, 0, 2, 3).contiguous()

        image = torch.stack([left_t, right_t], dim=0)  # (2, 1, D, H, W)

        meta = {
            "exam_id": exam_id,
            "date": str(row["date"]),
            "series_relpath": series_relpath,
        }

        label = torch.tensor([left_label, right_label], dtype=torch.long)
        label_mask = torch.tensor([left_present, right_present], dtype=torch.bool)

        if self.cache_dir is not None:
            cache_path = self._cache_path(exam_id=exam_id, series_relpath=series_relpath)
            tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
            try:
                arr = image.detach().cpu().numpy()
                if self.cache_dtype == "float16":
                    arr = arr.astype(np.float16)
                elif self.cache_dtype == "float32":
                    arr = arr.astype(np.float32)
                else:
                    raise ValueError(f"unsupported cache_dtype: {self.cache_dtype!r}")
                with open(tmp_path, "wb") as f:
                    np.save(f, arr)
                os.replace(tmp_path, cache_path)
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        if self.return_image:
            return {"image": image, "label": label, "label_mask": label_mask, "meta": meta}
        return {"label": label, "label_mask": label_mask, "meta": meta}
