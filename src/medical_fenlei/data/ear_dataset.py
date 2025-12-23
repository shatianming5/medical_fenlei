from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from medical_fenlei.data.dicom import list_dicom_files_ipp, read_dicom_hu


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if n >= k:
        return np.linspace(0, n - 1, num=k, dtype=int).tolist()
    idx = np.linspace(0, n - 1, num=n, dtype=int).tolist()
    while len(idx) < k:
        idx.append(n - 1)
    return idx


def _bone_midline_x(hu_slices: list[np.ndarray]) -> int | None:
    if not hu_slices:
        return None
    mids: list[float] = []
    for hu in hu_slices:
        if hu.ndim != 2:
            continue
        bone = hu > 300.0
        if not bone.any():
            continue
        xs = np.where(bone)[1].astype(np.float32)
        if xs.size <= 0:
            continue
        mids.append(float(xs.mean()))
    if not mids:
        return None
    return int(round(float(np.median(np.asarray(mids, dtype=np.float32)))))


def _split_left_right(hu: np.ndarray, *, mid_x: int | None, flip_right: bool) -> tuple[np.ndarray, np.ndarray]:
    h, w = hu.shape
    mid = int(mid_x) if mid_x is not None else (w // 2)
    mid = max(1, min(w - 1, mid))
    left = hu[:, :mid]
    right = hu[:, mid:]
    if flip_right:
        right = np.fliplr(right).copy()
    return left, right


def _crop_square(
    img: np.ndarray,
    *,
    crop_size: int,
    pad_value: float,
    lateral_bias: float = 0.25,
) -> np.ndarray:
    """
    Crop a square patch around a bone-based center.

    After right-ear flipping, the lateral side is expected to be near the left edge,
    so we bias x-center towards the left within the bone bbox.
    """
    h, w = img.shape
    bone = img > 300.0
    if bone.any():
        ys, xs = np.where(bone)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        cy = int(round((y0 + y1) / 2.0))
        cx = int(round(x0 + lateral_bias * max(1, (x1 - x0))))
    else:
        cy = h // 2
        cx = w // 2

    half = int(crop_size) // 2
    y0 = cy - half
    y1 = y0 + int(crop_size)
    x0 = cx - half
    x1 = x0 + int(crop_size)

    out = np.full((int(crop_size), int(crop_size)), float(pad_value), dtype=np.float32)

    src_y0 = max(0, y0)
    src_y1 = min(h, y1)
    src_x0 = max(0, x0)
    src_x1 = min(w, x1)

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def _air_ratio_in_bone_bbox(hu: np.ndarray) -> float:
    bone = hu > 300.0
    if not bone.any():
        return 0.0
    ys, xs = np.where(bone)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # expand bbox a bit to include cavities/tissue
    pad_y = max(1, int(round((y1 - y0) * 0.05)))
    pad_x = max(1, int(round((x1 - x0) * 0.05)))
    y0 = max(0, y0 - pad_y)
    y1 = min(hu.shape[0] - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(hu.shape[1] - 1, x1 + pad_x)
    roi = hu[y0 : y1 + 1, x0 : x1 + 1]
    if roi.size <= 0:
        return 0.0
    return float((roi < -500.0).mean())


def _choose_block_indices(
    files: list[Path],
    *,
    num_slices: int,
    block_len: int,
    max_probe_slices: int = 128,
) -> list[int]:
    n = len(files)
    if n <= 0:
        return []
    block_len = int(min(max(block_len, num_slices), n))
    if n <= block_len:
        return _evenly_spaced_indices(n, num_slices)

    step = max(1, n // int(max_probe_slices))
    probe_idx = list(range(0, n, step))
    ratios: list[float] = []
    for i in probe_idx:
        hu = read_dicom_hu(files[i])
        ratios.append(_air_ratio_in_bone_bbox(hu))
    if not ratios:
        return _evenly_spaced_indices(n, num_slices)

    # smooth with a small window to avoid spiky noise
    win = 5
    sm: list[float] = []
    for j in range(len(ratios)):
        lo = max(0, j - win)
        hi = min(len(ratios), j + win + 1)
        sm.append(float(np.mean(ratios[lo:hi])))
    best_j = int(np.argmax(np.asarray(sm)))
    center = int(probe_idx[best_j])
    start = max(0, min(n - block_len, center - block_len // 2))
    return np.linspace(start, start + block_len - 1, num=int(num_slices), dtype=int).tolist()


SamplingMode = Literal["even", "air_block"]


@dataclass(frozen=True)
class EarPreprocessSpec:
    num_slices: int = 32
    image_size: int = 224
    crop_size: int = 192
    sampling: SamplingMode = "even"
    block_len: int = 64
    flip_right: bool = True
    midline_slices: int = 5
    pad_hu: float = -1024.0
    version: str = "v1"

    def cache_key(self, *, series_relpath: str, side: str) -> str:
        s = (
            f"{series_relpath}|side={side}|d={int(self.num_slices)}|s={int(self.image_size)}|c={int(self.crop_size)}|"
            f"sampling={self.sampling}|block={int(self.block_len)}|flip={int(self.flip_right)}|midK={int(self.midline_slices)}|"
            f"pad={float(self.pad_hu)}|{self.version}"
        )
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def build_ear_hu_volume(
    *,
    series_dir: Path,
    series_relpath: str,
    exam_id: int,
    side: str,
    spec: EarPreprocessSpec,
) -> tuple[np.ndarray, dict]:
    files = list_dicom_files_ipp(series_dir)
    if not files:
        raise RuntimeError(f"no dicom files in: {series_dir}")

    if spec.sampling == "air_block":
        indices = _choose_block_indices(files, num_slices=int(spec.num_slices), block_len=int(spec.block_len))
    else:
        indices = _evenly_spaced_indices(len(files), int(spec.num_slices))
    if not indices:
        raise RuntimeError(f"no slice indices for: {series_dir}")

    # Midline estimation on a few slices (cheap-ish).
    midline = None
    if int(spec.midline_slices) > 0:
        probe = np.linspace(0, len(indices) - 1, num=min(int(spec.midline_slices), len(indices)), dtype=int).tolist()
        probe_hu = [read_dicom_hu(files[indices[j]]) for j in probe]
        midline = _bone_midline_x(probe_hu)

    vol: list[np.ndarray] = []
    for j in indices:
        hu = read_dicom_hu(files[j])
        left, right = _split_left_right(hu, mid_x=midline, flip_right=bool(spec.flip_right))
        img = left if side == "left" else right
        img = _crop_square(img, crop_size=int(spec.crop_size), pad_value=float(spec.pad_hu))
        t = torch.from_numpy(img[None, None, ...])  # (1,1,H,W)
        t = F.interpolate(t, size=(int(spec.image_size), int(spec.image_size)), mode="bilinear", align_corners=False)[0, 0]
        vol.append(t.numpy().astype(np.float32))

    arr = np.stack(vol, axis=0).astype(np.float16)  # (K,H,W) HU
    meta = {
        "exam_id": int(exam_id),
        "series_relpath": str(series_relpath),
        "side": str(side),
        "slice_indices": [int(x) for x in indices],
        "midline_x": int(midline) if midline is not None else -1,
        "n_files": int(len(files)),
    }
    return arr, meta


class EarCTHUEarDataset(Dataset):
    """
    Ear-level dataset that returns HU volumes.

    Expected columns in index_df:
      - exam_id, series_relpath, side
      - y (binary 0/1 float or int)
    """

    def __init__(
        self,
        *,
        index_df: pd.DataFrame,
        dicom_root: Path,
        spec: EarPreprocessSpec,
        cache_dir: Path | None = None,
        return_meta: bool = True,
    ) -> None:
        self.index = index_df.reset_index(drop=True)
        self.dicom_root = Path(dicom_root)
        self.spec = spec
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.return_meta = bool(return_meta)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return int(len(self.index))

    def _cache_path(self, *, exam_id: int, series_relpath: str, side: str) -> Path:
        h = self.spec.cache_key(series_relpath=str(series_relpath), side=str(side))
        return self.cache_dir / f"{int(exam_id)}_{str(side)}_{h}.npy"

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        exam_id = int(row["exam_id"])
        side = str(row["side"])
        series_relpath = str(row["series_relpath"])
        y = float(row["y"])
        try:
            label_code = int(row.get("label_code", -1))
        except Exception:
            label_code = -1

        meta = {
            "exam_id": exam_id,
            "side": side,
            "series_relpath": series_relpath,
            "label_code": int(label_code),
            "slice_indices": [-1 for _ in range(int(self.spec.num_slices))],
            "midline_x": -1,
            "n_files": -1,
        }

        if self.cache_dir is not None:
            cache_path = self._cache_path(exam_id=exam_id, series_relpath=series_relpath, side=side)
            if cache_path.exists():
                arr = np.load(cache_path)  # (K,H,W) float16 HU
                hu = torch.from_numpy(arr.astype(np.float32))
                out = {"hu": hu, "y": torch.tensor(y, dtype=torch.float32)}
                if self.return_meta:
                    meta_path = cache_path.with_suffix(".json")
                    if meta_path.exists():
                        try:
                            meta2 = json.loads(meta_path.read_text(encoding="utf-8"))
                            if isinstance(meta2, dict):
                                meta.update(meta2)
                        except Exception:
                            pass
                    out["meta"] = meta
                return out

        series_dir = self.dicom_root / series_relpath
        arr, meta2 = build_ear_hu_volume(series_dir=series_dir, series_relpath=series_relpath, exam_id=exam_id, side=side, spec=self.spec)
        if self.cache_dir is not None:
            cache_path = self._cache_path(exam_id=exam_id, series_relpath=series_relpath, side=side)
            tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
            try:
                with open(tmp_path, "wb") as f:
                    np.save(f, arr)
                os.replace(tmp_path, cache_path)
                meta_path = cache_path.with_suffix(".json")
                meta_path.write_text(json.dumps(meta2, ensure_ascii=False) + "\n", encoding="utf-8")
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        hu = torch.from_numpy(arr.astype(np.float32))
        out = {"hu": hu, "y": torch.tensor(y, dtype=torch.float32)}
        if self.return_meta:
            meta.update(meta2)
            out["meta"] = meta
        return out
