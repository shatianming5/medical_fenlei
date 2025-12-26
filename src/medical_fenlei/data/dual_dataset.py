from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.dicom import (
    list_dicom_files_ipp,
    read_dicom_hu,
    read_dicom_hu_and_spacing,
    read_dicom_hu_and_spacing_and_z,
    read_dicom_hu_and_z,
)


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
        if not bool(bone.any()):
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


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2 or not bool(mask.any()):
        return None
    ys, xs = np.where(mask)
    if ys.size <= 0 or xs.size <= 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _temporal_bone_bbox(
    mask: np.ndarray,
    *,
    lateral_band_frac: float,
    min_area: int,
) -> tuple[int, int, int, int] | None:
    """
    Pick a temporal-bone-ish bbox using connected components inside a lateral band.

    Assumptions:
      - After right-ear flipping, the lateral side is near the left edge for both sides.
      - Bone mask is `hu > 300`.
    """
    if mask.ndim != 2 or not bool(mask.any()):
        return None

    h, w = mask.shape
    band = int(round(float(lateral_band_frac) * float(w)))
    band = max(1, min(int(w), band))

    m = mask.copy()
    if band < w:
        m[:, band:] = False

    try:
        from scipy import ndimage as ndi
    except Exception:
        return _bbox_from_mask(m) or _bbox_from_mask(mask)

    labeled, n = ndi.label(m.astype(np.uint8))
    if int(n) <= 0:
        return _bbox_from_mask(mask)

    sizes = ndi.sum(m.astype(np.uint8), labeled, index=np.arange(1, int(n) + 1))
    sizes = np.asarray(sizes, dtype=np.float64)
    objs = ndi.find_objects(labeled)

    best_i = None
    best_area = -1.0
    for i in range(int(n)):
        area = float(sizes[i]) if i < sizes.size else 0.0
        if area <= 0:
            continue
        if int(min_area) > 0 and area < float(min_area):
            continue
        sl = objs[i] if i < len(objs) else None
        if sl is None or len(sl) != 2:
            continue
        if best_i is None or area > best_area:
            best_i = int(i)
            best_area = float(area)

    if best_i is None:
        best_i = int(np.argmax(sizes)) if sizes.size > 0 else None
    if best_i is None:
        return _bbox_from_mask(mask)

    sl = objs[best_i] if best_i < len(objs) else None
    if sl is None or len(sl) != 2:
        return _bbox_from_mask(mask)

    y0, y1 = int(sl[0].start), int(sl[0].stop) - 1
    x0, x1 = int(sl[1].start), int(sl[1].stop) - 1
    if y1 < y0 or x1 < x0:
        return _bbox_from_mask(mask)
    return y0, y1, x0, x1


CropMode = Literal["bbox_bias", "temporal_patch"]


def _infer_patch_center_from_bone(
    img: np.ndarray,
    *,
    crop_mode: CropMode,
    lateral_bias: float,
    lateral_band_frac: float,
    min_area: int,
) -> tuple[int, int] | None:
    if img.ndim != 2:
        return None
    h, w = img.shape
    bone = img > 300.0
    if not bool(bone.any()):
        return (h // 2, w // 2)

    if str(crop_mode) == "temporal_patch":
        bbox = _temporal_bone_bbox(bone, lateral_band_frac=float(lateral_band_frac), min_area=int(min_area)) or _bbox_from_mask(bone)
        if bbox is None:
            return (h // 2, w // 2)
        y0, y1, x0, x1 = bbox
        cy = int(round((y0 + y1) / 2.0))
        cx = int(round(float(x0) + float(lateral_bias) * float(max(1, x1 - x0))))
        return cy, cx

    bbox = _bbox_from_mask(bone)
    if bbox is None:
        return (h // 2, w // 2)
    y0, y1, x0, x1 = bbox
    cy = int(round((y0 + y1) / 2.0))
    cx = int(round(float(x0) + float(lateral_bias) * float(max(1, x1 - x0))))
    return cy, cx


def _infer_stack_center(
    imgs: list[np.ndarray],
    *,
    crop_mode: CropMode,
    lateral_bias: float,
    lateral_band_frac: float,
    min_area: int,
    max_slices: int,
) -> tuple[int, int] | None:
    if not imgs:
        return None
    k = len(imgs)
    take = min(int(max_slices), k) if int(max_slices) > 0 else k
    take = max(1, take)
    sel = np.linspace(0, k - 1, num=take, dtype=int).tolist()

    centers: list[tuple[int, int]] = []
    for j in sel:
        c = _infer_patch_center_from_bone(
            imgs[int(j)],
            crop_mode=crop_mode,
            lateral_bias=float(lateral_bias),
            lateral_band_frac=float(lateral_band_frac),
            min_area=int(min_area),
        )
        if c is not None:
            centers.append((int(c[0]), int(c[1])))
    if not centers:
        h, w = imgs[0].shape
        return (h // 2, w // 2)

    ys = np.asarray([c[0] for c in centers], dtype=np.float32)
    xs = np.asarray([c[1] for c in centers], dtype=np.float32)
    return int(round(float(np.median(ys)))), int(round(float(np.median(xs))))


def _crop_square_at(
    img: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    crop_size: int,
    pad_value: float,
) -> np.ndarray:
    h, w = img.shape
    half = int(crop_size) // 2
    y0 = int(center_y) - half
    x0 = int(center_x) - half
    y1 = y0 + int(crop_size)
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


def _resample_inplane_to_spacing(
    img: np.ndarray,
    *,
    spacing_y: float | None,
    spacing_x: float | None,
    target_spacing: float,
) -> np.ndarray:
    if img.ndim != 2:
        return img
    if spacing_y is None or spacing_x is None:
        return img
    ts = float(target_spacing)
    if ts <= 0:
        return img

    h, w = img.shape
    scale_y = float(spacing_y) / ts
    scale_x = float(spacing_x) / ts
    if not np.isfinite(scale_y) or not np.isfinite(scale_x) or scale_y <= 0 or scale_x <= 0:
        return img

    new_h = int(max(1, round(float(h) * float(scale_y))))
    new_w = int(max(1, round(float(w) * float(scale_x))))
    if new_h == h and new_w == w:
        return img

    t = torch.from_numpy(img.astype(np.float32)[None, None, ...])  # (1,1,H,W)
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)[0, 0]
    return t.numpy().astype(np.float32)


def _air_ratio_in_bone_bbox(hu: np.ndarray) -> float:
    bone = hu > 300.0
    if not bool(bone.any()):
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


def _air_ratio_in_bbox(
    hu: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    pad_frac: float = 0.05,
    x_min: int | None = None,
    x_max: int | None = None,
) -> float:
    if hu.ndim != 2:
        return 0.0
    h, w = hu.shape
    y0, y1, x0, x1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if y1 < y0 or x1 < x0:
        return 0.0
    pad_y = max(1, int(round((y1 - y0) * float(pad_frac))))
    pad_x = max(1, int(round((x1 - x0) * float(pad_frac))))
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    if x_min is not None:
        x0 = max(int(x_min), x0)
    if x_max is not None:
        x1 = min(int(x_max), x1)
    if x1 < x0:
        return 0.0
    roi = hu[y0 : y1 + 1, x0 : x1 + 1]
    if roi.size <= 0:
        return 0.0
    return float((roi < -500.0).mean())


def _temporal_air_score(hu: np.ndarray, *, band_frac: float = 0.25) -> float:
    """
    Heuristic score for "temporal bone region" along z.

    Use air ratio within bone bbox, but only inside *lateral* bands on both sides
    (to avoid sinuses/central structures).
    """
    if hu.ndim != 2:
        return 0.0
    h, w = hu.shape
    band = int(round(float(band_frac) * float(w)))
    band = max(1, min(w // 2, band))

    bone = hu > 300.0
    if not bool(bone.any()):
        return 0.0

    left = bone.copy()
    left[:, band:] = False
    right = bone.copy()
    right[:, : w - band] = False

    s_left = 0.0
    if bool(left.any()):
        bbox = _bbox_from_mask(left)
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            x0 = max(0, x0)
            x1 = min(band - 1, x1)
            if x1 >= x0:
                s_left = _air_ratio_in_bbox(hu, bbox=(y0, y1, x0, x1), x_min=0, x_max=band - 1)

    s_right = 0.0
    if bool(right.any()):
        bbox = _bbox_from_mask(right)
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            x0 = max(w - band, x0)
            x1 = min(w - 1, x1)
            if x1 >= x0:
                s_right = _air_ratio_in_bbox(hu, bbox=(y0, y1, x0, x1), x_min=w - band, x_max=w - 1)

    s = float(max(float(s_left), float(s_right)))
    if s > 0.0:
        return float(s)

    # Fallback: whole-slice bbox.
    return _air_ratio_in_bone_bbox(hu)


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
        ratios.append(_temporal_air_score(hu))
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
PairFeaturesMode = Literal["none", "self_other_diff"]


def _window_hu_to_unit(x: np.ndarray, *, wl: float, ww: float) -> np.ndarray:
    lower = float(wl) - float(ww) / 2.0
    upper = float(wl) + float(ww) / 2.0
    x = np.clip(x, lower, upper)
    return (x - lower) / (upper - lower + 1e-6)


@dataclass(frozen=True)
class DualPreprocessSpec:
    num_slices: int = 32
    image_size: int = 224
    crop_size: int = 192
    sampling: SamplingMode = "air_block"
    block_len: int = 64
    flip_right: bool = True
    midline_slices: int = 5
    crop_mode: CropMode = "temporal_patch"
    crop_lateral_band_frac: float = 0.6
    crop_lateral_bias: float = 0.25
    crop_min_area: int = 300
    pad_hu: float = -1024.0
    target_spacing: float | None = 0.7
    target_z_spacing: float | None = 0.8
    window_wl: float = 700.0
    window_ww: float = 4000.0
    # Optional second window (e.g., soft-tissue window). Disabled when window2_ww is None/<=0.
    window2_wl: float | None = None
    window2_ww: float | None = None
    # Pair features to approximate bilateral comparison (check.md Difference Map).
    # - none: per-ear channels only
    # - self_other_diff: concat(self, other, |self-other|) for each ear
    pair_features: PairFeaturesMode = "none"
    version: str = "v4"

    def cache_key(self, *, series_relpath: str) -> str:
        ts = self.target_spacing
        ts_s = f"|ts={float(ts):.6g}" if ts is not None and float(ts) > 0 else ""
        tz = self.target_z_spacing
        tz_s = f"|tz={float(tz):.6g}" if tz is not None and float(tz) > 0 else ""
        window2_s = ""
        if self.window2_ww is not None and float(self.window2_ww) > 0:
            wl2 = float(self.window2_wl) if self.window2_wl is not None else 0.0
            window2_s = f"|wl2={wl2:.4g}|ww2={float(self.window2_ww):.4g}"
        pair_s = f"|pair={str(self.pair_features)}"

        s = (
            f"{series_relpath}|d={int(self.num_slices)}|s={int(self.image_size)}|c={int(self.crop_size)}|"
            f"sampling={self.sampling}|block={int(self.block_len)}|flip={int(self.flip_right)}|midK={int(self.midline_slices)}|"
            f"crop={self.crop_mode}|band={float(self.crop_lateral_band_frac):.3g}|bias={float(self.crop_lateral_bias):.3g}|"
            f"minA={int(self.crop_min_area)}|pad={float(self.pad_hu)}|wl={float(self.window_wl):.4g}|ww={float(self.window_ww):.4g}"
            f"{window2_s}{pair_s}{ts_s}{tz_s}|{self.version}"
        )
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def build_dual_unit_volume(
    *,
    series_dir: Path,
    series_relpath: str,
    spec: DualPreprocessSpec,
) -> tuple[np.ndarray, dict]:
    files = list_dicom_files_ipp(series_dir)
    if not files:
        raise RuntimeError(f"no dicom files in: {series_dir}")

    n_files = int(len(files))
    if spec.sampling == "air_block":
        base_indices = _choose_block_indices(files, num_slices=int(spec.num_slices), block_len=int(spec.block_len))
    else:
        base_indices = _evenly_spaced_indices(n_files, int(spec.num_slices))
    if not base_indices:
        raise RuntimeError(f"no slice indices for: {series_dir}")

    target_spacing = spec.target_spacing
    target_z_spacing = spec.target_z_spacing
    need_z = target_z_spacing is not None and float(target_z_spacing) > 0

    def _read_hu(path: Path) -> tuple[np.ndarray, float | None, float | None, float | None]:
        if target_spacing is None or float(target_spacing) <= 0:
            if need_z:
                hu0, z0 = read_dicom_hu_and_z(path)
                return hu0, None, None, z0
            return read_dicom_hu(path), None, None, None

        if need_z:
            hu0, (sy, sx), z0 = read_dicom_hu_and_spacing_and_z(path)
        else:
            hu0, (sy, sx) = read_dicom_hu_and_spacing(path)
            z0 = None
        hu0 = _resample_inplane_to_spacing(hu0, spacing_y=sy, spacing_x=sx, target_spacing=float(target_spacing))
        return hu0, sy, sx, z0

    # If z-spacing normalization is requested, load a contiguous neighborhood around the center slice.
    spacing_z_est = None
    if need_z:
        center_idx = int(base_indices[len(base_indices) // 2])

        zs: list[float] = []
        probe_idx = [center_idx - 2, center_idx - 1, center_idx, center_idx + 1, center_idx + 2]
        for pi in probe_idx:
            if 0 <= int(pi) < n_files:
                try:
                    import pydicom

                    ds = pydicom.dcmread(
                        str(files[int(pi)]),
                        force=True,
                        stop_before_pixels=True,
                        specific_tags=["ImagePositionPatient"],
                    )
                    ipp = getattr(ds, "ImagePositionPatient", None)
                    if isinstance(ipp, (list, tuple)) and len(ipp) >= 3:
                        zs.append(float(ipp[2]))
                except Exception:
                    continue

        if len(zs) >= 3:
            diffs = np.diff(np.asarray(sorted(zs), dtype=np.float64))
            diffs = np.abs(diffs[np.isfinite(diffs)])
            diffs = diffs[diffs > 1e-6]
            if diffs.size > 0:
                spacing_z_est = float(np.median(diffs))
        if spacing_z_est is None or not np.isfinite(spacing_z_est) or spacing_z_est <= 0:
            spacing_z_est = float(target_z_spacing)

        window_mm = float(max(1, int(spec.num_slices) - 1)) * float(target_z_spacing) + 4.0 * float(target_z_spacing)
        n_needed = int(np.ceil(window_mm / float(spacing_z_est))) + 3
        n_needed = max(int(spec.num_slices), min(n_files, max(8, n_needed)))

        start = max(0, center_idx - n_needed // 2)
        end = start + n_needed - 1
        if end >= n_files:
            end = n_files - 1
            start = max(0, end - n_needed + 1)
        read_indices = list(range(int(start), int(end) + 1))
    else:
        read_indices = [int(x) for x in base_indices]

    hu_slices: list[np.ndarray] = []
    z_coords: list[float | None] = []
    spacing_y = None
    spacing_x = None
    for j in read_indices:
        hu, sy, sx, z = _read_hu(files[int(j)])
        spacing_y = sy
        spacing_x = sx
        hu_slices.append(hu.astype(np.float32))
        z_coords.append(z)

    if not hu_slices:
        raise RuntimeError(f"no slices loaded for: {series_dir}")

    # Midline estimation (reuse already-loaded slices).
    midline = None
    if int(spec.midline_slices) > 0:
        probe = np.linspace(0, len(hu_slices) - 1, num=min(int(spec.midline_slices), len(hu_slices)), dtype=int).tolist()
        probe_hu = [hu_slices[int(j)] for j in probe]
        midline = _bone_midline_x(probe_hu)

    # Split into left/right stacks (uncropped).
    left_imgs: list[np.ndarray] = []
    right_imgs: list[np.ndarray] = []
    for hu in hu_slices:
        left, right = _split_left_right(hu, mid_x=midline, flip_right=bool(spec.flip_right))
        left_imgs.append(left.astype(np.float32))
        right_imgs.append(right.astype(np.float32))

    # Infer one crop center per side for the whole stack to reduce jitter.
    left_center = _infer_stack_center(
        left_imgs,
        crop_mode=spec.crop_mode,
        lateral_bias=float(spec.crop_lateral_bias),
        lateral_band_frac=float(spec.crop_lateral_band_frac),
        min_area=int(spec.crop_min_area),
        max_slices=max(3, int(spec.midline_slices)),
    )
    if left_center is None:
        h0, w0 = left_imgs[0].shape
        left_center = (h0 // 2, w0 // 2)
    right_center = _infer_stack_center(
        right_imgs,
        crop_mode=spec.crop_mode,
        lateral_bias=float(spec.crop_lateral_bias),
        lateral_band_frac=float(spec.crop_lateral_band_frac),
        min_area=int(spec.crop_min_area),
        max_slices=max(3, int(spec.midline_slices)),
    )
    if right_center is None:
        h0, w0 = right_imgs[0].shape
        right_center = (h0 // 2, w0 // 2)

    def _crop_resize_stack(imgs: list[np.ndarray], *, center: tuple[int, int]) -> np.ndarray:
        cropped = [
            _crop_square_at(
                img,
                center_y=int(center[0]),
                center_x=int(center[1]),
                crop_size=int(spec.crop_size),
                pad_value=float(spec.pad_hu),
            )
            for img in imgs
        ]
        t = torch.from_numpy(np.stack(cropped, axis=0)[:, None, ...])  # (N,1,C,C)
        t = F.interpolate(t, size=(int(spec.image_size), int(spec.image_size)), mode="bilinear", align_corners=False)[:, 0]
        return t.numpy().astype(np.float32)

    left_vol = _crop_resize_stack(left_imgs, center=left_center)  # (N,H,W) HU
    right_vol = _crop_resize_stack(right_imgs, center=right_center)  # (N,H,W) HU

    slice_indices_out: list[int] = [int(x) for x in read_indices]
    z_out: list[float] | None = None

    if need_z:
        zs_np = np.asarray([float(z) if z is not None and np.isfinite(z) else np.nan for z in z_coords], dtype=np.float64)
        valid = np.isfinite(zs_np)
        if valid.sum() >= 2:
            if (~valid).any():
                idxs = np.arange(zs_np.size, dtype=np.float64)
                zs_np[~valid] = np.interp(idxs[~valid], idxs[valid], zs_np[valid])
            if spacing_z_est is not None and np.isfinite(float(spacing_z_est)) and float(spacing_z_est) > 0:
                zs_np = np.maximum.accumulate(zs_np)
        elif spacing_z_est is not None and np.isfinite(float(spacing_z_est)) and float(spacing_z_est) > 0:
            zs_np = np.arange(zs_np.size, dtype=np.float64) * float(spacing_z_est)
            valid = np.ones_like(zs_np, dtype=bool)

        if valid.sum() >= 2:
            z_center = float(zs_np[len(zs_np) // 2])
            k = int(spec.num_slices)
            offs = (np.arange(k, dtype=np.float64) - (float(k - 1) / 2.0)) * float(target_z_spacing)
            z_tgt = z_center + offs

            out_left: list[np.ndarray] = []
            out_right: list[np.ndarray] = []
            out_nearest: list[int] = []
            for zt in z_tgt.tolist():
                idx = int(np.searchsorted(zs_np, float(zt), side="left"))
                if idx <= 0:
                    out_left.append(left_vol[0])
                    out_right.append(right_vol[0])
                    out_nearest.append(int(read_indices[0]))
                    continue
                if idx >= len(zs_np):
                    out_left.append(left_vol[-1])
                    out_right.append(right_vol[-1])
                    out_nearest.append(int(read_indices[-1]))
                    continue

                z0 = float(zs_np[idx - 1])
                z1 = float(zs_np[idx])
                if not np.isfinite(z0) or not np.isfinite(z1) or abs(z1 - z0) < 1e-6:
                    out_left.append(left_vol[idx])
                    out_right.append(right_vol[idx])
                    out_nearest.append(int(read_indices[idx]))
                    continue
                a = float((float(zt) - z0) / (z1 - z0))
                a = float(min(1.0, max(0.0, a)))
                out_left.append((1.0 - a) * left_vol[idx - 1] + a * left_vol[idx])
                out_right.append((1.0 - a) * right_vol[idx - 1] + a * right_vol[idx])
                out_nearest.append(int(read_indices[idx] if a > 0.5 else read_indices[idx - 1]))

            left_vol = np.stack(out_left, axis=0).astype(np.float32)
            right_vol = np.stack(out_right, axis=0).astype(np.float32)
            slice_indices_out = out_nearest
            z_out = [float(x) for x in z_tgt.tolist()]

    # Final: trim/pad to K slices.
    k_out = int(spec.num_slices)
    for _name, vol in [("left", left_vol), ("right", right_vol)]:
        if vol.shape[0] <= 0:
            raise RuntimeError(f"empty volume after preprocessing: {series_dir} ({_name})")

    if left_vol.shape[0] >= k_out:
        left_vol = left_vol[:k_out]
        right_vol = right_vol[:k_out]
        slice_indices_out = slice_indices_out[:k_out]
        if z_out is not None:
            z_out = z_out[:k_out]
    else:
        while left_vol.shape[0] < k_out:
            left_vol = np.concatenate([left_vol, left_vol[-1:, ...]], axis=0)
            right_vol = np.concatenate([right_vol, right_vol[-1:, ...]], axis=0)
            slice_indices_out.append(int(slice_indices_out[-1]) if slice_indices_out else -1)
            if z_out is not None and len(z_out) > 0:
                z_out.append(float(z_out[-1]))

    # Window HU to [0,1] for model input.
    left_unit = _window_hu_to_unit(left_vol, wl=float(spec.window_wl), ww=float(spec.window_ww)).astype(np.float32)
    right_unit = _window_hu_to_unit(right_vol, wl=float(spec.window_wl), ww=float(spec.window_ww)).astype(np.float32)

    # Optional second window (e.g., soft-tissue).
    has_w2 = spec.window2_ww is not None and float(spec.window2_ww) > 0
    if has_w2:
        wl2 = float(spec.window2_wl) if spec.window2_wl is not None else 0.0
        ww2 = float(spec.window2_ww)
        left_unit2 = _window_hu_to_unit(left_vol, wl=wl2, ww=ww2).astype(np.float32)
        right_unit2 = _window_hu_to_unit(right_vol, wl=wl2, ww=ww2).astype(np.float32)
        left_ch = np.stack([left_unit, left_unit2], axis=0)  # (C,K,H,W)
        right_ch = np.stack([right_unit, right_unit2], axis=0)
    else:
        left_ch = left_unit[None, ...].astype(np.float32)
        right_ch = right_unit[None, ...].astype(np.float32)

    # Optional pair features (Difference Map) to approximate bilateral comparison.
    pair_mode = str(spec.pair_features or "none")
    if pair_mode not in {"none", "self_other_diff"}:
        raise ValueError(f"unknown pair_features: {pair_mode!r} (expected none|self_other_diff)")
    if pair_mode == "self_other_diff":
        diff = np.abs(left_ch - right_ch).astype(np.float32)
        left_in = np.concatenate([left_ch, right_ch, diff], axis=0)
        right_in = np.concatenate([right_ch, left_ch, diff], axis=0)
        arr = np.stack([left_in, right_in], axis=0).astype(np.float16)  # (2,3C,K,H,W)
    else:
        arr = np.stack([left_ch, right_ch], axis=0).astype(np.float16)  # (2,C,K,H,W)

    meta = {
        "series_relpath": str(series_relpath),
        "n_files": int(n_files),
        "slice_indices": [int(x) for x in slice_indices_out],
        "midline_x": int(midline) if midline is not None else -1,
        "left_center_yx": [int(left_center[0]), int(left_center[1])],
        "right_center_yx": [int(right_center[0]), int(right_center[1])],
        "spacing_y": float(spacing_y) if spacing_y is not None and np.isfinite(float(spacing_y)) else -1.0,
        "spacing_x": float(spacing_x) if spacing_x is not None and np.isfinite(float(spacing_x)) else -1.0,
        "target_spacing": float(target_spacing) if target_spacing is not None and np.isfinite(float(target_spacing)) else 0.0,
        "target_z_spacing": float(target_z_spacing) if target_z_spacing is not None and np.isfinite(float(target_z_spacing)) else 0.0,
        "z_mm": [float(x) for x in z_out] if z_out is not None else [float(-1.0) for _ in range(int(spec.num_slices))],
        "spec": {
            "num_slices": int(spec.num_slices),
            "image_size": int(spec.image_size),
            "crop_size": int(spec.crop_size),
            "sampling": str(spec.sampling),
            "block_len": int(spec.block_len),
            "flip_right": bool(spec.flip_right),
            "crop_mode": str(spec.crop_mode),
            "crop_lateral_band_frac": float(spec.crop_lateral_band_frac),
            "crop_lateral_bias": float(spec.crop_lateral_bias),
            "crop_min_area": int(spec.crop_min_area),
            "pad_hu": float(spec.pad_hu),
            "window_wl": float(spec.window_wl),
            "window_ww": float(spec.window_ww),
            "window2_wl": float(spec.window2_wl) if spec.window2_wl is not None and np.isfinite(float(spec.window2_wl)) else 0.0,
            "window2_ww": float(spec.window2_ww) if spec.window2_ww is not None and np.isfinite(float(spec.window2_ww)) else 0.0,
            "pair_features": str(spec.pair_features),
            "target_spacing": float(target_spacing) if target_spacing is not None and np.isfinite(float(target_spacing)) else 0.0,
            "target_z_spacing": float(target_z_spacing) if target_z_spacing is not None and np.isfinite(float(target_z_spacing)) else 0.0,
            "version": str(spec.version),
        },
    }

    return arr, meta


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
        spec: DualPreprocessSpec | None = None,
        num_slices: int = 32,
        image_size: int = 224,
        crop_size: int = 192,
        sampling: SamplingMode = "air_block",
        block_len: int = 64,
        flip_right: bool = True,
        target_spacing: float | None = 0.7,
        target_z_spacing: float | None = 0.8,
        cache_dir: Path | None = None,
        cache_dtype: str = "float16",
        return_image: bool = True,
    ) -> None:
        self.index = index_df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.spec = spec or DualPreprocessSpec(
            num_slices=int(num_slices),
            image_size=int(image_size),
            crop_size=int(crop_size),
            sampling=str(sampling) if sampling else "air_block",
            block_len=int(block_len),
            flip_right=bool(flip_right),
            target_spacing=float(target_spacing) if target_spacing is not None else None,
            target_z_spacing=float(target_z_spacing) if target_z_spacing is not None else None,
        )
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_dtype = str(cache_dtype)
        self.return_image = bool(return_image)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.index)

    def _cache_path(self, *, exam_id: int, series_relpath: str) -> Path:
        h = self.spec.cache_key(series_relpath=str(series_relpath))
        return self.cache_dir / f"{exam_id}_{h}.npy"

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        series_dir = self.dicom_root / str(row["series_relpath"])
        exam_id = int(row["exam_id"])
        series_relpath = str(row["series_relpath"])
        report_text = row.get("report_text") if hasattr(row, "get") else None
        if report_text is not None and (isinstance(report_text, float) and pd.isna(report_text)):
            report_text = None
        report_text_s = str(report_text) if report_text is not None else ""

        left_label, left_present = _code_to_label(row.get("left_code"))
        right_label, right_present = _code_to_label(row.get("right_code"))

        if self.cache_dir is not None:
            cache_path = self._cache_path(exam_id=exam_id, series_relpath=series_relpath)
            npz_path = cache_path.with_suffix(".npz")
            if cache_path.exists() or npz_path.exists():
                meta = {"exam_id": exam_id, "date": str(row["date"]), "series_relpath": series_relpath, "report_text": report_text_s}
                label = torch.tensor([left_label, right_label], dtype=torch.long)
                label_mask = torch.tensor([left_present, right_present], dtype=torch.bool)
                if self.return_image:
                    if cache_path.exists():
                        arr = np.load(cache_path)
                    else:
                        with np.load(npz_path) as z:
                            arr = z["arr"]
                    image = torch.from_numpy(arr).to(torch.float32)
                    meta_path = cache_path.with_suffix(".json")
                    if meta_path.exists():
                        try:
                            m2 = json.loads(meta_path.read_text(encoding="utf-8"))
                            if isinstance(m2, dict):
                                meta.update(m2)
                        except Exception:
                            pass
                    return {"image": image, "label": label, "label_mask": label_mask, "report_text": report_text_s, "meta": meta}
                return {"label": label, "label_mask": label_mask, "report_text": report_text_s, "meta": meta}

        arr, meta2 = build_dual_unit_volume(series_dir=series_dir, series_relpath=series_relpath, spec=self.spec)
        image = torch.from_numpy(arr.astype(np.float32))
        meta = {
            "exam_id": exam_id,
            "date": str(row["date"]),
            "series_relpath": series_relpath,
            "report_text": report_text_s,
        }
        meta.update(meta2)

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
                meta_path = cache_path.with_suffix(".json")
                meta_path.write_text(json.dumps(meta2, ensure_ascii=False) + "\n", encoding="utf-8")
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        if self.return_image:
            return {"image": image, "label": label, "label_mask": label_mask, "report_text": report_text_s, "meta": meta}
        return {"label": label, "label_mask": label_mask, "report_text": report_text_s, "meta": meta}
