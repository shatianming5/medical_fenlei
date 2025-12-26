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

from medical_fenlei.data.dicom import list_dicom_files_ipp, read_dicom_hu, read_dicom_hu_and_spacing, read_dicom_hu_and_spacing_and_z, read_dicom_hu_and_z


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

CropMode = Literal["bbox_bias", "temporal_patch"]


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
    target_spacing: float | None = None
    target_z_spacing: float | None = None
    crop_mode: CropMode = "temporal_patch"
    crop_lateral_band_frac: float = 0.6
    crop_lateral_bias: float = 0.25
    crop_min_area: int = 300
    version: str = "v3"

    def cache_key(self, *, series_relpath: str, side: str) -> str:
        ts = self.target_spacing
        ts_s = f"|ts={float(ts):.6g}" if ts is not None and float(ts) > 0 else ""
        tz = self.target_z_spacing
        tz_s = f"|tz={float(tz):.6g}" if tz is not None and float(tz) > 0 else ""
        s = (
            f"{series_relpath}|side={side}|d={int(self.num_slices)}|s={int(self.image_size)}|c={int(self.crop_size)}|"
            f"sampling={self.sampling}|block={int(self.block_len)}|flip={int(self.flip_right)}|midK={int(self.midline_slices)}|"
            f"crop={self.crop_mode}|band={float(self.crop_lateral_band_frac):.3g}|bias={float(self.crop_lateral_bias):.3g}|"
            f"minA={int(self.crop_min_area)}|pad={float(self.pad_hu)}{ts_s}{tz_s}|{self.version}"
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

    # Split into one side stack (uncropped).
    side_imgs: list[np.ndarray] = []
    for hu in hu_slices:
        left, right = _split_left_right(hu, mid_x=midline, flip_right=bool(spec.flip_right))
        img = left if side == "left" else right
        side_imgs.append(img.astype(np.float32))

    # Infer one crop center for the whole stack to reduce jitter.
    crop_center = _infer_stack_center(
        side_imgs,
        crop_mode=spec.crop_mode,
        lateral_bias=float(spec.crop_lateral_bias),
        lateral_band_frac=float(spec.crop_lateral_band_frac),
        min_area=int(spec.crop_min_area),
        max_slices=max(3, int(spec.midline_slices)),
    )
    if crop_center is None:
        h0, w0 = side_imgs[0].shape
        crop_center = (h0 // 2, w0 // 2)

    cropped_stack: list[np.ndarray] = []
    for img in side_imgs:
        if str(spec.crop_mode) == "bbox_bias":
            cropped = _crop_square(img, crop_size=int(spec.crop_size), pad_value=float(spec.pad_hu), lateral_bias=float(spec.crop_lateral_bias))
        else:
            cropped = _crop_square_at(
                img,
                center_y=int(crop_center[0]),
                center_x=int(crop_center[1]),
                crop_size=int(spec.crop_size),
                pad_value=float(spec.pad_hu),
            )
        t = torch.from_numpy(cropped[None, None, ...])  # (1,1,H,W)
        t = F.interpolate(t, size=(int(spec.image_size), int(spec.image_size)), mode="bilinear", align_corners=False)[0, 0]
        cropped_stack.append(t.numpy().astype(np.float32))

    vol_np = np.stack(cropped_stack, axis=0).astype(np.float32)  # (D,H,W)

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
                # enforce monotonicity if small inversions exist
                zs_np = np.maximum.accumulate(zs_np)
        elif spacing_z_est is not None and np.isfinite(float(spacing_z_est)) and float(spacing_z_est) > 0:
            zs_np = np.arange(zs_np.size, dtype=np.float64) * float(spacing_z_est)
            valid = np.ones_like(zs_np, dtype=bool)

        if valid.sum() >= 2:
            # Center around the middle of the loaded neighborhood.
            z_center = float(zs_np[len(zs_np) // 2])
            k = int(spec.num_slices)
            offs = (np.arange(k, dtype=np.float64) - (float(k - 1) / 2.0)) * float(target_z_spacing)
            z_tgt = z_center + offs

            out_slices: list[np.ndarray] = []
            out_nearest: list[int] = []
            for zt in z_tgt.tolist():
                idx = int(np.searchsorted(zs_np, float(zt), side="left"))
                if idx <= 0:
                    out_slices.append(vol_np[0])
                    out_nearest.append(int(read_indices[0]))
                    continue
                if idx >= len(zs_np):
                    out_slices.append(vol_np[-1])
                    out_nearest.append(int(read_indices[-1]))
                    continue

                z0 = float(zs_np[idx - 1])
                z1 = float(zs_np[idx])
                if not np.isfinite(z0) or not np.isfinite(z1) or abs(z1 - z0) < 1e-6:
                    out_slices.append(vol_np[idx])
                    out_nearest.append(int(read_indices[idx]))
                    continue
                a = float((float(zt) - z0) / (z1 - z0))
                a = float(min(1.0, max(0.0, a)))
                out_slices.append((1.0 - a) * vol_np[idx - 1] + a * vol_np[idx])
                out_nearest.append(int(read_indices[idx] if a > 0.5 else read_indices[idx - 1]))

            vol_np = np.stack(out_slices, axis=0).astype(np.float32)
            slice_indices_out = out_nearest
            z_out = [float(x) for x in z_tgt.tolist()]

    # Final: trim/pad to K slices and store as float16 HU.
    k_out = int(spec.num_slices)
    if vol_np.shape[0] >= k_out:
        vol_np = vol_np[:k_out]
        slice_indices_out = slice_indices_out[:k_out]
        if z_out is not None:
            z_out = z_out[:k_out]
    else:
        while vol_np.shape[0] < k_out:
            vol_np = np.concatenate([vol_np, vol_np[-1:, ...]], axis=0)
            slice_indices_out.append(int(slice_indices_out[-1]) if slice_indices_out else -1)
            if z_out is not None and len(z_out) > 0:
                z_out.append(float(z_out[-1]))

    arr = vol_np.astype(np.float16)  # (K,H,W) HU

    # Make meta collate-friendly (no None).
    if z_out is None:
        z_mm = [float(-1.0) for _ in range(k_out)]
    else:
        z_mm = [float(x) if x is not None and np.isfinite(float(x)) else float(-1.0) for x in list(z_out)]
        if len(z_mm) < k_out:
            z_mm = list(z_mm) + [float(z_mm[-1] if z_mm else -1.0) for _ in range(k_out - len(z_mm))]
        elif len(z_mm) > k_out:
            z_mm = z_mm[:k_out]

    crop_center_yx = [int(crop_center[0]), int(crop_center[1])] if crop_center is not None else [-1, -1]
    spacing_y_f = float(spacing_y) if spacing_y is not None and np.isfinite(float(spacing_y)) else -1.0
    spacing_x_f = float(spacing_x) if spacing_x is not None and np.isfinite(float(spacing_x)) else -1.0
    target_spacing_f = float(target_spacing) if target_spacing is not None and np.isfinite(float(target_spacing)) else 0.0
    target_z_spacing_f = float(target_z_spacing) if target_z_spacing is not None and np.isfinite(float(target_z_spacing)) else 0.0

    meta = {
        "exam_id": int(exam_id),
        "series_relpath": str(series_relpath),
        "side": str(side),
        "slice_indices": [int(x) for x in slice_indices_out],
        "z_mm": z_mm,
        "midline_x": int(midline) if midline is not None else -1,
        "n_files": int(len(files)),
        "crop_center_yx": crop_center_yx,
        "spacing_y": spacing_y_f,
        "spacing_x": spacing_x_f,
        "target_spacing": target_spacing_f,
        "target_z_spacing": target_z_spacing_f,
    }
    return arr, meta


def _sanitize_meta_for_collate(meta: dict, *, num_slices: int) -> dict:
    """
    DataLoader default_collate cannot handle None; sanitize known meta fields.
    """
    k = int(num_slices)
    k = max(1, k)

    def _as_float(v, *, default: float) -> float:
        if v is None:
            return float(default)
        try:
            x = float(v)
            if not np.isfinite(x):
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _as_int(v, *, default: int) -> int:
        if v is None:
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _int_list(v, *, length: int, default: int) -> list[int]:
        if not isinstance(v, (list, tuple)):
            return [int(default) for _ in range(int(length))]
        out: list[int] = []
        for x in list(v):
            out.append(_as_int(x, default=int(default)))
        if len(out) < int(length):
            out.extend([int(default) for _ in range(int(length) - len(out))])
        return out[: int(length)]

    def _float_list(v, *, length: int, default: float) -> list[float]:
        if not isinstance(v, (list, tuple)):
            return [float(default) for _ in range(int(length))]
        out: list[float] = []
        for x in list(v):
            out.append(_as_float(x, default=float(default)))
        if len(out) < int(length):
            out.extend([float(out[-1] if out else default) for _ in range(int(length) - len(out))])
        return out[: int(length)]

    meta["exam_id"] = _as_int(meta.get("exam_id"), default=-1)
    meta["label_code"] = _as_int(meta.get("label_code"), default=-1)
    meta["midline_x"] = _as_int(meta.get("midline_x"), default=-1)
    meta["n_files"] = _as_int(meta.get("n_files"), default=-1)
    meta["spacing_y"] = _as_float(meta.get("spacing_y"), default=-1.0)
    meta["spacing_x"] = _as_float(meta.get("spacing_x"), default=-1.0)
    meta["target_spacing"] = _as_float(meta.get("target_spacing"), default=0.0)
    meta["target_z_spacing"] = _as_float(meta.get("target_z_spacing"), default=0.0)

    meta["slice_indices"] = _int_list(meta.get("slice_indices"), length=k, default=-1)
    meta["z_mm"] = _float_list(meta.get("z_mm"), length=k, default=-1.0)
    meta["crop_center_yx"] = _int_list(meta.get("crop_center_yx"), length=2, default=-1)[:2]

    # Ensure strings are not None.
    meta["side"] = str(meta.get("side") or "")
    meta["series_relpath"] = str(meta.get("series_relpath") or "")
    return meta


class EarCTHUEarDataset(Dataset):
    """
    Ear-level dataset that returns HU volumes.

    Expected columns in index_df:
      - exam_id, series_relpath, side
      - y (binary 0/1 or multi-class id)
    """

    def __init__(
        self,
        *,
        index_df: pd.DataFrame,
        dicom_root: Path,
        spec: EarPreprocessSpec,
        cache_dir: Path | None = None,
        return_meta: bool = True,
        y_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.index = index_df.reset_index(drop=True)
        self.dicom_root = Path(dicom_root)
        self.spec = spec
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.return_meta = bool(return_meta)
        self.y_dtype = y_dtype

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
        if self.y_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long, torch.uint8):
            y = int(row["y"])
        else:
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
            "crop_center_yx": [-1, -1],
            "spacing_y": -1.0,
            "spacing_x": -1.0,
            "target_spacing": 0.0,
            "target_z_spacing": 0.0,
            "z_mm": [-1.0 for _ in range(int(self.spec.num_slices))],
        }

        if self.cache_dir is not None:
            cache_path = self._cache_path(exam_id=exam_id, series_relpath=series_relpath, side=side)
            npz_path = cache_path.with_suffix(".npz")
            if cache_path.exists() or npz_path.exists():
                if cache_path.exists():
                    arr = np.load(cache_path)  # (K,H,W) float16 HU
                else:
                    with np.load(npz_path) as z:
                        arr = z["arr"]
                hu = torch.from_numpy(arr.astype(np.float32))
                out = {"hu": hu, "y": torch.tensor(y, dtype=self.y_dtype)}
                if self.return_meta:
                    meta_path = cache_path.with_suffix(".json")
                    if meta_path.exists():
                        try:
                            meta2 = json.loads(meta_path.read_text(encoding="utf-8"))
                            if isinstance(meta2, dict):
                                meta.update(meta2)
                        except Exception:
                            pass
                    out["meta"] = _sanitize_meta_for_collate(meta, num_slices=int(self.spec.num_slices))
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
        out = {"hu": hu, "y": torch.tensor(y, dtype=self.y_dtype)}
        if self.return_meta:
            meta.update(meta2)
            out["meta"] = _sanitize_meta_for_collate(meta, num_slices=int(self.spec.num_slices))
        return out
