from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


def _pixel_spacing(ds: pydicom.Dataset) -> tuple[float | None, float | None]:
    px = getattr(ds, "PixelSpacing", None)
    if px is None:
        return None, None
    try:
        if isinstance(px, (list, tuple)) and len(px) >= 2:
            sy = float(px[0])
            sx = float(px[1])
            if sy <= 0 or sx <= 0:
                return None, None
            return sy, sx
    except Exception:
        return None, None
    return None, None


def _canonicalize_inplane_orientation_with_spacing(
    img: np.ndarray, ds: pydicom.Dataset, *, spacing_y: float | None, spacing_x: float | None
) -> tuple[np.ndarray, float | None, float | None]:
    if img.ndim != 2:
        return img, spacing_y, spacing_x

    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        return img, spacing_y, spacing_x
    try:
        vals = [float(x) for x in list(iop)[:6]]
    except Exception:
        return img, spacing_y, spacing_x
    if len(vals) < 6:
        return img, spacing_y, spacing_x

    x_dir = np.asarray(vals[:3], dtype=np.float64)
    y_dir = np.asarray(vals[3:6], dtype=np.float64)

    # Skip strongly oblique slices.
    if abs(float(x_dir[2])) > 0.2 or abs(float(y_dir[2])) > 0.2:
        return img, spacing_y, spacing_x

    x2 = x_dir[:2]
    y2 = y_dir[:2]
    ax_x = int(np.argmax(np.abs(x2)))  # 0=patient X, 1=patient Y
    ax_y = int(np.argmax(np.abs(y2)))
    if ax_x == ax_y:
        return img, spacing_y, spacing_x

    out = img
    transposed = False
    if ax_x == 1 and ax_y == 0:
        out = out.T
        x2, y2 = y2, x2
        ax_x, ax_y = ax_y, ax_x
        transposed = True

    if ax_x != 0 or ax_y != 1:
        if transposed:
            out = out.copy()
        return out, (spacing_x if transposed else spacing_y), (spacing_y if transposed else spacing_x)

    if float(x2[0]) < 0:
        out = np.fliplr(out)
    if float(y2[1]) < 0:
        out = np.flipud(out)

    if transposed:
        spacing_y, spacing_x = spacing_x, spacing_y
    return out.copy(), spacing_y, spacing_x


def _canonicalize_inplane_orientation(img: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """
    Canonicalize in-plane orientation using ImageOrientationPatient (IOP).

    Target (LPS):
      - columns increase towards patient Left  (+X)
      - rows    increase towards patient Posterior (+Y)

    Handles pure flips and 90-degree rotations (transpose).
    Falls back to the input when IOP is missing/unsupported.
    """
    out, _, _ = _canonicalize_inplane_orientation_with_spacing(img, ds, spacing_y=None, spacing_x=None)
    return out


def _last_numeric_token(path: Path) -> int:
    # Most files are named like "<SOPInstanceUID>.dcm", and the UID ends with a
    # numeric token that is monotonic per slice in this dataset.
    stem = path.stem
    try:
        return int(stem.split(".")[-1])
    except Exception:
        return 0


def list_dicom_files(series_dir: Path) -> list[Path]:
    files = [p for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
    files.sort(key=_last_numeric_token)
    return files


def list_dicom_files_ipp(series_dir: Path) -> list[Path]:
    """
    List DICOM files sorted by z position (ImagePositionPatient) when available.

    Falls back to InstanceNumber, then filename heuristic.
    """
    files = [p for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
    if not files:
        return []

    def _key(p: Path) -> tuple[int, float, int]:
        # smaller rank = better key source
        try:
            ds = pydicom.dcmread(
                str(p),
                force=True,
                stop_before_pixels=True,
                specific_tags=["ImagePositionPatient", "InstanceNumber"],
            )
            ipp = getattr(ds, "ImagePositionPatient", None)
            if isinstance(ipp, (list, tuple)) and len(ipp) >= 3:
                try:
                    return (0, float(ipp[2]), int(getattr(ds, "InstanceNumber", 0) or 0))
                except Exception:
                    pass
            inst = getattr(ds, "InstanceNumber", None)
            if inst is not None:
                try:
                    return (1, float(int(inst)), 0)
                except Exception:
                    pass
        except TypeError:
            # Older pydicom may not support specific_tags.
            try:
                ds = pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
                ipp = getattr(ds, "ImagePositionPatient", None)
                if isinstance(ipp, (list, tuple)) and len(ipp) >= 3:
                    return (0, float(ipp[2]), int(getattr(ds, "InstanceNumber", 0) or 0))
                inst = getattr(ds, "InstanceNumber", None)
                if inst is not None:
                    return (1, float(int(inst)), 0)
            except Exception:
                pass
        except Exception:
            pass

        return (2, float(_last_numeric_token(p)), 0)

    files.sort(key=_key)
    return files


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        # pydicom can return MultiValue
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        return float(value)
    except Exception:
        return None
    return None


def _ipp_z(ds: pydicom.Dataset) -> float | None:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None:
        return None
    try:
        if isinstance(ipp, (list, tuple)) and len(ipp) >= 3:
            return float(ipp[2])
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class Window:
    center: float
    width: float

    @property
    def lower(self) -> float:
        return self.center - self.width / 2.0

    @property
    def upper(self) -> float:
        return self.center + self.width / 2.0


def get_window(ds: pydicom.Dataset, *, default: Window = Window(center=500.0, width=3000.0)) -> Window:
    center = _as_float(getattr(ds, "WindowCenter", None))
    width = _as_float(getattr(ds, "WindowWidth", None))
    if center is None or width is None or width <= 0:
        return default
    return Window(center=center, width=width)


def dicom_to_hu(ds: pydicom.Dataset, pixels: np.ndarray) -> np.ndarray:
    slope = _as_float(getattr(ds, "RescaleSlope", 1.0)) or 1.0
    intercept = _as_float(getattr(ds, "RescaleIntercept", 0.0)) or 0.0
    return pixels.astype(np.float32) * slope + intercept


def read_dicom_hu(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    pixels = ds.pixel_array
    img = dicom_to_hu(ds, pixels)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img = _canonicalize_inplane_orientation(img, ds)
    return img.astype(np.float32)


def read_dicom_hu_and_z(path: Path) -> tuple[np.ndarray, float | None]:
    ds = pydicom.dcmread(str(path), force=True)
    z = _ipp_z(ds)
    pixels = ds.pixel_array
    img = dicom_to_hu(ds, pixels)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img = _canonicalize_inplane_orientation(img, ds)
    return img.astype(np.float32), z


def read_dicom_hu_and_spacing(path: Path) -> tuple[np.ndarray, tuple[float | None, float | None]]:
    ds = pydicom.dcmread(str(path), force=True)
    spacing_y, spacing_x = _pixel_spacing(ds)
    pixels = ds.pixel_array
    img = dicom_to_hu(ds, pixels)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img, spacing_y, spacing_x = _canonicalize_inplane_orientation_with_spacing(img, ds, spacing_y=spacing_y, spacing_x=spacing_x)
    return img.astype(np.float32), (spacing_y, spacing_x)


def read_dicom_hu_and_spacing_and_z(path: Path) -> tuple[np.ndarray, tuple[float | None, float | None], float | None]:
    ds = pydicom.dcmread(str(path), force=True)
    z = _ipp_z(ds)
    spacing_y, spacing_x = _pixel_spacing(ds)
    pixels = ds.pixel_array
    img = dicom_to_hu(ds, pixels)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img, spacing_y, spacing_x = _canonicalize_inplane_orientation_with_spacing(img, ds, spacing_y=spacing_y, spacing_x=spacing_x)
    return img.astype(np.float32), (spacing_y, spacing_x), z


def read_dicom_image(path: Path, *, window: Window | None = None) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    pixels = ds.pixel_array
    img = dicom_to_hu(ds, pixels)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img = _canonicalize_inplane_orientation(img, ds)

    win = window or get_window(ds)
    img = np.clip(img, win.lower, win.upper)
    img = (img - win.lower) / (win.upper - win.lower + 1e-6)
    return img.astype(np.float32)


def read_dicom_image_and_spacing(path: Path, *, window: Window | None = None) -> tuple[np.ndarray, tuple[float | None, float | None]]:
    ds = pydicom.dcmread(str(path), force=True)
    spacing_y, spacing_x = _pixel_spacing(ds)
    pixels = ds.pixel_array
    img = dicom_to_hu(ds, pixels)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img, spacing_y, spacing_x = _canonicalize_inplane_orientation_with_spacing(img, ds, spacing_y=spacing_y, spacing_x=spacing_x)

    win = window or get_window(ds)
    img = np.clip(img, win.lower, win.upper)
    img = (img - win.lower) / (win.upper - win.lower + 1e-6)
    return img.astype(np.float32), (spacing_y, spacing_x)
