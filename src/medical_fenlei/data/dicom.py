from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


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
    return img.astype(np.float32)


def read_dicom_image(path: Path, *, window: Window | None = None) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    pixels = ds.pixel_array
    img = dicom_to_hu(ds, pixels)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    win = window or get_window(ds)
    img = np.clip(img, win.lower, win.upper)
    img = (img - win.lower) / (win.upper - win.lower + 1e-6)
    return img.astype(np.float32)
