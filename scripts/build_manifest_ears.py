from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import typer

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.paths import infer_dicom_root
from medical_fenlei.tasks import TASKS

app = typer.Typer(add_completion=False)


def _sha1(text: str, *, salt: str) -> str:
    h = hashlib.sha1()
    h.update((salt + text).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    try:
        s = str(v).strip()
    except Exception:
        return None
    return s or None


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        # pydicom MultiValue -> take first
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        return float(v)
    except Exception:
        return None


def _read_series_meta(series_dir: Path, *, hash_salt: str, store_phi: bool) -> dict[str, Any]:
    files = sorted([p for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"])
    if not files:
        return {"meta_ok": False}

    first = files[0]
    tags = [
        "PatientID",
        "PatientName",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Manufacturer",
        "ManufacturerModelName",
        "ConvolutionKernel",
        "Modality",
        "BodyPartExamined",
        "PixelSpacing",
        "SliceThickness",
        "SpacingBetweenSlices",
        "Rows",
        "Columns",
        "ImageOrientationPatient",
    ]
    try:
        ds = pydicom.dcmread(str(first), force=True, stop_before_pixels=True, specific_tags=tags)
    except TypeError:
        # Older pydicom may not support specific_tags.
        ds = pydicom.dcmread(str(first), force=True, stop_before_pixels=True)
    except Exception:
        return {"meta_ok": False}

    patient_id = _as_str(getattr(ds, "PatientID", None))
    patient_name = _as_str(getattr(ds, "PatientName", None))
    study_uid = _as_str(getattr(ds, "StudyInstanceUID", None))
    series_uid = _as_str(getattr(ds, "SeriesInstanceUID", None))

    patient_id_hash = _sha1(patient_id, salt=hash_salt) if patient_id else None
    patient_name_hash = _sha1(patient_name, salt=hash_salt) if patient_name else None

    patient_key_hash = patient_id_hash or patient_name_hash or (_sha1(study_uid, salt=hash_salt) if study_uid else None)

    px = getattr(ds, "PixelSpacing", None)
    spacing_y = None
    spacing_x = None
    try:
        if isinstance(px, (list, tuple)) and len(px) >= 2:
            spacing_y = _as_float(px[0])
            spacing_x = _as_float(px[1])
    except Exception:
        spacing_y = None
        spacing_x = None

    thickness = _as_float(getattr(ds, "SliceThickness", None))
    spacing_between = _as_float(getattr(ds, "SpacingBetweenSlices", None))
    spacing_z = spacing_between if spacing_between is not None else thickness

    iop = getattr(ds, "ImageOrientationPatient", None)
    iop_s = None
    try:
        if isinstance(iop, (list, tuple)) and len(iop) >= 6:
            iop_s = ",".join(f"{float(x):.6g}" for x in iop[:6])
    except Exception:
        iop_s = None

    out: dict[str, Any] = {
        "meta_ok": True,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "patient_id_hash": patient_id_hash,
        "patient_name_hash": patient_name_hash,
        "patient_key_hash": patient_key_hash,
        "manufacturer": _as_str(getattr(ds, "Manufacturer", None)),
        "manufacturer_model": _as_str(getattr(ds, "ManufacturerModelName", None)),
        "convolution_kernel": _as_str(getattr(ds, "ConvolutionKernel", None)),
        "modality": _as_str(getattr(ds, "Modality", None)),
        "body_part": _as_str(getattr(ds, "BodyPartExamined", None)),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "spacing_z": spacing_z,
        "slice_thickness": thickness,
        "spacing_between_slices": spacing_between,
        "image_orientation": iop_s,
    }

    if store_phi:
        out["patient_id"] = patient_id
        out["patient_name"] = patient_name

    return out


def _expand_ears(index_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in index_df.itertuples(index=False):
        for side in ("left", "right"):
            code = getattr(r, f"{side}_code", None)
            has_label = not (code is None or (isinstance(code, float) and np.isnan(code)))
            code_i = int(code) if has_label else None
            label_id = int(code_i - 1) if code_i is not None and 1 <= code_i <= 6 else -1
            rows.append(
                {
                    "exam_id": int(r.exam_id),
                    "side": side,
                    "label_code": code_i,
                    "label_id": label_id,
                    "has_label": bool(has_label),
                    "date": str(getattr(r, "date", "")),
                    "label_date": str(getattr(r, "label_date", "")),
                    "folder_date": str(getattr(r, "folder_date", "")),
                    "date_match": bool(getattr(r, "date_match", False)),
                    "ambiguous_match": bool(getattr(r, "ambiguous_match", False)),
                    "series_relpath": str(getattr(r, "series_relpath", "")),
                    "n_instances": int(getattr(r, "n_instances", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


@app.command()
def main(
    index_csv: Path = typer.Option(Path("artifacts/dataset_index.csv"), exists=True, help="由 scripts/build_index.py 生成（不入库）"),
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录（会自动推断 dicom_root）"),
    out_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), help="耳朵级 manifest（不入库）"),
    num_workers: int = typer.Option(16, help="并行读取 DICOM header 的线程数"),
    hash_salt: str = typer.Option("", help="可选：对 patient_id/name 做 hash 的 salt（留空也可）"),
    store_phi: bool = typer.Option(False, help="是否把 PatientID/PatientName 原文写入 manifest（默认不写）"),
) -> None:
    dicom_root = infer_dicom_root(dicom_base)
    index_df = pd.read_csv(index_csv)
    if index_df.empty:
        raise typer.Exit(code=2)

    ears = _expand_ears(index_df)
    if ears.empty:
        raise typer.Exit(code=2)

    # Build per-series metadata (one header read per exam/series).
    series_relpaths = sorted(set(ears["series_relpath"].astype(str).tolist()))
    meta_map: dict[str, dict[str, Any]] = {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(num_workers)) as ex:
        futs = {}
        for rel in series_relpaths:
            series_dir = dicom_root / rel
            futs[ex.submit(_read_series_meta, series_dir, hash_salt=str(hash_salt), store_phi=bool(store_phi))] = rel

        done = 0
        for fut in as_completed(futs):
            rel = futs[fut]
            try:
                meta_map[rel] = fut.result()
            except Exception:
                meta_map[rel] = {"meta_ok": False}
            done += 1
            if done % 200 == 0:
                typer.echo(f"meta: {done}/{len(series_relpaths)}")

    typer.echo(f"meta done: {len(meta_map)} series  ({time.time()-t0:.1f}s)")

    meta_df = pd.DataFrame.from_records(
        [
            {"series_relpath": rel, **meta_map.get(rel, {"meta_ok": False})}
            for rel in series_relpaths
        ]
    )
    out = ears.merge(meta_df, on="series_relpath", how="left")

    # Derived flags (non-PHI)
    out["is_normal"] = out["label_code"] == 5
    out["is_other6"] = out["label_code"] == 6
    out["is_code4"] = out["label_code"] == 4
    out["is_abnormal_1_4"] = out["label_code"].isin([1, 2, 3, 4])
    out["is_abnormal_all"] = out["label_code"].isin([1, 2, 3, 4, 6])

    # Task masks / mapped labels.
    for name, task in TASKS.items():
        if task.kind != "binary":
            continue
        rel_codes = task.relevant_codes()
        out[f"task_valid__{name}"] = out["label_code"].isin(sorted(rel_codes))
        pos_codes = set(task.pos_codes)
        neg_codes = set(task.neg_codes)
        y = pd.Series([None] * len(out), dtype="float")
        y[out["label_code"].isin(sorted(neg_codes))] = 0.0
        y[out["label_code"].isin(sorted(pos_codes))] = 1.0
        out[f"task_y__{name}"] = y

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": os.getcwd(),
        "dicom_root": str(dicom_root),
        "index_csv": str(index_csv),
        "rows": int(len(out)),
        "labeled_ears": int(out["has_label"].sum()),
        "tasks_binary": [k for k, v in TASKS.items() if v.kind == "binary"],
        "class_id_to_name": {int(k): str(v) for k, v in CLASS_ID_TO_NAME.items()},
        "note": "This file is gitignored; do not commit PHI.",
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"saved: {out_csv}")
    typer.echo(f"saved: {out_csv.with_suffix('.meta.json')}")


if __name__ == "__main__":
    app()

