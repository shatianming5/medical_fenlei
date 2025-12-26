from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.data.dual_dataset import EarCTDualDataset
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _norm_code(v) -> int:
    if v is None:
        return -1
    if isinstance(v, float) and np.isnan(v):
        return -1
    try:
        return int(v)
    except Exception:
        return -1


def _pair_key(left_code, right_code) -> str:
    return f"{_norm_code(left_code)}_{_norm_code(right_code)}"


def _mad_mask(dist: np.ndarray, *, mad_mult: float, max_remove_frac: float) -> tuple[np.ndarray, dict]:
    if dist.size <= 0:
        return np.zeros((0,), dtype=bool), {"median": None, "mad": None, "threshold": None}

    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    if not np.isfinite(mad) or mad <= 1e-12:
        return np.zeros_like(dist, dtype=bool), {"median": med, "mad": mad, "threshold": None}

    thr = float(med + float(mad_mult) * mad)
    mask = dist > thr

    if max_remove_frac is not None and float(max_remove_frac) > 0:
        k = int(np.floor(float(max_remove_frac) * float(dist.size)))
        if k > 0 and int(mask.sum()) > k:
            order = np.argsort(dist)[::-1]  # high -> low
            mask2 = np.zeros_like(mask)
            mask2[order[:k]] = True
            mask = mask2
            thr = float(dist[order[k - 1]])

    return mask, {"median": med, "mad": mad, "threshold": thr}


@torch.inference_mode()
def _extract_features(
    df: pd.DataFrame,
    *,
    dicom_root: Path,
    cache_dir: Path,
    cache_dtype: str,
    num_slices: int,
    image_size: int,
    pool_d: int,
    pool_hw: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> np.ndarray:
    used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}"

    ds = EarCTDualDataset(
        index_df=df,
        dicom_root=dicom_root,
        num_slices=num_slices,
        image_size=image_size,
        flip_right=True,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
        return_image=True,
    )

    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
    )

    # pooled feature dim = (2 ears) * (C channels) * pool_d * pool_hw * pool_hw
    sample0 = ds[0]["image"]
    c = int(sample0.shape[1]) if sample0.ndim == 5 else 1
    dim = int(2 * int(c) * int(pool_d) * int(pool_hw) * int(pool_hw))
    out = np.zeros((len(ds), dim), dtype=np.float32)

    step = 0
    i = 0
    total = (len(ds) + int(batch_size) - 1) // int(batch_size)
    for batch in tqdm(loader, total=total, desc="extract_features"):
        x = batch["image"]  # (B,2,1,D,H,W) float32 cpu
        b = int(x.shape[0])
        c = int(x.shape[2]) if x.ndim >= 6 else 1
        if device.type == "cuda":
            x = x.to(device, non_blocking=True)

        x2 = x.reshape(b * 2, int(c), int(num_slices), int(image_size), int(image_size))
        pooled = F.adaptive_avg_pool3d(x2, (int(pool_d), int(pool_hw), int(pool_hw))).reshape(b, -1)

        if device.type == "cuda":
            pooled = pooled.float().cpu()
        else:
            pooled = pooled.float()
        out[i : i + b] = pooled.numpy()

        i += b
        step += 1

    return out


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual_patient"), help="输入 split 根目录"),
    out_splits_root: Path = typer.Option(Path("artifacts/splits_dual_patient_clustered"), help="输出 split 根目录"),
    pcts: str = typer.Option("100", help="要处理的 pct，例如 1,20,100"),
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录（实际会优先走 cache）"),
    cache_dir: Path = typer.Option(Path("cache/dual_volumes"), help="dual 缓存目录（cache/dual_volumes）"),
    cache_dtype: str = typer.Option("float16", help="缓存数据类型：float16 | float32"),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    pool_d: int = typer.Option(4, help="特征提取的 D 方向池化输出"),
    pool_hw: int = typer.Option(8, help="特征提取的 H/W 池化输出"),
    batch_size: int = typer.Option(8, help="特征提取 batch_size（不影响训练）"),
    num_workers: int = typer.Option(16, help="特征提取 DataLoader workers（不影响训练）"),
    prefetch_factor: int = typer.Option(4),
    device: str = typer.Option("cuda", help="cuda | cpu（特征提取用）"),
    min_group_size: int = typer.Option(30, help="每个(左,右)组合类最少样本数，小于则不做剔除"),
    mad_mult: float = typer.Option(6.0, help="MAD 阈值倍数（越大越保守）"),
    max_remove_frac: float = typer.Option(0.05, help="每个组合类最多剔除比例（0=不限制）"),
    drop_val_outliers: bool = typer.Option(False, help="是否连 val 也剔除（默认不剔除，保持验证集可比）"),
) -> None:
    pct_list = _parse_int_list(pcts)
    if not pct_list:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    out_splits_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_splits_root / "outlier_cleaning_config.json"
    meta = {
        "splits_root": str(splits_root),
        "dicom_root": str(dicom_root),
        "cache_dir": str(cache_dir),
        "cache_dtype": str(cache_dtype),
        "num_slices": int(num_slices),
        "image_size": int(image_size),
        "pool_d": int(pool_d),
        "pool_hw": int(pool_hw),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "device": str(dev),
        "min_group_size": int(min_group_size),
        "mad_mult": float(mad_mult),
        "max_remove_frac": float(max_remove_frac),
        "drop_val_outliers": bool(drop_val_outliers),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"cache_dir: {cache_dir}  cache_dtype={cache_dtype}")
    typer.echo(f"device: {dev}")
    typer.echo(f"out_splits_root: {out_splits_root}")

    for pct in pct_list:
        split_dir = splits_root / f"{int(pct)}pct"
        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        if not train_csv.exists():
            raise FileNotFoundError(train_csv)
        if not val_csv.exists():
            raise FileNotFoundError(val_csv)

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        if train_df.empty or val_df.empty:
            raise typer.Exit(code=2)

        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["_split"] = "train"
        val_df["_split"] = "val"
        df = pd.concat([train_df, val_df], ignore_index=True)

        df["_left_code_int"] = df["left_code"].map(_norm_code)
        df["_right_code_int"] = df["right_code"].map(_norm_code)
        df["_pair_key"] = [
            _pair_key(l, r) for l, r in zip(df["_left_code_int"].tolist(), df["_right_code_int"].tolist(), strict=False)
        ]

        typer.echo(f"\n=== pct={pct}%  items(train={len(train_df)} val={len(val_df)}) ===")
        feats = _extract_features(
            df,
            dicom_root=dicom_root,
            cache_dir=cache_dir,
            cache_dtype=cache_dtype,
            num_slices=num_slices,
            image_size=image_size,
            pool_d=pool_d,
            pool_hw=pool_hw,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            device=dev,
        )

        # Per-pair clustering (single-cluster + MAD outlier trimming in z-scored space).
        keep = np.ones((len(df),), dtype=bool)
        report_rows: list[dict] = []
        summary_rows: list[dict] = []

        for pair, g in df.groupby("_pair_key", sort=True):
            idx = g.index.to_numpy(dtype=np.int64)
            n = int(idx.size)
            if n < int(min_group_size):
                summary_rows.append({"pair": pair, "n": n, "removed": 0, "note": "skip_small_group"})
                continue

            X = feats[idx]
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            Xz = (X - mu) / sd
            center = np.median(Xz, axis=0, keepdims=True)
            dist = np.linalg.norm(Xz - center, axis=1)

            mask, stats = _mad_mask(dist, mad_mult=float(mad_mult), max_remove_frac=float(max_remove_frac))
            removed = int(mask.sum())
            if removed <= 0:
                summary_rows.append({"pair": pair, "n": n, "removed": 0, "note": "no_outliers", **stats})
                continue

            # Respect split: keep val outliers by default.
            split_arr = g["_split"].to_numpy()
            if not bool(drop_val_outliers):
                mask = mask & (split_arr == "train")

            removed = int(mask.sum())
            if removed <= 0:
                summary_rows.append({"pair": pair, "n": n, "removed": 0, "note": "val_only_outliers", **stats})
                continue

            keep[idx[mask]] = False
            summary_rows.append({"pair": pair, "n": n, "removed": removed, "note": "removed", **stats})

            for j, d in zip(idx[mask].tolist(), dist[mask].tolist(), strict=False):
                report_rows.append(
                    {
                        "pct": int(pct),
                        "split": str(df.loc[j, "_split"]),
                        "exam_id": int(df.loc[j, "exam_id"]),
                        "left_code": int(df.loc[j, "_left_code_int"]),
                        "right_code": int(df.loc[j, "_right_code_int"]),
                        "pair": str(pair),
                        "dist": float(d),
                        "median": stats.get("median"),
                        "mad": stats.get("mad"),
                        "threshold": stats.get("threshold"),
                        "series_relpath": str(df.loc[j, "series_relpath"]),
                    }
                )

        out_dir = out_splits_root / f"{int(pct)}pct"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_df = df.loc[keep].drop(columns=["_left_code_int", "_right_code_int", "_pair_key"]).reset_index(drop=True)
        out_train = out_df[out_df["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
        out_val = out_df[out_df["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)

        out_train.to_csv(out_dir / "train.csv", index=False)
        out_val.to_csv(out_dir / "val.csv", index=False)

        report_path = out_dir / "outliers_removed.csv"
        summary_path = out_dir / "outliers_summary.csv"
        pd.DataFrame(report_rows).to_csv(report_path, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        typer.echo(f"wrote: {out_dir / 'train.csv'}  (n={len(out_train)})")
        typer.echo(f"wrote: {out_dir / 'val.csv'}    (n={len(out_val)})")
        typer.echo(f"wrote: {report_path}  (n={len(report_rows)})")
        typer.echo(f"wrote: {summary_path}")


if __name__ == "__main__":
    app()
