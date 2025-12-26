from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.data.ear_dataset import EarCTHUEarDataset, EarPreprocessSpec
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _expand_labeled_ears(exam_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for r in exam_df.itertuples(index=False):
        for side in ("left", "right"):
            code = getattr(r, f"{side}_code", None)
            has_label = not (code is None or (isinstance(code, float) and np.isnan(code)))
            if not has_label:
                continue
            rows.append(
                {
                    "exam_id": int(r.exam_id),
                    "series_relpath": str(r.series_relpath),
                    "side": side,
                    "y": 0.0,  # unused for caching
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["exam_id", "side"]).reset_index(drop=True)


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="由 scripts/make_splits_dual.py 生成"),
    pct: int = typer.Option(100, help="使用哪个 split 来构建缓存（100% 推荐，能覆盖 1%/20%）"),
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录"),
    cache_dir: Path = typer.Option(Path("cache/ears_hu"), help="缓存目录（不入库）"),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    crop_size: int = typer.Option(192),
    crop_mode: str = typer.Option("temporal_patch", help="crop：bbox_bias | temporal_patch（颞骨 patch）"),
    crop_lateral_band_frac: float = typer.Option(0.6, help="temporal_patch：外侧区域 band 宽度（0~1，越小越靠边）"),
    crop_lateral_bias: float = typer.Option(0.25, help="x-center bias（越小越靠外侧）"),
    crop_min_area: int = typer.Option(300, help="temporal_patch：连通域最小面积阈值（像素）"),
    sampling: str = typer.Option("even", help="even | air_block"),
    block_len: int = typer.Option(64, help="sampling=air_block 时的连续块长度"),
    target_spacing: float = typer.Option(0.0, help=">0 时启用：in-plane 重采样到统一 spacing（mm/px），会写入新的 cache key"),
    target_z_spacing: float = typer.Option(0.0, help=">0 时启用：z 方向重采样到统一 spacing（mm），输出固定 num_slices 的物理窗口"),
    num_workers: int = typer.Option(16),
    prefetch_factor: int = typer.Option(2),
) -> None:
    crop_mode = str(crop_mode).strip()
    if crop_mode not in ("bbox_bias", "temporal_patch"):
        raise ValueError("crop_mode must be one of: bbox_bias, temporal_patch")

    split_dir = splits_root / f"{pct}pct"
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

    df = pd.concat([train_df, val_df], ignore_index=True)
    if "exam_id" in df.columns:
        df = df.drop_duplicates(subset=["exam_id"]).reset_index(drop=True)

    ear_df = _expand_labeled_ears(df)
    if ear_df.empty:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)

    spec = EarPreprocessSpec(
        num_slices=int(num_slices),
        image_size=int(image_size),
        crop_size=int(crop_size),
        crop_mode=str(crop_mode),
        crop_lateral_band_frac=float(crop_lateral_band_frac),
        crop_lateral_bias=float(crop_lateral_bias),
        crop_min_area=int(crop_min_area),
        sampling=str(sampling),
        block_len=int(block_len),
        target_spacing=float(target_spacing) if float(target_spacing) > 0 else None,
        target_z_spacing=float(target_z_spacing) if float(target_z_spacing) > 0 else None,
        version="v3",
    )
    ts_tag = f"_ts{float(target_spacing):.6g}" if float(target_spacing) > 0 else ""
    tz_tag = f"_tz{float(target_z_spacing):.6g}" if float(target_z_spacing) > 0 else ""
    used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}_c{int(crop_size)}_{str(sampling)}{ts_tag}{tz_tag}_crop{crop_mode}"

    ds = EarCTHUEarDataset(
        index_df=ear_df,
        dicom_root=dicom_root,
        spec=spec,
        cache_dir=used_cache_dir,
        return_meta=False,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"cache_dir: {used_cache_dir}")
    typer.echo(f"ears: {len(ds)}  workers: {num_workers}  sampling={sampling}")

    for _ in tqdm(loader, total=len(ds)):
        pass


if __name__ == "__main__":
    app()
