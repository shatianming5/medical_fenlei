from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.data.dual_dataset import DualPreprocessSpec, EarCTDualDataset
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="由 scripts/make_splits_dual.py 生成"),
    pct: int = typer.Option(100, help="使用哪个 split 来构建缓存（100% 推荐，能覆盖 1%/20%）"),
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录"),
    cache_dir: Path = typer.Option(Path("cache/dual_volumes"), help="缓存目录（不入库）"),
    cache_dtype: str = typer.Option("float16", help="缓存数据类型：float16 | float32"),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    crop_size: int = typer.Option(192, help="每侧颞骨 ROI patch 大小（像素；会再 resize 到 image_size）"),
    window_wl: float = typer.Option(700.0, help="CT 窗位（HU）"),
    window_ww: float = typer.Option(4000.0, help="CT 窗宽（HU）"),
    window2_wl: float = typer.Option(0.0, help="第二个窗位（HU；window2_ww<=0 关闭）"),
    window2_ww: float = typer.Option(0.0, help="第二个窗宽（HU；<=0 关闭）"),
    pair_features: str = typer.Option("none", help="双侧对比特征：none | self_other_diff"),
    sampling: str = typer.Option("air_block", help="z 采样：even | air_block"),
    block_len: int = typer.Option(64),
    target_spacing: float = typer.Option(0.7, help="统一 in-plane spacing（mm；<=0 关闭）"),
    target_z_spacing: float = typer.Option(0.8, help="统一 z spacing（mm；<=0 关闭）"),
    num_workers: int = typer.Option(16),
    prefetch_factor: int = typer.Option(2),
) -> None:
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

    # Cache union set, unique by exam_id.
    df = pd.concat([train_df, val_df], ignore_index=True)
    if "exam_id" in df.columns:
        df = df.drop_duplicates(subset=["exam_id"]).reset_index(drop=True)

    dicom_root = infer_dicom_root(dicom_base)
    used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}"

    if str(sampling) not in {"even", "air_block"}:
        raise ValueError(f"unknown sampling: {sampling!r} (expected even|air_block)")
    pair_features_s = str(pair_features).strip().lower() if pair_features is not None else "none"
    if pair_features_s not in {"none", "self_other_diff"}:
        raise ValueError(f"unknown pair_features: {pair_features_s!r} (expected none|self_other_diff)")
    w2_ww = float(window2_ww)
    w2_wl = float(window2_wl)
    window2_wl_v = w2_wl if w2_ww > 0 else None
    window2_ww_v = w2_ww if w2_ww > 0 else None
    preprocess_spec = DualPreprocessSpec(
        num_slices=int(num_slices),
        image_size=int(image_size),
        crop_size=int(crop_size),
        window_wl=float(window_wl),
        window_ww=float(window_ww),
        window2_wl=window2_wl_v,
        window2_ww=window2_ww_v,
        pair_features=pair_features_s,
        sampling=str(sampling),
        block_len=int(block_len),
        flip_right=True,
        target_spacing=float(target_spacing) if float(target_spacing) > 0 else None,
        target_z_spacing=float(target_z_spacing) if float(target_z_spacing) > 0 else None,
    )

    ds = EarCTDualDataset(
        index_df=df,
        dicom_root=dicom_root,
        spec=preprocess_spec,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
        return_image=False,
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
    typer.echo(f"cache_dir: {used_cache_dir} ({cache_dtype})")
    typer.echo(
        f"window: wl={float(preprocess_spec.window_wl):g} ww={float(preprocess_spec.window_ww):g} "
        f"wl2={float(preprocess_spec.window2_wl or 0.0):g} ww2={float(preprocess_spec.window2_ww or 0.0):g} "
        f"pair={str(preprocess_spec.pair_features)}"
    )
    typer.echo(f"items: {len(ds)}  workers: {num_workers}")

    for _ in tqdm(loader, total=len(ds)):
        pass


if __name__ == "__main__":
    app()
