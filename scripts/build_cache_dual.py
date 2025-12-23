from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from medical_fenlei.data.dual_dataset import EarCTDualDataset
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="由 scripts/make_splits_dual.py 生成"),
    pct: int = typer.Option(100, help="使用哪个 split 来构建缓存（100% 推荐，能覆盖 1%/20%）"),
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录"),
    cache_dir: Path = typer.Option(Path("cache/dual_volumes"), help="缓存目录（不入库）"),
    cache_dtype: str = typer.Option("float16", help="缓存数据类型：float16 | float32"),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
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

    ds = EarCTDualDataset(
        index_df=df,
        dicom_root=dicom_root,
        num_slices=num_slices,
        image_size=image_size,
        flip_right=True,
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
    typer.echo(f"items: {len(ds)}  workers: {num_workers}")

    for _ in tqdm(loader, total=len(ds)):
        pass


if __name__ == "__main__":
    app()

