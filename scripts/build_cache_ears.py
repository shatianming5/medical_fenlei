from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录"),
    cache_dir: Path = typer.Option(Path("cache/ears_hu"), help="缓存目录（不入库）"),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    crop_size: int = typer.Option(192),
    sampling: str = typer.Option("even", help="even | air_block"),
    block_len: int = typer.Option(64, help="sampling=air_block 时的连续块长度"),
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
        sampling=str(sampling),
        block_len=int(block_len),
        version="v1",
    )
    used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}_c{int(crop_size)}_{str(sampling)}"

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

