from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _epochs_for_pct(pct: int, *, e1: int, e20: int, e100: int) -> int:
    if pct == 1:
        return int(e1)
    if pct == 20:
        return int(e20)
    if pct == 100:
        return int(e100)
    return int(e100)


@app.command()
def main(
    pcts: str = typer.Option("1,20,100", help="按数据量顺序运行，例如 1,20,100"),
    models: str = typer.Option(
        "dual_resnet10_3d,dual_resnet18_3d,dual_resnet34_3d,dual_resnet50_3d,dual_resnet101_3d,dual_resnet152_3d,dual_resnet200_3d,dual_unet_3d,dual_vit_3d",
        help="按模型顺序运行（逗号分隔）",
    ),
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="split 根目录"),
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录"),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    num_workers: int = typer.Option(16),
    amp: bool = typer.Option(True),
    auto_batch: bool = typer.Option(True),
    max_batch_size: int = typer.Option(32),
    compile: bool = typer.Option(False),
    cache: bool = typer.Option(True),
    cache_dir: Path = typer.Option(Path("cache/dual_volumes")),
    cache_dtype: str = typer.Option("float16"),
    epochs_1pct: int = typer.Option(80, help="1% 的 epochs（默认偏多，因为每 epoch steps 很少）"),
    epochs_20pct: int = typer.Option(30, help="20% 的 epochs"),
    epochs_100pct: int = typer.Option(12, help="100% 的 epochs（默认偏少，因为每 epoch steps 很多）"),
    dry_run: bool = typer.Option(False, help="只打印命令不执行"),
) -> None:
    pct_list = _parse_int_list(pcts)
    model_list = _parse_str_list(models)

    if not pct_list:
        raise typer.Exit(code=2)
    if not model_list:
        raise typer.Exit(code=2)

    py = sys.executable

    for pct in pct_list:
        epochs = _epochs_for_pct(pct, e1=epochs_1pct, e20=epochs_20pct, e100=epochs_100pct)
        for model in model_list:
            cmd = [
                py,
                "scripts/train_dual.py",
                "--splits-root",
                str(splits_root),
                "--pct",
                str(int(pct)),
                "--dicom-base",
                str(dicom_base),
                "--model",
                model,
                "--epochs",
                str(int(epochs)),
                "--num-workers",
                str(int(num_workers)),
                "--num-slices",
                str(int(num_slices)),
                "--image-size",
                str(int(image_size)),
                "--max-batch-size",
                str(int(max_batch_size)),
                "--cache-dir",
                str(cache_dir),
                "--cache-dtype",
                str(cache_dtype),
            ]

            if amp:
                cmd.append("--amp")
            else:
                cmd.append("--no-amp")
            if auto_batch:
                cmd.append("--auto-batch")
            if compile:
                cmd.append("--compile")
            if cache:
                cmd.append("--cache")
            else:
                cmd.append("--no-cache")

            print("\n$ " + " ".join(cmd), flush=True)
            if dry_run:
                continue
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()

