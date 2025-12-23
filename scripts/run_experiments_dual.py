from __future__ import annotations

import os
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
    epochs_1pct: int = typer.Option(200, help="1% 的 max epochs（配合 early-stop；默认偏多，因为每 epoch steps 很少）"),
    epochs_20pct: int = typer.Option(80, help="20% 的 max epochs（配合 early-stop）"),
    epochs_100pct: int = typer.Option(40, help="100% 的 max epochs（配合 early-stop；默认偏少，因为每 epoch steps 很多）"),
    early_stop_patience: int = typer.Option(20, help="早停 patience（0=关闭）"),
    early_stop_metric: str = typer.Option("macro_f1", help="val_loss | macro_f1 | macro_recall | macro_specificity | weighted_f1"),
    early_stop_min_delta: float = typer.Option(0.001, help="最小提升幅度"),
    weight_decay: float = typer.Option(0.05, help="AdamW weight decay（更强正则）"),
    label_smoothing: float = typer.Option(0.10, help="CrossEntropy label smoothing（更强正则）"),
    augment: bool = typer.Option(True, "--augment/--no-augment", help="训练时启用数据增强"),
    aug_flip_prob: float = typer.Option(0.5),
    aug_intensity_prob: float = typer.Option(0.7),
    aug_noise_prob: float = typer.Option(0.2),
    aug_gamma_prob: float = typer.Option(0.2),
    dry_run: bool = typer.Option(False, help="只打印命令不执行"),
) -> None:
    pct_list = _parse_int_list(pcts)
    model_list = _parse_str_list(models)

    if not pct_list:
        raise typer.Exit(code=2)
    if not model_list:
        raise typer.Exit(code=2)

    py = sys.executable

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
                "--early-stop-patience",
                str(int(early_stop_patience)),
                "--early-stop-metric",
                str(early_stop_metric),
                "--early-stop-min-delta",
                str(float(early_stop_min_delta)),
                "--weight-decay",
                str(float(weight_decay)),
                "--label-smoothing",
                str(float(label_smoothing)),
                "--aug-flip-prob",
                str(float(aug_flip_prob)),
                "--aug-intensity-prob",
                str(float(aug_intensity_prob)),
                "--aug-noise-prob",
                str(float(aug_noise_prob)),
                "--aug-gamma-prob",
                str(float(aug_gamma_prob)),
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
            if augment:
                cmd.append("--augment")
            else:
                cmd.append("--no-augment")

            print("\n$ " + " ".join(cmd), flush=True)
            if dry_run:
                continue
            subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    app()
