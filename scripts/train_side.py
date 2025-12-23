from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.side_dataset import EarCTSideDataset
from medical_fenlei.models.resnet3d import ResNet10_3D
from medical_fenlei.models.slice_resnet import SliceMeanResNet
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def _make_model(name: str, *, num_classes: int, pretrained: bool) -> torch.nn.Module:
    if name == "slice_mean_resnet18":
        return SliceMeanResNet(num_classes=num_classes, in_channels=1, pretrained=pretrained)
    if name == "resnet10_3d":
        return ResNet10_3D(num_classes=num_classes, in_channels=1)
    raise ValueError(f"unknown model: {name}")


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        amp_enabled = amp and device.type == "cuda"
        if amp_enabled:
            if hasattr(torch, "amp"):
                autocast_ctx = torch.amp.autocast(device_type="cuda")
            else:
                autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            logits = model(x)
            loss = loss_fn(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_acc += _accuracy(logits.detach(), y.detach())
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "acc": total_acc / max(n_batches, 1)}


def _resolve_split_paths(splits_root: Path, pct: int) -> tuple[Path, Path]:
    split_dir = splits_root / f"{pct}pct"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    return train_csv, val_csv


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits"), help="由 scripts/make_splits.py 生成"),
    pct: int = typer.Option(100, help="训练数据比例：1 / 20 / 100"),
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录"),
    output_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/<timestamp>）"),
    model: str = typer.Option("slice_mean_resnet18", help="slice_mean_resnet18 | resnet10_3d"),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(2),
    num_workers: int = typer.Option(8),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    lr: float = typer.Option(1e-4),
    seed: int = typer.Option(42),
    amp: bool = typer.Option(True),
    pretrained: bool = typer.Option(False, help="仅对 slice_mean_resnet18 生效；会触发下载"),
) -> None:
    train_csv, val_csv = _resolve_split_paths(splits_root, pct)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    if train_df.empty or val_df.empty:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)
    num_classes = len(CLASS_ID_TO_NAME)

    _seed_everything(seed)

    train_ds = EarCTSideDataset(
        index_df=train_df,
        dicom_root=dicom_root,
        num_slices=num_slices,
        image_size=image_size,
        flip_right=True,
    )
    val_ds = EarCTSideDataset(
        index_df=val_df,
        dicom_root=dicom_root,
        num_slices=num_slices,
        image_size=image_size,
        flip_right=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _make_model(model, num_classes=num_classes, pretrained=pretrained).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    amp_enabled = amp and device.type == "cuda"
    if not amp_enabled:
        scaler = None
    elif hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = torch.cuda.amp.GradScaler()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = output_dir or Path("outputs") / f"{model}_{pct}pct_{ts}"
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "metrics.jsonl"

    typer.echo(f"task: 单耳(左/右) 6 分类  classes={num_classes}")
    typer.echo(f"model: {model}  pct={pct}%")
    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"train: {len(train_ds)}  val: {len(val_ds)}")
    typer.echo(f"output: {out}")

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(
            model=net,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
        )
        val_m = _run_epoch(
            model=net,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
            scaler=None,
            amp=amp,
        )

        rec = {"epoch": epoch, "train": train_m, "val": val_m}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        typer.echo(f"epoch {epoch}: train_loss={train_m['loss']:.4f} acc={train_m['acc']:.3f}  val_loss={val_m['loss']:.4f} acc={val_m['acc']:.3f}")

        ckpt = {
            "epoch": epoch,
            "model_name": model,
            "num_classes": num_classes,
            "state_dict": net.state_dict(),
            "val_loss": val_m["loss"],
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(ckpt, ckpt_dir / "best.pt")


if __name__ == "__main__":
    app()

