from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
import typer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from contextlib import nullcontext

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.dataset import EarCTDataset
from medical_fenlei.models.slice_resnet import SliceMeanResNet
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    mask = labels != -1
    if mask.sum().item() == 0:
        return 0, 0
    pred = logits.argmax(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


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
    n_batches = 0
    l_correct = l_total = 0
    r_correct = r_total = 0

    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        left_label = batch["left_label"].to(device, non_blocking=True)
        right_label = batch["right_label"].to(device, non_blocking=True)

        amp_enabled = amp and device.type == "cuda"
        if amp_enabled:
            if hasattr(torch, "amp"):
                autocast_ctx = torch.amp.autocast(device_type="cuda")
            else:
                autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            left_logits = model(left)
            right_logits = model(right)
            loss = loss_fn(left_logits, left_label) + loss_fn(right_logits, right_label)

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
        n_batches += 1

        c, t = _accuracy(left_logits.detach(), left_label.detach())
        l_correct += c
        l_total += t
        c, t = _accuracy(right_logits.detach(), right_label.detach())
        r_correct += c
        r_total += t

    avg_loss = total_loss / max(n_batches, 1)
    return {
        "loss": avg_loss,
        "left_acc": (l_correct / l_total) if l_total else None,
        "right_acc": (r_correct / r_total) if r_total else None,
    }


@app.command()
def main(
    index_csv: Path = typer.Option(Path("artifacts/dataset_index.csv"), exists=True, help="由 scripts/build_index.py 生成"),
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录"),
    output_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/<timestamp>）"),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(2),
    num_workers: int = typer.Option(8),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    lr: float = typer.Option(1e-4),
    val_ratio: float = typer.Option(0.2),
    seed: int = typer.Option(42),
    amp: bool = typer.Option(True),
    pretrained: bool = typer.Option(False, help="是否使用 ImageNet 预训练权重（会触发下载）"),
) -> None:
    dicom_root = infer_dicom_root(dicom_base)
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    num_classes = len(CLASS_ID_TO_NAME)
    _seed_everything(seed)

    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, shuffle=True)

    train_ds = EarCTDataset(index_df=train_df, dicom_root=dicom_root, num_slices=num_slices, image_size=image_size)
    val_ds = EarCTDataset(index_df=val_df, dicom_root=dicom_root, num_slices=num_slices, image_size=image_size)

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
    model = SliceMeanResNet(num_classes=num_classes, in_channels=1, pretrained=pretrained).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    amp_enabled = amp and device.type == "cuda"
    if not amp_enabled:
        scaler = None
    elif hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = torch.cuda.amp.GradScaler()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = output_dir or Path("outputs") / ts
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    log_path = out / "metrics.jsonl"

    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"train: {len(train_ds)}  val: {len(val_ds)}  classes: {num_classes}")
    typer.echo(f"output: {out}")

    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
        )
        val_m = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
            scaler=None,
            amp=amp,
        )

        rec = {"epoch": epoch, "train": train_m, "val": val_m}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        typer.echo(f"epoch {epoch}: train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f}")

        ckpt = {
            "epoch": epoch,
            "num_classes": num_classes,
            "model": model.state_dict(),
            "val_loss": val_m["loss"],
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(ckpt, ckpt_dir / "best.pt")


if __name__ == "__main__":
    app()
