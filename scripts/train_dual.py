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
from medical_fenlei.data.dual_dataset import EarCTDualDataset
from medical_fenlei.models.dual_resnet3d import DualResNet10_3D
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    *,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    # logits: (B,2,C)  labels: (B,2)  mask: (B,2)
    if logits.ndim != 3:
        raise ValueError(f"expected logits (B,2,C), got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels shape {tuple(labels.shape)} != {tuple(logits.shape[:2])}")
    if label_mask.shape != labels.shape:
        raise ValueError(f"mask shape {tuple(label_mask.shape)} != {tuple(labels.shape)}")

    total = torch.tensor(0.0, device=logits.device)
    n = 0
    for side in (0, 1):
        m = label_mask[:, side].bool()
        if m.any():
            total = total + loss_fn(logits[m, side], labels[m, side])
            n += 1
    if n <= 0:
        return total
    return total / float(n)


@torch.no_grad()
def _masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1)  # (B,2)
    mask = label_mask.bool()
    total_n = int(mask.sum().item())
    if total_n <= 0:
        return {"acc": 0.0, "left_acc": 0.0, "right_acc": 0.0, "n": 0, "left_n": 0, "right_n": 0}

    total_correct = int(((pred == labels) & mask).sum().item())

    left_mask = mask[:, 0]
    right_mask = mask[:, 1]
    left_n = int(left_mask.sum().item())
    right_n = int(right_mask.sum().item())
    left_correct = int(((pred[:, 0] == labels[:, 0]) & left_mask).sum().item())
    right_correct = int(((pred[:, 1] == labels[:, 1]) & right_mask).sum().item())

    left_acc = float(left_correct / left_n) if left_n > 0 else 0.0
    right_acc = float(right_correct / right_n) if right_n > 0 else 0.0

    return {
        "acc": float(total_correct / total_n),
        "left_acc": left_acc,
        "right_acc": right_acc,
        "n": total_n,
        "left_n": left_n,
        "right_n": right_n,
    }


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
    total_left = 0.0
    total_right = 0.0
    total_n = 0
    total_left_n = 0
    total_right_n = 0
    n_batches = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        m = batch["label_mask"].to(device, non_blocking=True)

        amp_enabled = amp and device.type == "cuda"
        if amp_enabled:
            if hasattr(torch, "amp"):
                autocast_ctx = torch.amp.autocast(device_type="cuda")
            else:
                autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            logits = model(x)  # (B,2,C)
            loss = _masked_ce_loss(logits, y, m, loss_fn=loss_fn)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        metrics = _masked_accuracy(logits.detach(), y.detach(), m.detach())
        left_n = int(m[:, 0].sum().item())
        right_n = int(m[:, 1].sum().item())

        total_loss += float(loss.detach().cpu().item())
        total_acc += float(metrics["acc"]) * int(metrics["n"])
        total_left += float(metrics["left_acc"]) * left_n
        total_right += float(metrics["right_acc"]) * right_n
        total_n += int(metrics["n"])
        total_left_n += left_n
        total_right_n += right_n
        n_batches += 1

    if total_n <= 0:
        return {"loss": total_loss / max(n_batches, 1), "acc": 0.0, "left_acc": 0.0, "right_acc": 0.0, "n": 0}

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_acc / total_n,
        "left_acc": (total_left / max(total_left_n, 1)) if total_left_n > 0 else 0.0,
        "right_acc": (total_right / max(total_right_n, 1)) if total_right_n > 0 else 0.0,
        "n": total_n,
    }


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
    splits_root: Path = typer.Option(Path("artifacts/splits_dual"), help="由 scripts/make_splits_dual.py 生成"),
    pct: int = typer.Option(100, help="训练数据比例：1 / 20 / 100"),
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录"),
    output_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/<timestamp>）"),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(1),
    num_workers: int = typer.Option(8),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
    lr: float = typer.Option(1e-4),
    seed: int = typer.Option(42),
    amp: bool = typer.Option(True),
) -> None:
    train_csv, val_csv = _resolve_split_paths(splits_root, pct)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    if train_df.empty or val_df.empty:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)
    num_classes = len(CLASS_ID_TO_NAME)

    _seed_everything(seed)

    train_ds = EarCTDualDataset(
        index_df=train_df,
        dicom_root=dicom_root,
        num_slices=num_slices,
        image_size=image_size,
        flip_right=True,
    )
    val_ds = EarCTDualDataset(
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
    net = DualResNet10_3D(num_classes=num_classes, in_channels=1).to(device)

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
    out = output_dir or Path("outputs") / f"dual_resnet10_3d_{pct}pct_{ts}"
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "metrics.jsonl"

    typer.echo(f"task: 一次检查 -> 左/右双输出 6 分类  classes={num_classes}")
    typer.echo(f"model: dual_resnet10_3d  pct={pct}%")
    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"train_exams: {len(train_ds)}  val_exams: {len(val_ds)}")
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

        typer.echo(
            "epoch "
            f"{epoch}: train_loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} "
            f"(L={train_m['left_acc']:.3f} R={train_m['right_acc']:.3f})  "
            f"val_loss={val_m['loss']:.4f} acc={val_m['acc']:.3f} "
            f"(L={val_m['left_acc']:.3f} R={val_m['right_acc']:.3f})"
        )

        ckpt = {
            "epoch": epoch,
            "task": "dual",
            "model_name": "dual_resnet10_3d",
            "num_classes": num_classes,
            "num_slices": num_slices,
            "image_size": image_size,
            "state_dict": net.state_dict(),
            "val_loss": val_m["loss"],
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(ckpt, ckpt_dir / "best.pt")


if __name__ == "__main__":
    app()
