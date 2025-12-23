from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from medical_fenlei.data.ear_dataset import EarCTHUEarDataset, EarPreprocessSpec
from medical_fenlei.models.slice_attention_resnet import SliceAttentionResNet
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _load_exam_ids(path: Path) -> set[int]:
    df = pd.read_csv(path)
    if "exam_id" not in df.columns:
        raise ValueError(f"missing exam_id in {path}")
    return set(df["exam_id"].astype(int).tolist())


@torch.no_grad()
def _infer_embeddings(
    *,
    model: SliceAttentionResNet,
    loader: DataLoader,
    device: torch.device,
    wl: float,
    ww: float,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    embs: list[np.ndarray] = []
    exam_ids: list[int] = []
    sides: list[str] = []
    label_codes: list[int] = []

    amp_enabled = bool(amp) and device.type == "cuda"
    autocast_ctx = (
        (torch.amp.autocast(device_type="cuda") if hasattr(torch, "amp") else torch.cuda.amp.autocast())
        if amp_enabled
        else None
    )

    for batch in loader:
        hu = batch["hu"].to(device, non_blocking=True)  # (B,K,H,W)
        meta = batch.get("meta") or {}
        x = hu.clamp(min=float(wl) - float(ww) / 2.0, max=float(wl) + float(ww) / 2.0)
        x = (x - (float(wl) - float(ww) / 2.0)) / (float(ww) + 1e-6)
        x = x.unsqueeze(2)  # (B,K,1,H,W)

        if autocast_ctx is None:
            out = model(x, return_embedding=True)
        else:
            with autocast_ctx:
                out = model(x, return_embedding=True)

        emb = out["embedding"].detach().float().cpu().numpy()  # (B,D)
        embs.append(emb)

        eid = meta.get("exam_id")
        sd = meta.get("side")
        lc = meta.get("label_code")

        if torch.is_tensor(eid):
            eid = eid.cpu().numpy().tolist()
        if torch.is_tensor(lc):
            lc = lc.cpu().numpy().tolist()

        if isinstance(eid, list):
            exam_ids.extend([int(x) for x in eid])
        else:
            exam_ids.extend([int(eid)] * emb.shape[0])

        if isinstance(sd, list):
            sides.extend([str(x) for x in sd])
        else:
            sides.extend([str(sd)] * emb.shape[0])

        if isinstance(lc, list):
            label_codes.extend([int(x) for x in lc])
        else:
            label_codes.extend([int(lc)] * emb.shape[0])

    arr = np.concatenate(embs, axis=0) if embs else np.zeros((0, int(model.embed_dim)), dtype=np.float32)
    return {"embedding": arr.astype(np.float32), "exam_id": np.asarray(exam_ids, dtype=np.int64), "side": np.asarray(sides, dtype=object), "label_code": np.asarray(label_codes, dtype=np.int64)}


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="outputs/.../checkpoints/best.pt 或 last.pt"),
    split: str = typer.Option("train", help="train | val"),
    splits_root: Path = typer.Option(Path("artifacts/splits_dual")),
    pct: int = typer.Option(100),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), exists=True),
    dicom_base: Path = typer.Option(Path("data/medical_data_2")),
    cache_dir: Path = typer.Option(Path("cache/ears_hu")),
    amp: bool = typer.Option(True),
    batch_size: int = typer.Option(32),
    num_workers: int = typer.Option(8),
    limit: int | None = typer.Option(None, help="仅处理前 N 个耳朵样本（用于 smoke）"),
    out_npz: Path = typer.Option(Path("artifacts/embeddings_ear2d.npz"), help="输出 embeddings（不入库）"),
) -> None:
    split = str(split).strip().lower()
    if split not in ("train", "val"):
        raise typer.Exit(code=2)

    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config") or {}

    spec_d = (cfg.get("data") or {}).get("spec") or {}
    spec = EarPreprocessSpec(
        num_slices=int(spec_d.get("num_slices", 32)),
        image_size=int(spec_d.get("image_size", 224)),
        crop_size=int(spec_d.get("crop_size", 192)),
        sampling=str(spec_d.get("sampling", "even")),
        block_len=int(spec_d.get("block_len", 64)),
        version=str(spec_d.get("version", "v1")),
    )

    model_spec = ckpt.get("model_spec") or (cfg.get("model") or {}).get("spec") or {}
    model = SliceAttentionResNet(
        backbone=str(model_spec.get("backbone", "resnet18")),
        in_channels=1,
        attn_hidden=int(model_spec.get("attn_hidden", 128)),
        dropout=float(model_spec.get("dropout", 0.2)),
        out_dim=int(model_spec.get("out_dim", 1)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)

    train_csv = splits_root / f"{pct}pct" / "train.csv"
    val_csv = splits_root / f"{pct}pct" / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise typer.Exit(code=2)
    exam_ids = _load_exam_ids(train_csv if split == "train" else val_csv)

    man = pd.read_csv(manifest_csv)
    df = man.loc[man["has_label"].fillna(False)].copy()
    df["label_code"] = pd.to_numeric(df["label_code"], errors="coerce")
    df = df[np.isfinite(df["label_code"])].copy()
    df["label_code"] = df["label_code"].astype(int)
    df = df[df["exam_id"].astype(int).isin(exam_ids)].copy()
    if df.empty:
        raise typer.Exit(code=2)

    df = df[["exam_id", "series_relpath", "side", "label_code"]].reset_index(drop=True)
    df["y"] = 0.0  # unused
    if limit is not None and int(limit) > 0 and len(df) > int(limit):
        df = df.sample(n=int(limit), random_state=42).reset_index(drop=True)

    dicom_root = infer_dicom_root(dicom_base)
    used_cache_dir = cache_dir / f"d{int(spec.num_slices)}_s{int(spec.image_size)}_c{int(spec.crop_size)}_{str(spec.sampling)}"
    ds = EarCTHUEarDataset(index_df=df, dicom_root=dicom_root, spec=spec, cache_dir=used_cache_dir, return_meta=True)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=True, persistent_workers=num_workers > 0)

    base_wl = float((cfg.get("train") or {}).get("base_wl", 500.0) or 500.0)
    base_ww = float((cfg.get("train") or {}).get("base_ww", 3000.0) or 3000.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    out = _infer_embeddings(model=model, loader=loader, device=device, wl=base_wl, ww=base_ww, amp=bool(amp))

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out)
    meta = {
        "checkpoint": str(checkpoint),
        "split": split,
        "pct": int(pct),
        "splits_root": str(splits_root),
        "manifest_csv": str(manifest_csv),
        "dicom_root": str(dicom_root),
        "cache_dir": str(used_cache_dir),
        "spec": spec.__dict__,
        "rows": int(out["embedding"].shape[0]),
        "dim": int(out["embedding"].shape[1]) if out["embedding"].ndim == 2 else None,
    }
    out_npz.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"saved: {out_npz}")
    typer.echo(f"saved: {out_npz.with_suffix('.meta.json')}")


if __name__ == "__main__":
    app()
