from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from PIL import Image
from torch.utils.data import DataLoader

from medical_fenlei.data.ear_dataset import EarCTHUEarDataset, EarPreprocessSpec
from medical_fenlei.models.slice_attention_resnet import SliceAttentionResNet
from medical_fenlei.paths import infer_dicom_root
from medical_fenlei.tasks import resolve_task

app = typer.Typer(add_completion=False)


def _default_out_dir(checkpoint: Path) -> Path:
    p = checkpoint.resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent / "reports" / "gradcam"
    return p.parent / "gradcam"


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    x = np.clip(img01, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="outputs/.../checkpoints/best.pt 或 last.pt"),
    exam_id: int = typer.Option(..., help="要解释的检查号"),
    side: str = typer.Option(..., help="left | right"),
    splits_root: Path = typer.Option(Path("artifacts/splits_dual")),
    pct: int = typer.Option(20),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), exists=True),
    dicom_base: Path = typer.Option(Path("data/medical_data_2")),
    cache_dir: Path = typer.Option(Path("cache/ears_hu")),
    wl: float = typer.Option(500.0),
    ww: float = typer.Option(3000.0),
    topk_slices: int = typer.Option(3, help="按 attention 权重取 top-k slice 做 Grad-CAM"),
    out_dir: Path | None = typer.Option(None, help="输出目录（默认 outputs/<run>/reports/gradcam）"),
) -> None:
    side = str(side).strip().lower()
    if side not in ("left", "right"):
        raise typer.Exit(code=2)

    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config") or {}
    task_name = str(ckpt.get("label_task") or (cfg.get("task") or {}).get("name") or "six_class")
    task = resolve_task(task_name)

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

    dicom_root = infer_dicom_root(dicom_base)
    used_cache_dir = cache_dir / f"d{int(spec.num_slices)}_s{int(spec.image_size)}_c{int(spec.crop_size)}_{str(spec.sampling)}"

    # Locate the ear row in manifest.
    man = pd.read_csv(manifest_csv)
    m = man["has_label"].fillna(False) & (man["exam_id"].astype(int) == int(exam_id)) & (man["side"].astype(str) == side)
    row_df = man[m].copy()
    if row_df.empty:
        typer.echo("no such labeled ear in manifest")
        raise typer.Exit(code=2)

    # For binary tasks, keep only relevant codes.
    if task.kind == "binary":
        code = int(pd.to_numeric(row_df.iloc[0]["label_code"], errors="coerce"))
        if code not in task.relevant_codes():
            typer.echo(f"ear label_code={code} is not in task relevant codes={sorted(task.relevant_codes())}")
            raise typer.Exit(code=2)
        y = 1.0 if code in set(task.pos_codes) else 0.0
    else:
        y = float("nan")

    df = pd.DataFrame(
        [
            {
                "exam_id": int(exam_id),
                "series_relpath": str(row_df.iloc[0]["series_relpath"]),
                "side": side,
                "y": float(y) if y == y else 0.0,
                "label_code": int(pd.to_numeric(row_df.iloc[0]["label_code"], errors="coerce")),
            }
        ]
    )

    ds = EarCTHUEarDataset(index_df=df, dicom_root=dicom_root, spec=spec, cache_dir=used_cache_dir, return_meta=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []

    def fwd_hook(_m, _inp, outp):
        acts.append(outp)

    def bwd_hook(_m, grad_in, grad_out):
        grads.append(grad_out[0])

    # last conv block for ResNet
    h1 = model.backbone.layer4.register_forward_hook(fwd_hook)
    h2 = model.backbone.layer4.register_full_backward_hook(bwd_hook)

    try:
        batch = next(iter(loader))
        hu = batch["hu"].to(device)  # (1,K,H,W)
        meta = batch["meta"]

        x = hu.clamp(min=float(wl) - float(ww) / 2.0, max=float(wl) + float(ww) / 2.0)
        x = (x - (float(wl) - float(ww) / 2.0)) / (float(ww) + 1e-6)
        x = x.unsqueeze(2)  # (1,K,1,H,W)

        out = model(x, return_attention=True)
        logits = out["logits"].view(-1)[0]
        prob = torch.sigmoid(logits).item()

        model.zero_grad(set_to_none=True)
        logits.backward()

        if not acts or not grads:
            raise RuntimeError("failed to capture activations/gradients")

        A = acts[-1].detach()  # (K,C,h,w) because B=1 and we flattened slices
        G = grads[-1].detach()

        # attention weights
        attn = out["attention"].detach().cpu().numpy()[0]  # (K,)
        topk = int(min(int(topk_slices), attn.shape[0]))
        top_idx = np.argsort(-attn)[:topk].astype(int).tolist()

        out_root = Path(out_dir) if out_dir is not None else _default_out_dir(checkpoint)
        out_root.mkdir(parents=True, exist_ok=True)

        report: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "task": str(task.name),
            "exam_id": int(exam_id),
            "side": side,
            "label_code": int(meta["label_code"][0]) if torch.is_tensor(meta["label_code"]) else int(meta.get("label_code", -1)),
            "prob": float(prob),
            "top_slices_idx": top_idx,
            "top_slices_attn": [float(attn[i]) for i in top_idx],
        }

        # Original grayscale slices for visualization.
        x01 = x.detach().cpu().numpy()[0, :, 0]  # (K,H,W)

        for i in top_idx:
            a = A[i]  # (C,h,w)
            g = G[i]
            w = g.mean(dim=(1, 2))  # (C,)
            cam = torch.relu((w[:, None, None] * a).sum(dim=0))  # (h,w)
            cam = cam / (cam.max() + 1e-6)
            cam_np = cam.detach().cpu().numpy()
            cam_up = np.array(Image.fromarray(_to_uint8(cam_np)).resize((x01.shape[-1], x01.shape[-2]), resample=Image.BILINEAR)).astype(np.float32) / 255.0

            base = x01[i]
            overlay = np.stack([base, base, base], axis=-1)
            heat = np.stack([cam_up, np.zeros_like(cam_up), 1.0 - cam_up], axis=-1)
            mix = (0.55 * overlay + 0.45 * heat).clip(0.0, 1.0)

            Image.fromarray(_to_uint8(overlay)).save(out_root / f"slice_{i:02d}_gray.png")
            Image.fromarray(_to_uint8(cam_up)).save(out_root / f"slice_{i:02d}_cam.png")
            Image.fromarray(_to_uint8(mix)).save(out_root / f"slice_{i:02d}_overlay.png")

        (out_root / "gradcam_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"saved: {out_root}")
    finally:
        h1.remove()
        h2.remove()


if __name__ == "__main__":
    app()

