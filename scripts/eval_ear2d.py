from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.data.ear_dataset import EarCTHUEarDataset, EarPreprocessSpec
from medical_fenlei.metrics import binary_metrics, bootstrap_binary_metrics_by_exam
from medical_fenlei.models.ear2d_factory import make_ear2d_model_from_checkpoint
from medical_fenlei.paths import infer_dicom_root
from medical_fenlei.tasks import resolve_task

app = typer.Typer(add_completion=False)


def _default_out_dir(checkpoint: Path) -> Path:
    p = checkpoint.resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent
    return p.parent


@torch.no_grad()
def _predict(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    wl: float,
    ww: float,
    amp: bool,
    topk_slices: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    ys: list[int] = []
    ps: list[float] = []
    exam_ids: list[int] = []

    amp_enabled = bool(amp) and device.type == "cuda"
    autocast_ctx = (
        (torch.amp.autocast(device_type="cuda") if hasattr(torch, "amp") else torch.cuda.amp.autocast())
        if amp_enabled
        else None
    )

    for batch in loader:
        hu = batch["hu"].to(device, non_blocking=True)  # (B,K,H,W)
        y = batch["y"].view(-1).cpu().numpy().astype(int).tolist()
        meta = batch.get("meta") or {}

        x = hu.clamp(min=wl - ww / 2.0, max=wl + ww / 2.0)
        x = (x - (wl - ww / 2.0)) / (ww + 1e-6)
        x = x.unsqueeze(2)  # (B,K,1,H,W)

        if autocast_ctx is None:
            out = model(x, return_attention=True)
        else:
            with autocast_ctx:
                out = model(x, return_attention=True)

        logits = out["logits"].float().view(-1)
        prob = torch.sigmoid(logits).cpu().numpy().astype(float).tolist()

        attn = out.get("attention")
        if attn is None:
            attn_np = None
        else:
            attn_np = attn.detach().float().cpu().numpy()  # (B,K)

        exam = meta.get("exam_id")
        side = meta.get("side")
        series_relpath = meta.get("series_relpath")
        label_code = meta.get("label_code")
        slice_indices = meta.get("slice_indices")

        # Normalize meta containers
        if torch.is_tensor(exam):
            exam = exam.cpu().numpy().tolist()
        if torch.is_tensor(label_code):
            label_code = label_code.cpu().numpy().tolist()

        # side/series may come as list[str]
        exam_list = [int(x) for x in (exam or [])] if isinstance(exam, list) else [int(exam)] * len(y)
        side_list = list(side) if isinstance(side, list) else [str(side)] * len(y)
        rel_list = list(series_relpath) if isinstance(series_relpath, list) else [str(series_relpath)] * len(y)
        code_list = [int(x) if x is not None else None for x in (label_code or [])] if isinstance(label_code, list) else [label_code] * len(y)

        # slice_indices can be collated as:
        # - Tensor (B,K)
        # - list[K] of Tensor(B,)  (default_collate "transposes" lists)
        # - list[B] of list[K]
        if slice_indices is None:
            slice_indices_list = [None] * len(y)
        elif isinstance(slice_indices, list) and slice_indices and torch.is_tensor(slice_indices[0]):
            try:
                stacked = torch.stack(slice_indices, dim=1)  # (B,K)
                slice_indices_list = stacked.cpu().numpy().astype(int).tolist()
            except Exception:
                slice_indices_list = [None] * len(y)
        elif torch.is_tensor(slice_indices):
            slice_indices_list = slice_indices.cpu().numpy().astype(int).tolist()
        elif isinstance(slice_indices, list) and slice_indices and isinstance(slice_indices[0], (list, tuple)):
            slice_indices_list = slice_indices
        elif isinstance(slice_indices, list):
            slice_indices_list = [slice_indices for _ in range(len(y))]
        else:
            slice_indices_list = [None] * len(y)

        for i, (eid, sd, rel, lc, yt, pr) in enumerate(zip(exam_list, side_list, rel_list, code_list, y, prob)):
            pred = 1 if float(pr) >= 0.5 else 0
            row: dict[str, Any] = {
                "exam_id": int(eid),
                "side": str(sd),
                "series_relpath": str(rel),
                "label_code": int(lc) if lc is not None else None,
                "y_true": int(yt),
                "y_prob": float(pr),
                "y_pred": int(pred),
                "correct": bool(int(pred) == int(yt)),
            }

            if attn_np is not None:
                w = attn_np[i]
                k = min(int(topk_slices), int(w.shape[0]))
                top_idx = np.argsort(-w)[:k].astype(int).tolist()
                row["top_slices_k"] = k
                row["top_slices_idx"] = top_idx
                row["top_slices_w"] = [float(w[j]) for j in top_idx]

            si = slice_indices_list[i]
            if isinstance(si, (list, tuple)):
                row["slice_indices"] = [int(x) for x in si]

            rows.append(row)
            ys.append(int(yt))
            ps.append(float(pr))
            exam_ids.append(int(eid))

    df = pd.DataFrame(rows)
    metrics = binary_metrics(np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.float64))
    metrics["n_rows"] = int(len(df))
    metrics["n_unique_exams"] = int(df["exam_id"].nunique()) if not df.empty else 0
    return df, metrics | {"y_true": np.asarray(ys, dtype=np.int64), "y_prob": np.asarray(ps, dtype=np.float64), "exam_id": np.asarray(exam_ids, dtype=np.int64)}


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="outputs/.../checkpoints/best.pt 或 last.pt"),
    splits_root: Path = typer.Option(Path("artifacts/splits_dual")),
    pct: int = typer.Option(20),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), exists=True),
    dicom_base: Path = typer.Option(default_dicom_base()),
    cache_dir: Path = typer.Option(Path("cache/ears_hu")),
    amp: bool = typer.Option(True),
    num_workers: int = typer.Option(4),
    batch_size: int = typer.Option(16),
    topk_slices: int = typer.Option(3),
    n_boot: int = typer.Option(500, help="bootstrap 次数（按 exam_id 抽样）"),
    seed: int = typer.Option(42),
    out_dir: Path | None = typer.Option(None, help="默认保存到 run_dir（不入库）"),
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config") or {}

    task_name = str(ckpt.get("label_task") or (cfg.get("task") or {}).get("name") or "six_class")
    task = resolve_task(task_name)
    if task.kind != "binary":
        raise ValueError(f"eval_ear2d supports binary task only; got {task.kind}")

    spec_d = (cfg.get("data") or {}).get("spec") or {}
    spec = EarPreprocessSpec(
        num_slices=int(spec_d.get("num_slices", 32)),
        image_size=int(spec_d.get("image_size", 224)),
        crop_size=int(spec_d.get("crop_size", 192)),
        sampling=str(spec_d.get("sampling", "even")),
        block_len=int(spec_d.get("block_len", 64)),
        crop_mode=str(spec_d.get("crop_mode", "temporal_patch")),
        crop_lateral_band_frac=float(spec_d.get("crop_lateral_band_frac", 0.6) or 0.6),
        crop_lateral_bias=float(spec_d.get("crop_lateral_bias", 0.25) or 0.25),
        crop_min_area=int(spec_d.get("crop_min_area", 300) or 300),
        target_spacing=float(spec_d.get("target_spacing")) if spec_d.get("target_spacing") not in (None, "", 0, 0.0) else None,
        target_z_spacing=float(spec_d.get("target_z_spacing")) if spec_d.get("target_z_spacing") not in (None, "", 0, 0.0) else None,
        version=str(spec_d.get("version", "v1")),
    )

    model = make_ear2d_model_from_checkpoint(ckpt=ckpt, in_channels=1)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    base_wl = float((cfg.get("train") or {}).get("base_wl", 500.0) or 500.0)
    base_ww = float((cfg.get("train") or {}).get("base_ww", 3000.0) or 3000.0)
    # fallback: use CLI defaults
    wl, ww = base_wl, base_ww

    dicom_root = infer_dicom_root(dicom_base)
    tz_tag = f"_tz{float(spec.target_z_spacing):.6g}" if spec.target_z_spacing is not None and float(spec.target_z_spacing) > 0 else ""

    # Build val ear df from manifest + task.
    val_csv = splits_root / f"{pct}pct" / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    val_exam = set(pd.read_csv(val_csv)["exam_id"].astype(int).tolist())

    man = pd.read_csv(manifest_csv)
    df = man.loc[man["has_label"].fillna(False)].copy()
    df["label_code"] = pd.to_numeric(df["label_code"], errors="coerce")
    df = df[np.isfinite(df["label_code"])].copy()
    df["label_code"] = df["label_code"].astype(int)

    rel_codes = task.relevant_codes()
    pos_codes = set(task.pos_codes)
    df = df[df["exam_id"].astype(int).isin(val_exam)].copy()
    df = df[df["label_code"].isin(sorted(rel_codes))].copy()
    if df.empty:
        raise typer.Exit(code=2)
    df["y"] = df["label_code"].map(lambda c: 1.0 if int(c) in pos_codes else 0.0)
    df = df[["exam_id", "series_relpath", "side", "y", "label_code"]].reset_index(drop=True)

    ts_tag = f"_ts{float(spec.target_spacing):.6g}" if spec.target_spacing is not None and float(spec.target_spacing) > 0 else ""
    used_cache_dir = cache_dir / f"d{int(spec.num_slices)}_s{int(spec.image_size)}_c{int(spec.crop_size)}_{str(spec.sampling)}{ts_tag}{tz_tag}_crop{str(spec.crop_mode)}"
    ds = EarCTHUEarDataset(index_df=df, dicom_root=dicom_root, spec=spec, cache_dir=used_cache_dir, return_meta=True)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=True, persistent_workers=num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pred_df, m = _predict(model=model, loader=loader, device=device, wl=wl, ww=ww, amp=bool(amp), topk_slices=int(topk_slices))
    ci = bootstrap_binary_metrics_by_exam(m["y_true"], m["y_prob"], m["exam_id"], n_boot=int(n_boot), seed=int(seed))

    report = {
        "task": {"name": task.name, "pos_codes": list(task.pos_codes), "neg_codes": list(task.neg_codes)},
        "val": {k: v for k, v in m.items() if k not in ("y_true", "y_prob", "exam_id")},
        "bootstrap_ci_by_exam": ci,
        "checkpoint": str(checkpoint),
        "spec": spec.__dict__,
        "wl_ww": {"wl": float(wl), "ww": float(ww)},
    }

    out = out_dir or _default_out_dir(checkpoint)
    out = Path(out)
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "eval_binary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pred_df.to_csv(rep_dir / "predictions_val.csv", index=False)

    # Hard cases: high-confidence wrong + top loss.
    if not pred_df.empty:
        p = pred_df["y_prob"].astype(float)
        y = pred_df["y_true"].astype(int)
        loss = -(y * np.log(p.clip(1e-6, 1 - 1e-6)) + (1 - y) * np.log((1 - p).clip(1e-6, 1 - 1e-6)))
        pred_df["bce_loss"] = loss.astype(float)
        high_conf_wrong = pred_df[(pred_df["correct"] == False) & ((p >= 0.9) | (p <= 0.1))].sort_values("bce_loss", ascending=False).head(200)
        top_loss = pred_df.sort_values("bce_loss", ascending=False).head(200)
        high_conf_wrong.to_csv(rep_dir / "hard_cases_high_conf_wrong.csv", index=False)
        top_loss.to_csv(rep_dir / "hard_cases_top_loss.csv", index=False)

    typer.echo(f"saved: {rep_dir / 'eval_binary.json'}")
    typer.echo(f"saved: {rep_dir / 'predictions_val.csv'}")


if __name__ == "__main__":
    app()
