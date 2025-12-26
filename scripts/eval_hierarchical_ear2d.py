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
from medical_fenlei.metrics import binary_metrics, classification_report_from_confusion
from medical_fenlei.models.slice_attention_resnet import SliceAttentionResNet
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _default_out_dir(checkpoint: Path) -> Path:
    p = checkpoint.resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent
    return p.parent


def _load_spec(cfg: dict[str, Any]) -> EarPreprocessSpec:
    spec_d = (cfg.get("data") or {}).get("spec") or {}
    return EarPreprocessSpec(
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


@torch.no_grad()
def _predict(
    *,
    model_stage1: SliceAttentionResNet,
    model_stage2: SliceAttentionResNet,
    loader: DataLoader,
    device: torch.device,
    wl: float,
    ww: float,
    amp: bool,
    stage1_threshold: float,
    class_id_to_code_stage2: dict[int, int],
    merge_code4_into_other: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_stage1.eval()
    model_stage2.eval()

    rows: list[dict[str, Any]] = []
    y_true_codes: list[int] = []
    y_pred_codes: list[int] = []
    exam_ids: list[int] = []

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
            logits1 = model_stage1(x).float().view(-1)  # (B,)
            logits2 = model_stage2(x).float()  # (B,C)
        else:
            with autocast_ctx:
                logits1 = model_stage1(x).float().view(-1)
                logits2 = model_stage2(x).float()

        p_abn = torch.sigmoid(logits1).detach().cpu().numpy().astype(float).tolist()
        pred2 = logits2.detach().cpu().argmax(dim=1).numpy().astype(int).tolist()

        exam = meta.get("exam_id")
        side = meta.get("side")
        series_relpath = meta.get("series_relpath")
        label_code = meta.get("label_code")

        if torch.is_tensor(exam):
            exam = exam.cpu().numpy().tolist()
        if torch.is_tensor(label_code):
            label_code = label_code.cpu().numpy().tolist()

        exam_list = [int(x) for x in (exam or [])] if isinstance(exam, list) else [int(exam)] * len(pred2)
        side_list = list(side) if isinstance(side, list) else [str(side)] * len(pred2)
        rel_list = list(series_relpath) if isinstance(series_relpath, list) else [str(series_relpath)] * len(pred2)
        code_list = [int(x) if x is not None else None for x in (label_code or [])] if isinstance(label_code, list) else [label_code] * len(pred2)

        for i, (eid, sd, rel, lc, pabn, p2) in enumerate(zip(exam_list, side_list, rel_list, code_list, p_abn, pred2)):
            true_code = int(lc) if lc is not None else -1
            true_code_m = 6 if (merge_code4_into_other and int(true_code) == 4) else int(true_code)

            is_abn = float(pabn) >= float(stage1_threshold)
            pred_code2 = int(class_id_to_code_stage2.get(int(p2), 6))
            final_pred = int(pred_code2 if is_abn else 5)

            rows.append(
                {
                    "exam_id": int(eid),
                    "side": str(sd),
                    "series_relpath": str(rel),
                    "label_code": int(true_code),
                    "label_code_mapped": int(true_code_m),
                    "p_abnormal": float(pabn),
                    "pred_stage2_code": int(pred_code2),
                    "pred_final_code": int(final_pred),
                }
            )

            if true_code_m in (1, 2, 3, 5, 6):
                y_true_codes.append(int(true_code_m))
                y_pred_codes.append(int(final_pred))
                exam_ids.append(int(eid))

    df = pd.DataFrame(rows)

    # 5-way report: {1,2,3,5,6} with code4 merged into 6 (if enabled).
    codes = [1, 2, 3, 5, 6]
    code_to_idx = {int(c): int(i) for i, c in enumerate(codes)}
    cm = np.zeros((len(codes), len(codes)), dtype=np.int64)
    for yt, yp in zip(y_true_codes, y_pred_codes):
        if int(yt) in code_to_idx and int(yp) in code_to_idx:
            cm[code_to_idx[int(yt)], code_to_idx[int(yp)]] += 1

    class_id_to_name = {
        0: "慢性化脓性中耳炎",
        1: "中耳胆脂瘤",
        2: "分泌性中耳炎",
        3: "正常",
        4: "其他(含code4)",
    }
    rep = classification_report_from_confusion(cm, class_id_to_name=class_id_to_name)

    # Stage1 binary metrics (normal vs abnormal) on the same set.
    y_abn = np.asarray([0 if int(c) == 5 else 1 for c in y_true_codes], dtype=np.int64)
    p_abn_all = np.asarray([float(x) for x in df.loc[df["label_code_mapped"].isin(codes), "p_abnormal"].tolist()], dtype=np.float64)
    m1 = binary_metrics(y_abn, p_abn_all, threshold=float(stage1_threshold), specificity_target=0.95)

    out = {
        "hierarchical": {"codes": codes, "report": rep},
        "stage1_binary": m1,
        "n_rows": int(df.shape[0]),
        "n_eval": int(len(y_true_codes)),
        "merge_code4_into_other": bool(merge_code4_into_other),
        "stage1_threshold": float(stage1_threshold),
    }
    return df, out


@app.command()
def main(
    stage1_checkpoint: Path = typer.Option(..., exists=True, help="Stage1：normal vs abnormal 的 best.pt/last.pt"),
    stage2_checkpoint: Path = typer.Option(..., exists=True, help="Stage2：abnormal subtype 的 best.pt/last.pt"),
    splits_root: Path = typer.Option(Path("artifacts/splits_dual")),
    pct: int = typer.Option(20),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), exists=True),
    dicom_base: Path = typer.Option(default_dicom_base()),
    cache_dir: Path = typer.Option(Path("cache/ears_hu")),
    stage1_threshold: float = typer.Option(0.5, help="Stage1 abnormal 概率阈值（固定阈值，避免用 val 调参）"),
    amp: bool = typer.Option(True),
    batch_size: int = typer.Option(16),
    num_workers: int = typer.Option(8),
    out_dir: Path | None = typer.Option(None, help="默认保存到 stage1 run_dir（不入库）"),
) -> None:
    ckpt1 = torch.load(stage1_checkpoint, map_location="cpu")
    cfg1 = ckpt1.get("config") or {}
    ckpt2 = torch.load(stage2_checkpoint, map_location="cpu")
    cfg2 = ckpt2.get("config") or {}

    spec1 = _load_spec(cfg1)
    spec2 = _load_spec(cfg2)
    if (spec1.num_slices, spec1.image_size, spec1.crop_size) != (spec2.num_slices, spec2.image_size, spec2.crop_size):
        raise ValueError(f"stage1/spec != stage2/spec: {spec1} vs {spec2}")

    ms1 = ckpt1.get("model_spec") or (cfg1.get("model") or {}).get("spec") or {}
    ms2 = ckpt2.get("model_spec") or (cfg2.get("model") or {}).get("spec") or {}
    model1 = SliceAttentionResNet.from_spec(ms1, in_channels=1)
    model2 = SliceAttentionResNet.from_spec(ms2, in_channels=1)
    model1.load_state_dict(ckpt1["state_dict"], strict=True)
    model2.load_state_dict(ckpt2["state_dict"], strict=True)

    base_wl = float((cfg1.get("train") or {}).get("base_wl", 500.0) or 500.0)
    base_ww = float((cfg1.get("train") or {}).get("base_ww", 3000.0) or 3000.0)

    merge_code4_into_other = bool(ckpt2.get("merge_code4_into_other") or cfg2.get("task", {}).get("merge_code4_into_other") or True)
    code_to_class_id = ckpt2.get("code_to_class_id") or cfg2.get("task", {}).get("code_to_class_id") or {}
    code_to_class_id = {int(k): int(v) for k, v in dict(code_to_class_id).items()}
    class_id_to_code = {int(v): int(k) for k, v in code_to_class_id.items()}

    # Build val ear df (all labeled ears, code4 optionally merged).
    val_csv = splits_root / f"{pct}pct" / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    val_exam = set(pd.read_csv(val_csv)["exam_id"].astype(int).tolist())

    man = pd.read_csv(manifest_csv)
    df = man.loc[man["has_label"].fillna(False)].copy()
    df["label_code"] = pd.to_numeric(df["label_code"], errors="coerce")
    df = df[np.isfinite(df["label_code"])].copy()
    df["label_code"] = df["label_code"].astype(int)
    df = df[df["exam_id"].astype(int).isin(val_exam)].copy()
    if df.empty:
        raise typer.Exit(code=2)

    df = df[["exam_id", "series_relpath", "side", "label_code"]].reset_index(drop=True)
    df["y"] = 0.0  # unused

    dicom_root = infer_dicom_root(dicom_base)
    ts_tag = f"_ts{float(spec1.target_spacing):.6g}" if spec1.target_spacing is not None and float(spec1.target_spacing) > 0 else ""
    tz_tag = f"_tz{float(spec1.target_z_spacing):.6g}" if spec1.target_z_spacing is not None and float(spec1.target_z_spacing) > 0 else ""
    used_cache_dir = cache_dir / f"d{int(spec1.num_slices)}_s{int(spec1.image_size)}_c{int(spec1.crop_size)}_{str(spec1.sampling)}{ts_tag}{tz_tag}_crop{str(spec1.crop_mode)}"
    ds = EarCTHUEarDataset(index_df=df, dicom_root=dicom_root, spec=spec1, cache_dir=used_cache_dir, return_meta=True)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=True, persistent_workers=num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    model2 = model2.to(device)

    pred_df, rep = _predict(
        model_stage1=model1,
        model_stage2=model2,
        loader=loader,
        device=device,
        wl=base_wl,
        ww=base_ww,
        amp=bool(amp),
        stage1_threshold=float(stage1_threshold),
        class_id_to_code_stage2=class_id_to_code,
        merge_code4_into_other=bool(merge_code4_into_other),
    )

    report = {
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "spec": spec1.__dict__,
        "wl_ww": {"wl": float(base_wl), "ww": float(base_ww)},
        "hierarchical_report": rep,
    }

    out = out_dir or _default_out_dir(stage1_checkpoint)
    out = Path(out)
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "eval_hierarchical.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pred_df.to_csv(rep_dir / "predictions_hierarchical_val.csv", index=False)

    typer.echo(f"saved: {rep_dir / 'eval_hierarchical.json'}")
    typer.echo(f"saved: {rep_dir / 'predictions_hierarchical_val.csv'}")


if __name__ == "__main__":
    app()
