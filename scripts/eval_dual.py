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
from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.dual_dataset import DualPreprocessSpec, EarCTDualDataset
from medical_fenlei.metrics import binary_average_precision, binary_auroc, binary_metrics, bootstrap_binary_metrics_by_exam, classification_report_from_confusion
from medical_fenlei.models.dual_factory import make_dual_model
from medical_fenlei.paths import infer_dicom_root
from medical_fenlei.tasks import resolve_task

app = typer.Typer(add_completion=False)


def _default_out_dir(checkpoint: Path) -> Path:
    p = checkpoint.resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent
    return p.parent


def _strip_compile_prefix(state_dict: dict) -> dict:
    # torch.compile state_dict keys often start with "_orig_mod."
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if any(k.startswith("_orig_mod.") for k in keys):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _apply_binary_task(
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_label_ids: tuple[int, ...],
    neg_label_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_m = torch.zeros_like(mask, dtype=torch.bool)
    for v in pos_label_ids:
        pos_m |= labels == int(v)
    neg_m = torch.zeros_like(mask, dtype=torch.bool)
    for v in neg_label_ids:
        neg_m |= labels == int(v)
    keep = mask.bool() & (pos_m | neg_m)
    out = torch.zeros_like(labels)
    out[pos_m] = 1
    return out, keep


def _ovr_auc_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, class_id_to_name: dict[int, str]) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float64)
    if y_prob.ndim != 2:
        raise ValueError(f"expected y_prob (N,C), got {y_prob.shape}")
    n, c = y_prob.shape
    if int(y_true.size) != int(n):
        raise ValueError("y_true and y_prob must have same length")

    per_class: list[dict[str, Any]] = []
    aucs: list[float] = []
    aps: list[float] = []
    for k in range(int(c)):
        yk = (y_true == int(k)).astype(int)
        score = y_prob[:, int(k)]
        auroc = binary_auroc(yk, score)
        auprc = binary_average_precision(yk, score)
        if auroc is not None:
            aucs.append(float(auroc))
        if auprc is not None:
            aps.append(float(auprc))
        per_class.append(
            {
                "id": int(k),
                "name": class_id_to_name.get(int(k), str(k)),
                "auroc_ovr": float(auroc) if auroc is not None else None,
                "auprc_ovr": float(auprc) if auprc is not None else None,
                "support": int((yk == 1).sum()),
            }
        )

    return {
        "macro_auroc_ovr": float(np.mean(np.asarray(aucs, dtype=np.float64))) if aucs else None,
        "macro_auprc_ovr": float(np.mean(np.asarray(aps, dtype=np.float64))) if aps else None,
        "per_class": per_class,
    }


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="outputs/.../checkpoints/best.pt 或 last.pt"),
    index_csv: Path = typer.Option(..., exists=True, help="artifacts/splits_dual/*pct/{val,test}.csv 或自定义索引"),
    dicom_base: Path = typer.Option(default_dicom_base()),
    out_dir: Path | None = typer.Option(None, help="默认保存到 run_dir/reports/（不入库）"),
    split_name: str = typer.Option("val", help="用于报告标识：val | test | custom"),
    label_task: str | None = typer.Option(None, help="默认从 checkpoint 读取（没有则按 six_class）"),
    batch_size: int = typer.Option(1),
    num_workers: int = typer.Option(4),
    amp: bool = typer.Option(True),
    num_slices: int | None = typer.Option(None, help="默认从 checkpoint 读取"),
    image_size: int | None = typer.Option(None, help="默认从 checkpoint 读取"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="使用 cache/ 缓存体数据，提高吞吐"),
    cache_dir: Path = typer.Option(Path("cache/dual_volumes"), help="缓存目录（不入库）"),
    cache_dtype: str = typer.Option("float16", help="缓存数据类型：float16 | float32"),
    tail_class_ids: str = typer.Option("", help="多分类: 逗号分隔 class id（0-based）用于 tail metrics（如设置B的 Label2 -> class_id=1）"),
    threshold: float = typer.Option(0.5, help="二分类: 评估 threshold（binary_metrics）"),
    spec_target: float = typer.Option(0.95, help="二分类: Sens@Spec 的 spec target"),
    n_boot: int = typer.Option(1000, help="二分类: exam-level bootstrap 次数（0=不做）"),
    seed: int = typer.Option(42),
) -> None:
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)

    ckpt = torch.load(checkpoint, map_location="cpu")
    model_name = str(ckpt.get("model_name", "dual_resnet10_3d"))
    model_kwargs = dict(ckpt.get("model_kwargs", {}) or {})
    num_classes = int(ckpt.get("num_classes", len(CLASS_ID_TO_NAME)))
    if num_slices is None:
        num_slices = int(ckpt.get("num_slices", 32))
    if image_size is None:
        image_size = int(ckpt.get("image_size", 224))

    inferred_task = label_task or ckpt.get("label_task")
    if inferred_task is None:
        inferred_task = "six_class"
    task_spec = resolve_task(str(inferred_task))
    if int(task_spec.num_classes) != int(num_classes):
        typer.echo(f"warning: task={task_spec.name} expects num_classes={task_spec.num_classes} but ckpt has {num_classes}")

    raw_names = ckpt.get("class_id_to_name")
    if isinstance(raw_names, dict) and raw_names:
        class_id_to_name = {int(k): str(v) for k, v in raw_names.items()}
    else:
        class_id_to_name = dict(task_spec.class_id_to_name)

    used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}"
    if not cache:
        used_cache_dir = None

    pre = ckpt.get("preprocess")
    if isinstance(pre, dict):
        crop_size = int(pre.get("crop_size", 192) or 192)
        sampling = str(pre.get("sampling", "air_block") or "air_block")
        block_len = int(pre.get("block_len", 64) or 64)
        target_spacing = float(pre.get("target_spacing", 0.0) or 0.0)
        target_z_spacing = float(pre.get("target_z_spacing", 0.0) or 0.0)
        window_wl = float(pre.get("window_wl", 700.0) or 700.0)
        window_ww = float(pre.get("window_ww", 4000.0) or 4000.0)
        window2_wl = float(pre.get("window2_wl", 0.0) or 0.0)
        window2_ww = float(pre.get("window2_ww", 0.0) or 0.0)
        pair_features = str(pre.get("pair_features", "none") or "none").strip().lower()
    else:
        crop_size = 192
        sampling = "air_block"
        block_len = 64
        target_spacing = 0.7
        target_z_spacing = 0.8
        window_wl = 700.0
        window_ww = 4000.0
        window2_wl = 0.0
        window2_ww = 0.0
        pair_features = "none"

    w2_ww = float(window2_ww)
    w2_wl = float(window2_wl)
    window2_wl_v = w2_wl if w2_ww > 0 else None
    window2_ww_v = w2_ww if w2_ww > 0 else None
    base_channels = 2 if window2_ww_v is not None else 1
    pair_factor = 3 if str(pair_features) == "self_other_diff" else 1
    in_channels = int(base_channels) * int(pair_factor)
    if "in_channels" in ckpt:
        try:
            in_channels = int(ckpt.get("in_channels") or in_channels)
        except Exception:
            pass

    preprocess_spec = DualPreprocessSpec(
        num_slices=int(num_slices),
        image_size=int(image_size),
        crop_size=int(crop_size),
        window_wl=float(window_wl),
        window_ww=float(window_ww),
        window2_wl=window2_wl_v,
        window2_ww=window2_ww_v,
        pair_features=str(pair_features),
        sampling=str(sampling),
        block_len=int(block_len),
        flip_right=True,
        target_spacing=float(target_spacing) if float(target_spacing) > 0 else None,
        target_z_spacing=float(target_z_spacing) if float(target_z_spacing) > 0 else None,
    )

    ds = EarCTDualDataset(
        index_df=df,
        dicom_root=dicom_root,
        spec=preprocess_spec,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (int(num_slices), int(image_size), int(image_size))

    model, _ = make_dual_model(
        model_name,
        num_classes=num_classes,
        in_channels=int(in_channels),
        img_size=img_size,
        vit_patch_size=tuple(model_kwargs.get("vit_patch_size", (4, 16, 16))),
        vit_pool=str(model_kwargs.get("vit_pool", "cls")),
        vit_hidden_size=int(model_kwargs.get("vit_hidden_size", 768)),
        vit_mlp_dim=int(model_kwargs.get("vit_mlp_dim", 3072)),
        vit_num_layers=int(model_kwargs.get("vit_num_layers", 12)),
        vit_num_heads=int(model_kwargs.get("vit_num_heads", 12)),
        unet_channels=tuple(model_kwargs.get("unet_channels", (16, 32, 64, 128, 256))),
        unet_strides=tuple(model_kwargs.get("unet_strides", (2, 2, 2, 2))),
        unet_num_res_units=int(model_kwargs.get("unet_num_res_units", 2)),
    )
    model = model.to(device)
    state_dict = _strip_compile_prefix(dict(ckpt.get("state_dict", {}) or {}))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    amp_enabled = bool(amp) and device.type == "cuda"
    autocast_ctx = (
        (torch.amp.autocast(device_type="cuda") if hasattr(torch, "amp") else torch.cuda.amp.autocast())
        if amp_enabled
        else None
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_rows: list[np.ndarray] = []
    exam_ids: list[int] = []
    sides: list[str] = []
    dates: list[str] = []
    series_relpaths: list[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"]  # (B,2) on CPU
            m = batch["label_mask"]

            if task_spec.kind == "binary":
                y, m = _apply_binary_task(
                    y,
                    m,
                    pos_label_ids=task_spec.pos_label_ids(),
                    neg_label_ids=task_spec.neg_label_ids(),
                )

            if autocast_ctx is None:
                logits = model(x)
            else:
                with autocast_ctx:
                    logits = model(x)

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # (B,2,C)
            pred = probs.argmax(axis=-1)  # (B,2)

            meta = batch["meta"]
            b_exam_ids = meta["exam_id"].cpu().numpy().tolist() if torch.is_tensor(meta["exam_id"]) else meta["exam_id"]
            b_dates = meta["date"]
            b_series = meta["series_relpath"]

            y_np = y.cpu().numpy().astype(int)
            m_np = m.cpu().numpy().astype(bool)

            for bi, (eid, dt, rel) in enumerate(zip(b_exam_ids, b_dates, b_series)):
                for si, side_name in enumerate(("left", "right")):
                    if not bool(m_np[bi, si]):
                        continue
                    yt = int(y_np[bi, si])
                    pr = int(pred[bi, si])
                    pb = probs[bi, si].astype(np.float64)
                    y_true.append(yt)
                    y_pred.append(pr)
                    y_prob_rows.append(pb)
                    exam_ids.append(int(eid))
                    sides.append(str(side_name))
                    dates.append(str(dt))
                    series_relpaths.append(str(rel))

    if not y_true:
        typer.echo("no labeled ears found after filtering/masking")
        raise typer.Exit(code=2)

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.stack(y_prob_rows, axis=0).astype(np.float64)
    exam_id_np = np.asarray(exam_ids, dtype=np.int64)

    pred_df = pd.DataFrame(
        {
            "exam_id": exam_id_np,
            "date": dates,
            "series_relpath": series_relpaths,
            "side": sides,
            "y_true": y_true_np,
            "y_true_name": [class_id_to_name.get(int(v)) for v in y_true_np.tolist()],
            "y_pred": y_pred_np,
            "y_pred_name": [class_id_to_name.get(int(v)) for v in y_pred_np.tolist()],
            "correct": (y_true_np == y_pred_np),
        }
    )

    # Attach probabilities (kept small for 6-class / binary).
    if int(y_prob_np.shape[1]) <= 12:
        for k in range(int(y_prob_np.shape[1])):
            pred_df[f"p_{k}"] = y_prob_np[:, k].astype(np.float32)

    out_base = Path(out_dir) if out_dir is not None else (_default_out_dir(checkpoint) / "reports")
    out_base.mkdir(parents=True, exist_ok=True)
    pred_path = out_base / f"predictions_{str(split_name)}.csv"
    pred_df.to_csv(pred_path, index=False)

    report: dict[str, Any] = {
        "task": {"name": str(task_spec.name), "kind": str(task_spec.kind), "num_classes": int(num_classes)},
        "split": str(split_name),
        "checkpoint": str(checkpoint),
        "index_csv": str(index_csv),
        "n_ears": int(y_true_np.size),
        "n_exams": int(np.unique(exam_id_np).size),
    }

    if task_spec.kind == "binary":
        y_pos = y_prob_np[:, 1]
        metrics = binary_metrics(y_true_np, y_pos, threshold=float(threshold), specificity_target=float(spec_target))
        report["binary"] = {k: v for k, v in metrics.items() if k not in ("y_true", "y_prob")}
        if int(n_boot) > 0:
            report["bootstrap_ci_by_exam"] = bootstrap_binary_metrics_by_exam(
                y_true_np,
                y_pos,
                exam_id_np,
                n_boot=int(n_boot),
                seed=int(seed),
                threshold=float(threshold),
                specificity_target=float(spec_target),
            )
    else:
        k = int(num_classes)
        cm = torch.zeros((k, k), dtype=torch.int64)
        idx = y_true_np * k + y_pred_np
        binc = np.bincount(idx.astype(np.int64), minlength=k * k).reshape(k, k)
        cm += torch.from_numpy(binc.astype(np.int64))

        cls_report = classification_report_from_confusion(cm, class_id_to_name=class_id_to_name)
        ovr = _ovr_auc_metrics(y_true_np, y_prob_np, class_id_to_name=class_id_to_name)

        # Merge AUROC/AUPRC into per_class entries.
        per_auc = {int(d["id"]): d for d in ovr["per_class"]}
        per_class = []
        for d in cls_report.get("per_class") or []:
            dd = dict(d)
            extra = per_auc.get(int(dd["id"]))
            if extra:
                dd["auroc_ovr"] = extra.get("auroc_ovr")
                dd["auprc_ovr"] = extra.get("auprc_ovr")
            per_class.append(dd)

        report["multiclass"] = {
            "accuracy": float(cls_report["accuracy"]),
            "macro_f1": float(cls_report["macro_f1"]),
            "macro_recall": float(cls_report["macro_recall"]),
            "macro_specificity": float(cls_report["macro_specificity"]),
            "weighted_f1": float(cls_report["weighted_f1"]),
            "macro_auroc_ovr": ovr["macro_auroc_ovr"],
            "macro_auprc_ovr": ovr["macro_auprc_ovr"],
            "per_class": per_class,
            "confusion_matrix": cls_report["confusion_matrix"],
        }

        tail_ids = _parse_int_list(tail_class_ids)
        if tail_ids:
            tail: list[dict[str, Any]] = []
            per_map = {int(d["id"]): d for d in per_class}
            for cid in tail_ids:
                if int(cid) not in per_map:
                    continue
                tail.append(per_map[int(cid)])
            report["tail_metrics"] = tail

    rep_path = out_base / f"eval_dual_{str(split_name)}.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote: {rep_path}")
    typer.echo(f"wrote: {pred_path}")


if __name__ == "__main__":
    app()

