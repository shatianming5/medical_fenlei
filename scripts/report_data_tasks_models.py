from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import typer

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.tasks import TASKS, resolve_task

app = typer.Typer(add_completion=False)


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{100.0 * float(part) / float(total):.2f}%"


def _vc_int(series: pd.Series) -> dict[int, int]:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    return {int(k): int(v) for k, v in s.value_counts().sort_index().items()}


def _describe_numeric(series: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).to_dict()
    return {
        "n": int(s.size),
        "min": float(q[0.0]),
        "p10": float(q[0.1]),
        "p25": float(q[0.25]),
        "p50": float(q[0.5]),
        "p75": float(q[0.75]),
        "p90": float(q[0.9]),
        "p95": float(q[0.95]),
        "p99": float(q[0.99]),
        "max": float(q[1.0]),
    }


def _fmt_desc(d: dict[str, float | int]) -> str:
    n = int(d.get("n", 0) or 0)
    if n <= 0:
        return "n=0"
    return (
        f"n={n} min={d['min']:.4g} p10={d['p10']:.4g} p25={d['p25']:.4g} p50={d['p50']:.4g} "
        f"p75={d['p75']:.4g} p90={d['p90']:.4g} p95={d['p95']:.4g} p99={d['p99']:.4g} max={d['max']:.4g}"
    )


def _filter_df_for_codes(df: pd.DataFrame, *, codes: set[int]) -> pd.DataFrame:
    if not codes:
        return df
    if "left_code" not in df.columns or "right_code" not in df.columns:
        return df
    mask = df["left_code"].isin(codes) | df["right_code"].isin(codes)
    return df.loc[mask].reset_index(drop=True)


def _count_codes(df: pd.DataFrame, *, codes: set[int]) -> int:
    if not codes:
        return 0
    if "left_code" not in df.columns or "right_code" not in df.columns:
        return 0
    return int(df["left_code"].isin(codes).sum() + df["right_code"].isin(codes).sum())


def _top_pair_counts(df: pd.DataFrame, *, topk: int = 15) -> dict[str, int]:
    if "left_code" not in df.columns or "right_code" not in df.columns:
        return {}
    left = pd.to_numeric(df["left_code"], errors="coerce").fillna(-1).astype(int)
    right = pd.to_numeric(df["right_code"], errors="coerce").fillna(-1).astype(int)
    key = left.astype(str) + "," + right.astype(str)
    return {str(k): int(v) for k, v in key.value_counts().head(int(topk)).items()}


@app.command()
def main(
    out_md: Path = typer.Option(Path("docs/CURRENT_DATA_TASK_MODEL_REPORT.md")),
    out_json: Path = typer.Option(Path("docs/CURRENT_DATA_TASK_MODEL_REPORT.json")),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv")),
    splits_root: Path = typer.Option(Path("artifacts/splits_dual_patient_clustered_v1")),
    pct: int = typer.Option(100),
    dual_cache_dir: Path = typer.Option(Path("cache/dual_volumes/d32_s224"), help="dual cache 目录（统计用）"),
    ear_cache_dir: Path = typer.Option(Path("cache/ears_hu/d32_s224_c192_even"), help="ear2d cache 目录（统计用）"),
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- Manifest (ear-level)
    m = pd.read_csv(manifest_csv)
    labeled = m[m.get("has_label", True)].copy()

    manifest_summary: dict = {
        "rows": int(len(m)),
        "unique_exams": int(m["exam_id"].nunique()) if "exam_id" in m.columns else None,
        "unique_series_uid": int(m["series_uid"].nunique()) if "series_uid" in m.columns else None,
        "unique_patient_key_hash": int(m["patient_key_hash"].nunique()) if "patient_key_hash" in m.columns else None,
        "has_label_true": int(m["has_label"].sum()) if "has_label" in m.columns else None,
        "has_label_false": int((~m["has_label"]).sum()) if "has_label" in m.columns else None,
        "side_counts": m["side"].value_counts().to_dict() if "side" in m.columns else None,
    }

    if not labeled.empty and "label_code" in labeled.columns:
        manifest_summary["label_code_counts_labeled_ears"] = _vc_int(labeled["label_code"])

    if "date_match" in labeled.columns:
        manifest_summary["date_match_counts_labeled_ears"] = labeled["date_match"].value_counts().to_dict()

    for col in ["manufacturer", "manufacturer_model", "convolution_kernel"]:
        if col in labeled.columns:
            manifest_summary[f"{col}_top15_labeled_ears"] = labeled[col].fillna("NA").value_counts().head(15).to_dict()

    for col in ["spacing_z", "slice_thickness", "spacing_between_slices", "n_instances"]:
        if col in labeled.columns:
            manifest_summary[f"{col}_desc"] = _describe_numeric(labeled[col])
            manifest_summary[f"abs_{col}_desc"] = _describe_numeric(pd.to_numeric(labeled[col], errors="coerce").abs())

    for col in ["date", "label_date", "folder_date"]:
        if col in labeled.columns:
            s = pd.to_datetime(labeled[col], errors="coerce").dropna()
            if not s.empty:
                manifest_summary[f"{col}_range"] = {"min": str(s.min().date()), "max": str(s.max().date()), "n": int(s.size)}

    # ---- Splits (exam-level dual)
    split_dir = splits_root / f"{int(pct)}pct"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"missing split files under: {split_dir}")

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)

    splits_summary: dict = {
        "splits_root": str(splits_root),
        "pct": int(pct),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "train_exams": int(len(train)),
        "val_exams": int(len(val)),
        "train_unique_patients": int(train["patient_key"].nunique()) if "patient_key" in train.columns else None,
        "val_unique_patients": int(val["patient_key"].nunique()) if "patient_key" in val.columns else None,
        "train_missing_left_code": int(train["left_code"].isna().sum()) if "left_code" in train.columns else None,
        "train_missing_right_code": int(train["right_code"].isna().sum()) if "right_code" in train.columns else None,
        "val_missing_left_code": int(val["left_code"].isna().sum()) if "left_code" in val.columns else None,
        "val_missing_right_code": int(val["right_code"].isna().sum()) if "right_code" in val.columns else None,
        "train_left_code_counts": _vc_int(train["left_code"]) if "left_code" in train.columns else None,
        "train_right_code_counts": _vc_int(train["right_code"]) if "right_code" in train.columns else None,
        "val_left_code_counts": _vc_int(val["left_code"]) if "left_code" in val.columns else None,
        "val_right_code_counts": _vc_int(val["right_code"]) if "right_code" in val.columns else None,
        "train_both_sides_code_counts": _vc_int(pd.concat([train["left_code"], train["right_code"]], ignore_index=True))
        if "left_code" in train.columns
        else None,
        "val_both_sides_code_counts": _vc_int(pd.concat([val["left_code"], val["right_code"]], ignore_index=True))
        if "left_code" in val.columns
        else None,
        "train_top_pairs_left_right_top15": _top_pair_counts(train, topk=15),
        "val_top_pairs_left_right_top15": _top_pair_counts(val, topk=15),
    }
    if "date_match" in train.columns:
        splits_summary["train_date_match_counts"] = train["date_match"].value_counts().to_dict()
    if "date_match" in val.columns:
        splits_summary["val_date_match_counts"] = val["date_match"].value_counts().to_dict()

    # ---- Outlier cleaning (if present)
    outlier_summary: dict | None = None
    outliers_removed = split_dir / "outliers_removed.csv"
    outliers_summary = split_dir / "outliers_summary.csv"
    if outliers_removed.exists() and outliers_summary.exists():
        r = pd.read_csv(outliers_removed)
        s = pd.read_csv(outliers_summary)
        outlier_summary = {
            "outliers_removed_csv": str(outliers_removed),
            "outliers_summary_csv": str(outliers_summary),
            "removed_rows": int(len(r)),
            "removed_pair_counts_top15": r["pair"].value_counts().head(15).to_dict() if "pair" in r.columns else None,
            "pairs_with_removed": int((s.get("removed", 0) > 0).sum()) if "removed" in s.columns else None,
        }

    # ---- Task summary on this split
    tasks_summary: list[dict] = []
    for name in sorted(TASKS.keys()):
        spec = resolve_task(name)
        row: dict = {
            "name": spec.name,
            "kind": spec.kind,
            "num_classes": int(spec.num_classes),
            "pos_codes": list(spec.pos_codes),
            "neg_codes": list(spec.neg_codes),
        }
        if spec.kind == "binary":
            rel = spec.relevant_codes()
            tr = _filter_df_for_codes(train, codes=rel)
            va = _filter_df_for_codes(val, codes=rel)
            row.update(
                {
                    "relevant_codes": sorted(rel),
                    "train_exams": int(len(tr)),
                    "val_exams": int(len(va)),
                    "train_pos": _count_codes(tr, codes=set(spec.pos_codes)),
                    "train_neg": _count_codes(tr, codes=set(spec.neg_codes)),
                    "val_pos": _count_codes(va, codes=set(spec.pos_codes)),
                    "val_neg": _count_codes(va, codes=set(spec.neg_codes)),
                }
            )
        else:
            row.update(
                {
                    "train_exams": int(len(train)),
                    "val_exams": int(len(val)),
                }
            )
        tasks_summary.append(row)

    report = {
        "generated_at": now,
        "manifest_csv": str(manifest_csv),
        "splits_root": str(splits_root),
        "pct": int(pct),
        "cache": {},
        "manifest": manifest_summary,
        "splits": splits_summary,
        "outlier_cleaning": outlier_summary,
        "tasks": tasks_summary,
    }

    def _cache_stats(path: Path) -> dict:
        if not path.exists():
            return {"path": str(path), "exists": False}
        files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in (".npy", ".npz")]
        total_bytes = sum(p.stat().st_size for p in files)
        return {"path": str(path), "exists": True, "files": int(len(files)), "bytes": int(total_bytes)}

    report["cache"]["dual_cache"] = _cache_stats(dual_cache_dir)
    report["cache"]["ear_cache"] = _cache_stats(ear_cache_dir)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Markdown
    lines: list[str] = []
    lines.append("# Current Data / Tasks / Models Report")
    lines.append("")
    lines.append(f"- generated_at: `{now}`")
    lines.append(f"- manifest: `{manifest_csv}`")
    lines.append(f"- dual splits: `{splits_root}` (pct={pct})")
    lines.append("")

    lines.append("## Label Mapping (label_code 1..6)")
    lines.append("")
    lines.append("| label_code | label_id | name |")
    lines.append("|---:|---:|---|")
    for code in range(1, 7):
        label_id = code - 1
        name = CLASS_ID_TO_NAME.get(label_id, "?")
        lines.append(f"| {code} | {label_id} | {name} |")
    lines.append("")

    msum = manifest_summary
    lines.append("## Manifest (Ear-level)")
    lines.append("")
    lines.append(
        f"- ears(rows)={msum.get('rows')}  exams={msum.get('unique_exams')}  patients={msum.get('unique_patient_key_hash')}  series={msum.get('unique_series_uid')}"
    )
    lines.append(f"- has_label: true={msum.get('has_label_true')} false={msum.get('has_label_false')}")
    if msum.get("side_counts") is not None:
        lines.append(f"- side_counts: {json.dumps(msum['side_counts'], ensure_ascii=False)}")
    if msum.get("label_code_counts_labeled_ears") is not None:
        total = sum(msum["label_code_counts_labeled_ears"].values())
        parts = [f"{k}:{v}({_pct(v,total)})" for k, v in sorted(msum["label_code_counts_labeled_ears"].items())]
        lines.append("- label_code_counts(labeled ears): " + ", ".join(parts))
    if msum.get("date_match_counts_labeled_ears") is not None:
        lines.append(f"- date_match(labeled ears): {json.dumps(msum['date_match_counts_labeled_ears'], ensure_ascii=False)}")
    for k in ["manufacturer_top15_labeled_ears", "manufacturer_model_top15_labeled_ears", "convolution_kernel_top15_labeled_ears"]:
        if k in msum:
            lines.append(f"- {k}: {json.dumps(msum[k], ensure_ascii=False)}")
    for k in ["spacing_z_desc", "abs_spacing_z_desc", "slice_thickness_desc", "n_instances_desc"]:
        if k in msum:
            lines.append(f"- {k}: {_fmt_desc(msum[k])}")
    for k in ["date_range", "label_date_range", "folder_date_range"]:
        if k in msum:
            lines.append(f"- {k}: {json.dumps(msum[k], ensure_ascii=False)}")
    lines.append("")

    lines.append("## Caches (Preprocessed)")
    lines.append("")
    for k, label in [("dual_cache", "dual (3D)"), ("ear_cache", "ear2d")]:
        c = report["cache"].get(k, {})
        if not c:
            continue
        if not c.get("exists"):
            lines.append(f"- {label}: missing `{c.get('path')}`")
        else:
            gb = float(c.get("bytes", 0)) / (1024.0**3)
            lines.append(f"- {label}: `{c.get('path')}`  files={c.get('files')}  size={gb:.2f}GB")
    lines.append("")

    ssum = splits_summary
    lines.append("## Dual Splits (Exam-level)")
    lines.append("")
    lines.append(f"- train: {ssum['train_exams']} exams  ({ssum.get('train_unique_patients')} patients)")
    lines.append(f"- val:   {ssum['val_exams']} exams  ({ssum.get('val_unique_patients')} patients)")
    lines.append(
        f"- missing labels: train(left={ssum.get('train_missing_left_code')}, right={ssum.get('train_missing_right_code')})  "
        f"val(left={ssum.get('val_missing_left_code')}, right={ssum.get('val_missing_right_code')})"
    )
    if ssum.get("train_date_match_counts") is not None:
        lines.append(f"- train date_match: {json.dumps(ssum['train_date_match_counts'], ensure_ascii=False)}")
    if ssum.get("val_date_match_counts") is not None:
        lines.append(f"- val date_match: {json.dumps(ssum['val_date_match_counts'], ensure_ascii=False)}")
    lines.append(f"- train both-sides code counts: {json.dumps(ssum.get('train_both_sides_code_counts'), ensure_ascii=False)}")
    lines.append(f"- val both-sides code counts: {json.dumps(ssum.get('val_both_sides_code_counts'), ensure_ascii=False)}")
    if ssum.get("train_top_pairs_left_right_top15"):
        lines.append(
            f"- train top15 (left_code,right_code) pairs: {json.dumps(ssum['train_top_pairs_left_right_top15'], ensure_ascii=False)}"
        )
    if ssum.get("val_top_pairs_left_right_top15"):
        lines.append(
            f"- val top15 (left_code,right_code) pairs: {json.dumps(ssum['val_top_pairs_left_right_top15'], ensure_ascii=False)}"
        )
    lines.append("")

    if outlier_summary is not None:
        lines.append("## Outlier Cleaning (By (left_code,right_code) groups)")
        lines.append("")
        lines.append(f"- removed_rows: {outlier_summary.get('removed_rows')} (train-only by default)")
        lines.append(f"- removed_pair_counts_top15: {json.dumps(outlier_summary.get('removed_pair_counts_top15'), ensure_ascii=False)}")
        lines.append(f"- files: `{outlier_summary.get('outliers_removed_csv')}` `{outlier_summary.get('outliers_summary_csv')}`")
        lines.append("")

    lines.append("## Task Catalog (On This Split)")
    lines.append("")
    lines.append("| task | kind | classes | relevant_codes | train_exams | val_exams | train(pos,neg) | val(pos,neg) |")
    lines.append("|---|---|---:|---|---:|---:|---|---|")
    for row in tasks_summary:
        if row["kind"] == "binary":
            rel = ",".join(str(x) for x in row.get("relevant_codes", []))
            trpn = f"{row.get('train_pos')},{row.get('train_neg')}"
            vapn = f"{row.get('val_pos')},{row.get('val_neg')}"
        else:
            rel = "-"
            trpn = "-"
            vapn = "-"
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['num_classes']} | {rel} | {row['train_exams']} | {row['val_exams']} | {trpn} | {vapn} |"
        )
    lines.append("")

    # Keep the model/task list here (manual; this script doesn't introspect tmux).
    lines.append("## Current Dual Training Setup (What We Run Now)")
    lines.append("")
    lines.append("- launcher: `scripts/run_dual_models_200ep_max.sh`")
    lines.append("- models (3): `dual_resnet200_3d`, `dual_vit_3d`, `dual_unet_3d`")
    lines.append(
        "- tasks (6): `normal_vs_abnormal`, `normal_vs_cholesteatoma`, `normal_vs_csoma`, `normal_vs_ome`, "
        "`cholesteatoma_vs_other_abnormal`, `normal_vs_cholesterol_granuloma`"
    )
    lines.append(f"- split used: `{splits_root}/100pct` (train outliers removed)")
    lines.append("- epochs: 200  pct: 100")
    lines.append("- speed knobs: `--auto-batch` (max=128), `num_workers=64`, cache=float16")
    lines.append("- precision: AMP on, TF32 on, cudnn benchmark on, torch.compile off")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    typer.echo(f"wrote: {out_md}")
    typer.echo(f"wrote: {out_json}")


if __name__ == "__main__":
    app()
