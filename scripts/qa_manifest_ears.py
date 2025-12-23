from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from medical_fenlei.tasks import TASKS

app = typer.Typer(add_completion=False)


def _load_exam_ids(path: Path) -> set[int]:
    df = pd.read_csv(path)
    if "exam_id" not in df.columns:
        raise ValueError(f"missing exam_id in {path}")
    return set(df["exam_id"].astype(int).tolist())


def _describe_numeric(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p10": float(q.get(0.1)),
        "p25": float(q.get(0.25)),
        "p50": float(q.get(0.5)),
        "p75": float(q.get(0.75)),
        "p90": float(q.get(0.9)),
        "max": float(s.max()),
    }


@app.command()
def main(
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), exists=True, help="由 scripts/build_manifest_ears.py 生成（不入库）"),
    train_csv: Path = typer.Option(Path("artifacts/splits_dual/100pct/train.csv"), exists=True),
    val_csv: Path = typer.Option(Path("artifacts/splits_dual/100pct/val.csv"), exists=True),
    out_dir: Path = typer.Option(Path("artifacts/qa_manifest"), help="QA 输出目录（不入库）"),
) -> None:
    man = pd.read_csv(manifest_csv)
    if man.empty:
        raise typer.Exit(code=2)

    train_exam_ids = _load_exam_ids(train_csv)
    val_exam_ids = _load_exam_ids(val_csv)

    df = man.loc[man["has_label"].fillna(False)].copy()
    df["split"] = df["exam_id"].apply(lambda x: "train" if int(x) in train_exam_ids else ("val" if int(x) in val_exam_ids else "other"))
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scan params distribution (train vs val) + date_match flag.
    numeric_cols = [c for c in ["spacing_x", "spacing_y", "spacing_z", "slice_thickness", "spacing_between_slices", "n_instances"] if c in df.columns]
    cat_cols = [c for c in ["manufacturer", "manufacturer_model", "convolution_kernel", "modality"] if c in df.columns]

    scan_rows: list[dict] = []
    for split in ["train", "val"]:
        for date_match in [True, False]:
            sub = df[(df["split"] == split) & (df["date_match"].fillna(False) == date_match)]
            row = {"split": split, "date_match": bool(date_match), "ears": int(len(sub))}
            for c in numeric_cols:
                stats = _describe_numeric(sub[c])
                for k, v in stats.items():
                    row[f"{c}__{k}"] = v
            scan_rows.append(row)

    pd.DataFrame(scan_rows).to_csv(out_dir / "scan_params_numeric.csv", index=False)

    for c in cat_cols:
        vc = df.groupby(["split", "date_match"])[c].value_counts(dropna=False).rename("count").reset_index()
        vc.to_csv(out_dir / f"scan_params_{c}.csv", index=False)

    # 2) Per-task counts (val) for binary tasks.
    task_rows: list[dict] = []
    val_df = df[df["split"] == "val"].copy()
    for name, task in TASKS.items():
        if task.kind != "binary":
            continue
        valid_col = f"task_valid__{name}"
        y_col = f"task_y__{name}"
        if valid_col not in val_df.columns or y_col not in val_df.columns:
            continue
        sub = val_df[val_df[valid_col].fillna(False)]
        y = pd.to_numeric(sub[y_col], errors="coerce")
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        task_rows.append({"task": name, "val_ears": int(len(sub)), "val_pos": pos, "val_neg": neg})

    pd.DataFrame(task_rows).sort_values(["val_ears", "task"], ascending=[False, True]).to_csv(out_dir / "tasks_val_counts.csv", index=False)

    meta = {
        "manifest": str(manifest_csv),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "rows_labeled_used": int(len(df)),
        "out_dir": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote: {out_dir}")


if __name__ == "__main__":
    app()

