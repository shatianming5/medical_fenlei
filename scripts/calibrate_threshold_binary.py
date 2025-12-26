from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import typer

from medical_fenlei.metrics import binary_metrics

app = typer.Typer(add_completion=False)


Objective = Literal["f1", "youden", "accuracy"]


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray, *, objective: Objective) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float64)
    if y_true.size <= 0:
        return 0.5

    # Candidate thresholds: unique probs + boundaries.
    uniq = np.unique(y_prob)
    if uniq.size <= 0:
        return 0.5
    cand = np.concatenate([np.asarray([0.0], dtype=np.float64), uniq, np.asarray([1.0], dtype=np.float64)])

    best_thr = 0.5
    best_score = float("-inf")
    for thr in cand:
        m = binary_metrics(y_true, y_prob, threshold=float(thr))
        if objective == "accuracy":
            score = _safe_float(m.get("accuracy"))
        elif objective == "youden":
            sens = _safe_float(m.get("sensitivity"))
            spec = _safe_float(m.get("specificity"))
            score = (sens if sens is not None else 0.0) + (spec if spec is not None else 0.0) - 1.0
        else:
            score = _safe_float(m.get("f1"))
        if score is None:
            continue
        if float(score) > float(best_score):
            best_score = float(score)
            best_thr = float(thr)
    return float(best_thr)


@app.command()
def main(
    pred_csv: Path = typer.Option(..., exists=True, help="scripts/eval_ear2d.py 生成的 reports/predictions_val.csv"),
    objective: Objective = typer.Option("f1", help="阈值选择目标：f1 | youden | accuracy"),
    specificity_target: float = typer.Option(0.95, help="同时报告 sensitivity@spec 的阈值"),
    out_json: Path | None = typer.Option(None, help="输出 json（默认同目录 calib_threshold.json；不入库）"),
) -> None:
    df = pd.read_csv(pred_csv)
    if df.empty:
        raise typer.Exit(code=2)
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        raise ValueError("pred_csv must contain columns: y_true, y_prob")

    y_true = df["y_true"].astype(int).to_numpy()
    y_prob = df["y_prob"].astype(float).to_numpy()

    thr = _best_threshold(y_true, y_prob, objective=str(objective))  # type: ignore[arg-type]
    m_obj = binary_metrics(y_true, y_prob, threshold=float(thr), specificity_target=float(specificity_target))
    m_default = binary_metrics(y_true, y_prob, threshold=0.5, specificity_target=float(specificity_target))

    out = {
        "pred_csv": str(pred_csv),
        "objective": str(objective),
        "threshold_best": float(thr),
        "metrics_best": {k: v for k, v in m_obj.items() if k not in ("t0.500_tp", "t0.500_tn", "t0.500_fp", "t0.500_fn")},
        "metrics_default_0.5": {k: v for k, v in m_default.items() if k not in ("t0.500_tp", "t0.500_tn", "t0.500_fp", "t0.500_fn")},
    }

    out_path = out_json or (pred_csv.parent / "calib_threshold.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"saved: {out_path}")


if __name__ == "__main__":
    app()

