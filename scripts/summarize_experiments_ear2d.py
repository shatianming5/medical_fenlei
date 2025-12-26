from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class RunRow:
    run_dir: str
    backbone: str | None
    label_task: str | None
    pct: int | None
    seed: int | None
    auprc: float | None
    auroc: float | None
    accuracy: float | None
    sensitivity: float | None
    specificity: float | None
    sensitivity_at_spec: float | None
    specificity_at_spec: float | None
    f1: float | None
    status: str


_RUN_RE = re.compile(r"^ear2d_(?P<backbone>[^_]+)__?(?P<task>.+)_(?P<pct>\d+)pct_seed(?P<seed>\d+)$")


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _parse_run_name(name: str) -> tuple[str | None, str | None, int | None, int | None]:
    m = _RUN_RE.match(str(name))
    if not m:
        return None, None, None, None
    try:
        return str(m.group("backbone")), str(m.group("task")), int(m.group("pct")), int(m.group("seed"))
    except Exception:
        return None, None, None, None


def _iter_run_dirs(runs_root: Path) -> list[Path]:
    if runs_root.is_file():
        return [runs_root.parent]
    if not runs_root.exists():
        return []
    out: list[Path] = []
    for p in runs_root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "reports" / "eval_binary.json").exists():
            out.append(p)
    out.sort()
    return out


def _load_run(run_dir: Path) -> RunRow:
    rep = run_dir / "reports" / "eval_binary.json"
    backbone, label_task, pct, seed = _parse_run_name(run_dir.name)
    if not rep.exists():
        return RunRow(
            run_dir=str(run_dir),
            backbone=backbone,
            label_task=label_task,
            pct=pct,
            seed=seed,
            auprc=None,
            auroc=None,
            accuracy=None,
            sensitivity=None,
            specificity=None,
            sensitivity_at_spec=None,
            specificity_at_spec=None,
            f1=None,
            status="missing_eval",
        )

    try:
        data = json.loads(rep.read_text(encoding="utf-8"))
    except Exception:
        return RunRow(
            run_dir=str(run_dir),
            backbone=backbone,
            label_task=label_task,
            pct=pct,
            seed=seed,
            auprc=None,
            auroc=None,
            accuracy=None,
            sensitivity=None,
            specificity=None,
            sensitivity_at_spec=None,
            specificity_at_spec=None,
            f1=None,
            status="eval_parse_error",
        )

    val = data.get("val") or {}
    return RunRow(
        run_dir=str(run_dir),
        backbone=backbone,
        label_task=label_task or (data.get("task") or {}).get("name"),
        pct=pct,
        seed=seed,
        auprc=_safe_float(val.get("auprc")),
        auroc=_safe_float(val.get("auroc")),
        accuracy=_safe_float(val.get("accuracy")),
        sensitivity=_safe_float(val.get("sensitivity")),
        specificity=_safe_float(val.get("specificity")),
        sensitivity_at_spec=_safe_float(val.get("sensitivity_at_spec")),
        specificity_at_spec=_safe_float(val.get("specificity_at_spec")),
        f1=_safe_float(val.get("f1")),
        status="ok",
    )


def _fmt(v: float | None, *, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{float(v):.{digits}f}"


def _write_markdown(df: pd.DataFrame, path: Path, *, sort_metric: str, topk: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        path.write_text("# ear2d seeds summary\n\n(no runs found)\n", encoding="utf-8")
        return

    group_cols = ["backbone", "label_task", "pct"]
    metrics = [
        "auprc",
        "auroc",
        "accuracy",
        "sensitivity",
        "specificity",
        "sensitivity_at_spec",
        "specificity_at_spec",
        "f1",
    ]
    g = df[df["status"] == "ok"].groupby(group_cols, dropna=False)
    agg = g[metrics].agg(["mean", "std", "count"]).reset_index()
    # Flatten columns: metric_mean, metric_std, metric_count (count is shared but keep)
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    # Rename grouping cols back
    agg = agg.rename(columns={"backbone_": "backbone", "label_task_": "label_task", "pct_": "pct"})

    sort_key = f"{sort_metric}_mean"
    if sort_key not in agg.columns:
        sort_key = "auprc_mean"
    agg = agg.sort_values(sort_key, ascending=False).reset_index(drop=True)
    if int(topk) > 0:
        agg = agg.head(int(topk))

    lines: list[str] = []
    lines.append("# ear2d 多 seed 汇总（mean±std）")
    lines.append("")
    lines.append(f"- runs: {int((df['status']=='ok').sum())} ok / {int(len(df))} total")
    lines.append(f"- sort: {sort_key}")
    lines.append("")
    lines.append("| backbone | label_task | pct | n | auprc | auroc | sens@spec | spec@spec | f1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in agg.itertuples(index=False):
        n = int(getattr(r, "auprc_count", 0) or 0)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(getattr(r, "backbone") or "-"),
                    str(getattr(r, "label_task") or "-"),
                    str(int(getattr(r, "pct") or 0)),
                    str(n),
                    f"{_fmt(getattr(r, 'auprc_mean'))}±{_fmt(getattr(r, 'auprc_std'))}",
                    f"{_fmt(getattr(r, 'auroc_mean'))}±{_fmt(getattr(r, 'auroc_std'))}",
                    f"{_fmt(getattr(r, 'sensitivity_at_spec_mean'))}±{_fmt(getattr(r, 'sensitivity_at_spec_std'))}",
                    f"{_fmt(getattr(r, 'specificity_at_spec_mean'))}±{_fmt(getattr(r, 'specificity_at_spec_std'))}",
                    f"{_fmt(getattr(r, 'f1_mean'))}±{_fmt(getattr(r, 'f1_std'))}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## per-run 明细")
    lines.append("")
    lines.append("| run_dir | seed | auprc | auroc | sens@spec | f1 | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in df.sort_values(["status", "label_task", "pct", "seed", "run_dir"]).itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.run_dir),
                    str(r.seed if pd.notna(r.seed) else "-"),
                    _fmt(r.auprc),
                    _fmt(r.auroc),
                    _fmt(r.sensitivity_at_spec),
                    _fmt(r.f1),
                    str(r.status),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command()
def main(
    runs_root: Path = typer.Option(Path("outputs/ear2d_seeds"), help="run_seeds_ear2d 的输出根目录（不入库）"),
    out_csv: Path = typer.Option(Path("outputs/ear2d_seeds_summary.csv"), help="逐 run 明细 CSV（不入库）"),
    out_md: Path = typer.Option(Path("outputs/ear2d_seeds_summary.md"), help="mean±std 汇总 Markdown（不入库）"),
    sort_metric: str = typer.Option("auprc", help="排序指标：auprc | auroc | f1 | sensitivity_at_spec ..."),
    topk: int = typer.Option(50, help="Markdown 汇总只保留 topk 个组合（0=不截断）"),
) -> None:
    run_dirs = _iter_run_dirs(Path(runs_root))
    if not run_dirs:
        typer.echo(f"no runs found under: {runs_root}")
        raise typer.Exit(code=0)

    rows = [_load_run(p) for p in run_dirs]
    df = pd.DataFrame([r.__dict__ for r in rows])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _write_markdown(df, out_md, sort_metric=str(sort_metric), topk=int(topk))

    typer.echo(f"saved: {out_csv}")
    typer.echo(f"saved: {out_md}")


if __name__ == "__main__":
    app()

