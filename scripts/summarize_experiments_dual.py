from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    model: str
    pct: int
    epochs_ran: int
    batch_size: int | None
    best_epoch: int | None
    best_metric: float | None
    best_val_loss: float | None
    best_accuracy: float | None
    best_macro_recall: float | None
    best_macro_specificity: float | None
    best_weighted_f1: float | None
    status: str


_PCT_RE = re.compile(r"_(?P<pct>\d+)pct_")


def _parse_run_name(name: str) -> tuple[str, int] | None:
    m = _PCT_RE.search(name)
    if not m:
        return None
    pct = int(m.group("pct"))
    model = name[: m.start()]
    return model, pct


def _iter_metrics(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _get_metric(rec: dict[str, Any], key: str) -> float | None:
    vm = rec.get("val_metrics") or {}
    if key in vm:
        try:
            return float(vm[key])
        except Exception:
            return None
    if key == "val_loss":
        try:
            return float((rec.get("val") or {}).get("loss"))
        except Exception:
            return None
    return None


def _summarize_run(run_dir: Path, *, metric: str) -> RunSummary:
    parsed = _parse_run_name(run_dir.name)
    model, pct = (parsed if parsed is not None else (run_dir.name, -1))

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return RunSummary(
            run_dir=str(run_dir),
            model=str(model),
            pct=int(pct),
            epochs_ran=0,
            batch_size=None,
            best_epoch=None,
            best_metric=None,
            best_val_loss=None,
            best_accuracy=None,
            best_macro_recall=None,
            best_macro_specificity=None,
            best_weighted_f1=None,
            status="no_metrics",
        )

    try:
        recs = _iter_metrics(metrics_path)
    except Exception:
        return RunSummary(
            run_dir=str(run_dir),
            model=str(model),
            pct=int(pct),
            epochs_ran=0,
            batch_size=None,
            best_epoch=None,
            best_metric=None,
            best_val_loss=None,
            best_accuracy=None,
            best_macro_recall=None,
            best_macro_specificity=None,
            best_weighted_f1=None,
            status="metrics_parse_error",
        )

    if not recs:
        return RunSummary(
            run_dir=str(run_dir),
            model=str(model),
            pct=int(pct),
            epochs_ran=0,
            batch_size=None,
            best_epoch=None,
            best_metric=None,
            best_val_loss=None,
            best_accuracy=None,
            best_macro_recall=None,
            best_macro_specificity=None,
            best_weighted_f1=None,
            status="empty_metrics",
        )

    best_epoch = None
    best_val = None
    best_val_loss = None
    best_acc = None
    best_macro_recall = None
    best_macro_spec = None
    best_weighted_f1 = None
    batch_size = None

    if metric == "val_loss":
        best_cmp = float("inf")
        better = lambda v, b: v < b
    else:
        best_cmp = float("-inf")
        better = lambda v, b: v > b

    for rec in recs:
        if batch_size is None:
            try:
                batch_size = int(rec.get("batch_size"))
            except Exception:
                batch_size = None

        v = _get_metric(rec, metric)
        if v is None:
            continue
        if better(float(v), float(best_cmp)):
            best_cmp = float(v)
            try:
                best_epoch = int(rec.get("epoch"))
            except Exception:
                best_epoch = None
            best_val = float(v)
            best_val_loss = _get_metric(rec, "val_loss")
            best_acc = _get_metric(rec, "accuracy")
            best_macro_recall = _get_metric(rec, "macro_recall")
            best_macro_spec = _get_metric(rec, "macro_specificity")
            best_weighted_f1 = _get_metric(rec, "weighted_f1")

    status = "ok"
    if (run_dir / "checkpoints" / "last.pt").exists():
        status = "ok"
    elif (run_dir / "checkpoints").exists():
        status = "incomplete"

    return RunSummary(
        run_dir=str(run_dir),
        model=str(model),
        pct=int(pct),
        epochs_ran=int(len(recs)),
        batch_size=batch_size,
        best_epoch=best_epoch,
        best_metric=best_val,
        best_val_loss=best_val_loss,
        best_accuracy=best_acc,
        best_macro_recall=best_macro_recall,
        best_macro_specificity=best_macro_spec,
        best_weighted_f1=best_weighted_f1,
        status=status,
    )


def _write_csv(rows: list[RunSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pct",
                "model",
                "status",
                "epochs_ran",
                "batch_size",
                "best_epoch",
                "best_metric",
                "best_val_loss",
                "best_accuracy",
                "best_macro_recall",
                "best_macro_specificity",
                "best_weighted_f1",
                "run_dir",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.pct,
                    r.model,
                    r.status,
                    r.epochs_ran,
                    r.batch_size,
                    r.best_epoch,
                    r.best_metric,
                    r.best_val_loss,
                    r.best_accuracy,
                    r.best_macro_recall,
                    r.best_macro_specificity,
                    r.best_weighted_f1,
                    r.run_dir,
                ]
            )


def _fmt(v: float | None, *, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{float(v):.{digits}f}"


def _write_markdown(rows: list[RunSummary], path: Path, *, metric: str, topk: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_pct: dict[int, list[RunSummary]] = {}
    for r in rows:
        by_pct.setdefault(int(r.pct), []).append(r)

    def sort_key(r: RunSummary) -> tuple[int, float]:
        is_ok = 1 if r.status == "ok" else 0
        v = r.best_metric
        if v is None:
            score = float("-inf")
        else:
            score = float(v)
            if metric == "val_loss":
                score = -score  # smaller val_loss is better
        return is_ok, score

    lines: list[str] = []
    lines.append("# Dual Experiments Summary\n")
    lines.append(f"- metric: `{metric}`\n")
    lines.append("- note: `outputs/` is gitignored; this file summarizes local runs.\n")

    for pct in sorted(by_pct.keys()):
        if pct < 0:
            continue
        lines.append(f"\n## {pct}%\n")
        runs = sorted(by_pct[pct], key=sort_key, reverse=True)[: int(topk)]
        lines.append("| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        r.model,
                        r.status,
                        str(r.best_epoch) if r.best_epoch is not None else "-",
                        _fmt(r.best_metric),
                        _fmt(r.best_val_loss),
                        _fmt(r.best_accuracy),
                        _fmt(r.best_macro_recall),
                        _fmt(r.best_macro_specificity),
                        _fmt(r.best_weighted_f1),
                        str(r.batch_size) if r.batch_size is not None else "-",
                        f"`{Path(r.run_dir)}`",
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


@app.command()
def main(
    outputs_dir: Path = typer.Option(Path("outputs"), help="训练输出目录（默认 outputs/）"),
    metric: str = typer.Option("macro_f1", help="排序用指标：macro_f1 | macro_recall | weighted_f1 | accuracy | val_loss"),
    out_csv: Path = typer.Option(Path("docs/experiments_dual_summary.csv")),
    out_md: Path = typer.Option(Path("docs/EXPERIMENTS_DUAL_SUMMARY.md")),
    topk: int = typer.Option(5, help="每个 pct 展示 top-k"),
) -> None:
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        raise typer.Exit(code=2)

    run_dirs = [p for p in outputs_dir.iterdir() if p.is_dir()]
    rows: list[RunSummary] = []
    for d in sorted(run_dirs, key=lambda p: p.name):
        parsed = _parse_run_name(d.name)
        if parsed is None:
            continue
        rows.append(_summarize_run(d, metric=str(metric)))

    # Sort for CSV: pct asc, metric desc (val_loss asc).
    def metric_sort(r: RunSummary) -> float:
        if r.best_metric is None:
            return float("-inf") if metric != "val_loss" else float("inf")
        return float(r.best_metric)

    if metric == "val_loss":
        rows_sorted = sorted(rows, key=lambda r: (r.pct, metric_sort(r), r.model))
    else:
        rows_sorted = sorted(rows, key=lambda r: (r.pct, -metric_sort(r), r.model))

    _write_csv(rows_sorted, Path(out_csv))
    _write_markdown(rows_sorted, Path(out_md), metric=str(metric), topk=int(topk))
    typer.echo(f"wrote: {out_csv}")
    typer.echo(f"wrote: {out_md}")


if __name__ == "__main__":
    app()
