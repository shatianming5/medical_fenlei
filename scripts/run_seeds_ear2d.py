from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


@app.command()
def main(
    seeds: str = typer.Option("0,1,2", help="多个 seed（逗号分隔，建议>=3）"),
    pct: int = typer.Option(20),
    label_task: str = typer.Option("normal_vs_diseased"),
    backbone: str = typer.Option("resnet18"),
    splits_root: Path = typer.Option(Path("artifacts/splits_dual")),
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv")),
    dicom_base: Path = typer.Option(Path("data/medical_data_2")),
    cache_dir: Path = typer.Option(Path("cache/ears_hu")),
    out_root: Path = typer.Option(Path("outputs/ear2d_seeds"), help="多个 seed 的输出根目录（不入库）"),
    epochs: int = typer.Option(50),
    batch_size: int = typer.Option(16),
    num_workers: int = typer.Option(8),
    early_stop_patience: int = typer.Option(10),
    early_stop_metric: str = typer.Option("auprc"),
    n_boot: int = typer.Option(500, help="eval bootstrap 次数"),
    dry_run: bool = typer.Option(False, help="只打印命令不执行"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing", help="如果已有 best.pt，则跳过训练"),
) -> None:
    seed_list = _parse_int_list(seeds)
    if not seed_list:
        raise typer.Exit(code=2)

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for sd in seed_list:
        run_dir = out_root / f"ear2d_{backbone}__{label_task}_{int(pct)}pct_seed{int(sd)}"
        run_dirs.append(run_dir)
        ckpt = run_dir / "checkpoints" / "best.pt"

        train_cmd = [
            py,
            "scripts/train_ear2d.py",
            "--splits-root",
            str(splits_root),
            "--pct",
            str(int(pct)),
            "--manifest-csv",
            str(manifest_csv),
            "--dicom-base",
            str(dicom_base),
            "--cache-dir",
            str(cache_dir),
            "--label-task",
            str(label_task),
            "--backbone",
            str(backbone),
            "--epochs",
            str(int(epochs)),
            "--batch-size",
            str(int(batch_size)),
            "--num-workers",
            str(int(num_workers)),
            "--early-stop-patience",
            str(int(early_stop_patience)),
            "--early-stop-metric",
            str(early_stop_metric),
            "--seed",
            str(int(sd)),
            "--output-dir",
            str(run_dir),
        ]

        eval_cmd = [
            py,
            "scripts/eval_ear2d.py",
            "--checkpoint",
            str(ckpt),
            "--splits-root",
            str(splits_root),
            "--pct",
            str(int(pct)),
            "--manifest-csv",
            str(manifest_csv),
            "--dicom-base",
            str(dicom_base),
            "--cache-dir",
            str(cache_dir),
            "--n-boot",
            str(int(n_boot)),
            "--seed",
            str(int(sd)),
            "--out-dir",
            str(run_dir),
        ]

        print("\n$ " + " ".join(train_cmd), flush=True)
        if not dry_run:
            if skip_existing and ckpt.exists():
                print(f"[skip] exists: {ckpt}", flush=True)
            else:
                subprocess.run(train_cmd, check=True, env=env)

        print("\n$ " + " ".join(eval_cmd), flush=True)
        if not dry_run:
            if ckpt.exists():
                subprocess.run(eval_cmd, check=True, env=env)
            else:
                print(f"[skip] missing: {ckpt}", flush=True)

    # Summarize across seeds (uses eval outputs).
    rows: list[dict] = []
    for run_dir in run_dirs:
        rep = run_dir / "reports" / "eval_binary.json"
        if not rep.exists():
            continue
        try:
            data = json.loads(rep.read_text(encoding="utf-8"))
        except Exception:
            continue
        val = data.get("val") or {}
        rows.append(
            {
                "run_dir": str(run_dir),
                "seed": int(str(run_dir.name).split("seed")[-1]),
                "auprc": val.get("auprc"),
                "auroc": val.get("auroc"),
                "accuracy": val.get("accuracy"),
                "sensitivity": val.get("sensitivity"),
                "specificity": val.get("specificity"),
                "sensitivity_at_spec": val.get("sensitivity_at_spec"),
                "specificity_at_spec": val.get("specificity_at_spec"),
                "f1": val.get("f1"),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        out_csv = out_root / f"summary_{label_task}_{int(pct)}pct.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nsummary saved: {out_csv}", flush=True)


if __name__ == "__main__":
    app()

