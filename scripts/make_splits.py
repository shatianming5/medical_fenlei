from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from medical_fenlei.constants import CLASS_ID_TO_NAME

app = typer.Typer(add_completion=False)


def _to_side_df(index_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for r in index_df.itertuples(index=False):
        for side in ("left", "right"):
            code = getattr(r, f"{side}_code", None)
            if code is None or (isinstance(code, float) and np.isnan(code)):
                continue
            try:
                code_int = int(code)
            except Exception:
                continue
            if not (1 <= code_int <= len(CLASS_ID_TO_NAME)):
                continue
            rows.append(
                {
                    "exam_id": int(r.exam_id),
                    "date": str(r.date),
                    "series_relpath": str(r.series_relpath),
                    "side": side,
                    "label_code": code_int,
                    "label": code_int - 1,
                }
            )
    return pd.DataFrame(rows)


def _downsample_per_class(df: pd.DataFrame, *, ratio: float, seed: int, min_per_class: int) -> pd.DataFrame:
    if ratio >= 1.0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for label, g in df.groupby("label"):
        n = len(g)
        k = int(round(n * ratio))
        k = max(min_per_class, k)
        k = min(k, n)
        parts.append(g.sample(n=k, random_state=seed))

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _write_stats(path: Path, *, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    def _counts(df: pd.DataFrame) -> dict:
        vc = df["label"].value_counts().to_dict()
        return {CLASS_ID_TO_NAME[int(k)]: int(v) for k, v in sorted(vc.items(), key=lambda x: int(x[0]))}

    stats = {"train": _counts(train_df), "val": _counts(val_df)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@app.command()
def main(
    index_csv: Path = typer.Option(Path("artifacts/dataset_index.csv"), exists=True, help="由 scripts/build_index.py 生成"),
    out_dir: Path = typer.Option(Path("artifacts/splits"), help="输出目录（不入库）"),
    val_ratio: float = typer.Option(0.2, help="验证集比例"),
    ratios: str = typer.Option("0.01,0.2,1.0", help="训练集抽样比例（逗号分隔）"),
    seed: int = typer.Option(42),
    min_per_class: int = typer.Option(1, help="每类最少保留样本数（避免 1% 时缺类）"),
) -> None:
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    side_df = _to_side_df(df)
    if side_df.empty:
        typer.echo("no labeled rows found after expanding left/right")
        raise typer.Exit(code=2)

    train_df, val_df = train_test_split(
        side_df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=side_df["label"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_list = [float(x.strip()) for x in ratios.split(",") if x.strip()]
    for ratio in ratio_list:
        pct = int(round(ratio * 100))
        split_dir = out_dir / f"{pct}pct"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_sub = _downsample_per_class(train_df, ratio=ratio, seed=seed, min_per_class=min_per_class)

        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"
        stats_path = split_dir / "stats.json"

        train_sub.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        _write_stats(stats_path, train_df=train_sub, val_df=val_df)

        typer.echo(f"{pct}%: train={len(train_sub)} val={len(val_df)} -> {split_dir}")


if __name__ == "__main__":
    app()

