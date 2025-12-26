from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from medical_fenlei.constants import CLASS_ID_TO_NAME

app = typer.Typer(add_completion=False)


def _code_to_label(code) -> int | None:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    try:
        code_int = int(code)
    except Exception:
        return None
    if not (1 <= code_int <= 6):
        return None
    return int(code_int - 1)


def _expand_side_rows(exam_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for r in exam_df.itertuples(index=False):
        for side in ("left", "right"):
            label = _code_to_label(getattr(r, f"{side}_code", None))
            if label is None:
                continue
            rows.append({"exam_id": int(r.exam_id), "label": int(label), "side": side})
    return pd.DataFrame(rows)


def _side_label_counts(exam_df: pd.DataFrame) -> dict[int, int]:
    side_df = _expand_side_rows(exam_df)
    if side_df.empty:
        return {k: 0 for k in CLASS_ID_TO_NAME.keys()}
    vc = side_df["label"].value_counts().to_dict()
    return {int(k): int(vc.get(k, 0)) for k in sorted(CLASS_ID_TO_NAME.keys())}


def _write_stats(path: Path, *, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame | None = None) -> None:
    def _counts(df: pd.DataFrame) -> dict:
        counts = _side_label_counts(df)
        return {CLASS_ID_TO_NAME[int(k)]: int(v) for k, v in counts.items()}

    stats = {"train_side_counts": _counts(train_df), "val_side_counts": _counts(val_df)}
    if test_df is not None:
        stats["test_side_counts"] = _counts(test_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _select_balanced_exams(
    train_exam_df: pd.DataFrame,
    *,
    ratio: float,
    seed: int,
    min_per_class: int,
) -> pd.DataFrame:
    if ratio >= 1.0:
        return train_exam_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    side_df = _expand_side_rows(train_exam_df)
    if side_df.empty:
        return train_exam_df.iloc[0:0].copy()

    totals = side_df["label"].value_counts().to_dict()
    desired: dict[int, int] = {}
    for label in sorted(CLASS_ID_TO_NAME.keys()):
        n = int(totals.get(label, 0))
        if n <= 0:
            desired[label] = 0
            continue
        k = int(round(n * ratio))
        k = max(int(min_per_class), k)
        k = min(k, n)
        desired[label] = int(k)

    exam_to_labels: dict[int, list[int]] = defaultdict(list)
    for r in side_df.itertuples(index=False):
        exam_to_labels[int(r.exam_id)].append(int(r.label))

    rng = np.random.default_rng(int(seed))
    exam_ids_by_label: dict[int, list[int]] = {}
    for label in sorted(CLASS_ID_TO_NAME.keys()):
        cand = side_df.loc[side_df["label"] == label, "exam_id"].astype(int).unique().tolist()
        rng.shuffle(cand)
        exam_ids_by_label[label] = cand

    selected: set[int] = set()
    current: dict[int, int] = {label: 0 for label in desired.keys()}
    ptr: dict[int, int] = {label: 0 for label in desired.keys()}

    def done() -> bool:
        return all(current[l] >= desired[l] for l in desired.keys())

    while not done():
        progressed = False
        for label in sorted(desired.keys()):
            if current[label] >= desired[label]:
                continue
            cand = exam_ids_by_label.get(label, [])
            while ptr[label] < len(cand) and cand[ptr[label]] in selected:
                ptr[label] += 1
            if ptr[label] >= len(cand):
                continue
            exam_id = int(cand[ptr[label]])
            ptr[label] += 1

            selected.add(exam_id)
            for lab in exam_to_labels.get(exam_id, []):
                current[int(lab)] += 1
            progressed = True
        if not progressed:
            break

    out = train_exam_df[train_exam_df["exam_id"].astype(int).isin(selected)].copy()
    out = out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return out


@app.command()
def main(
    splits_root: Path = typer.Option(Path("artifacts/splits_dual_patient_clustered_v1"), help="包含 100pct/train.csv 的目录"),
    pcts: str = typer.Option("1,20", help="要生成的 pct 列表（逗号分隔，不含 100 也可以）"),
    seed: int = typer.Option(42),
    min_per_class: int = typer.Option(1, help="每类最少保留样本数（按耳朵计）"),
) -> None:
    base_dir = Path(splits_root) / "100pct"
    train_csv = base_dir / "train.csv"
    val_csv = base_dir / "val.csv"
    test_csv = base_dir / "test.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    has_test = test_csv.exists()

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv) if has_test else None
    if train_df.empty or val_df.empty:
        raise typer.Exit(code=2)

    pct_list = [int(x.strip()) for x in str(pcts).split(",") if x.strip()]
    for pct in pct_list:
        if pct <= 0 or pct > 100:
            raise ValueError(f"invalid pct: {pct}")
        ratio = float(pct) / 100.0

        split_dir = Path(splits_root) / f"{pct}pct"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_sub = _select_balanced_exams(train_df, ratio=ratio, seed=int(seed), min_per_class=int(min_per_class))
        (split_dir / "train.csv").write_text(train_sub.to_csv(index=False), encoding="utf-8")
        (split_dir / "val.csv").write_text(val_df.to_csv(index=False), encoding="utf-8")
        if has_test and test_df is not None:
            (split_dir / "test.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")
            _write_stats(split_dir / "stats.json", train_df=train_sub, val_df=val_df, test_df=test_df)
        else:
            _write_stats(split_dir / "stats.json", train_df=train_sub, val_df=val_df)

        test_n = int(len(test_df)) if test_df is not None else 0
        typer.echo(f"{pct}%: train_exams={len(train_sub)} val_exams={len(val_df)} test_exams={test_n} -> {split_dir}")


if __name__ == "__main__":
    app()
