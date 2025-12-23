from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from medical_fenlei.constants import CLASS_ID_TO_NAME

app = typer.Typer(add_completion=False)


def _code_to_label(code) -> int | None:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    try:
        code_int = int(code)
    except Exception:
        return None
    if not (1 <= code_int <= len(CLASS_ID_TO_NAME)):
        return None
    return code_int - 1


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


def _write_stats(path: Path, *, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    def _counts(df: pd.DataFrame) -> dict:
        counts = _side_label_counts(df)
        return {CLASS_ID_TO_NAME[int(k)]: int(v) for k, v in counts.items()}

    stats = {"train_side_counts": _counts(train_df), "val_side_counts": _counts(val_df)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stratify_key(df: pd.DataFrame) -> pd.Series:
    def _k(row) -> str:
        l = _code_to_label(row.get("left_code"))
        r = _code_to_label(row.get("right_code"))
        l_s = "NA" if l is None else str(int(l))
        r_s = "NA" if r is None else str(int(r))
        return f"L{l_s}_R{r_s}"

    return df.apply(_k, axis=1)


def _primary_label(df: pd.DataFrame) -> pd.Series:
    def _p(row) -> int:
        l = _code_to_label(row.get("left_code"))
        if l is not None:
            return int(l)
        r = _code_to_label(row.get("right_code"))
        if r is not None:
            return int(r)
        return -1

    return df.apply(_p, axis=1)


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
        k = max(min_per_class, k)
        k = min(k, n)
        desired[label] = int(k)

    # exam_id -> labels present (one or two)
    exam_to_labels: dict[int, list[int]] = defaultdict(list)
    for r in side_df.itertuples(index=False):
        exam_to_labels[int(r.exam_id)].append(int(r.label))

    rng = np.random.default_rng(seed)
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
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


@app.command()
def main(
    index_csv: Path = typer.Option(Path("artifacts/dataset_index.csv"), exists=True, help="由 scripts/build_index.py 生成"),
    out_dir: Path = typer.Option(Path("artifacts/splits_dual"), help="输出目录（不入库）"),
    val_ratio: float = typer.Option(0.2, help="验证集比例（按 exam_id 切分，避免左右耳泄漏）"),
    ratios: str = typer.Option("0.01,0.2,1.0", help="训练集抽样比例（逗号分隔）"),
    seed: int = typer.Option(42),
    min_per_class: int = typer.Option(1, help="每类最少保留样本数（按耳朵计）"),
) -> None:
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    # keep exams that have at least one labeled side
    left_ok = df["left_code"].notna()
    right_ok = df["right_code"].notna()
    df = df[left_ok | right_ok].reset_index(drop=True)
    if df.empty:
        typer.echo("no labeled exams found (left_code/right_code are all empty)")
        raise typer.Exit(code=2)

    key = _stratify_key(df)
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=key,
        )
    except ValueError:
        # fallback: stratify by a single label to avoid sparse pair classes
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=_primary_label(df),
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_list = [float(x.strip()) for x in ratios.split(",") if x.strip()]
    for ratio in ratio_list:
        pct = int(round(ratio * 100))
        split_dir = out_dir / f"{pct}pct"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_sub = _select_balanced_exams(train_df, ratio=ratio, seed=seed, min_per_class=min_per_class)

        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"
        stats_path = split_dir / "stats.json"

        train_sub.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        _write_stats(stats_path, train_df=train_sub, val_df=val_df)

        typer.echo(f"{pct}%: train_exams={len(train_sub)} val_exams={len(val_df)} -> {split_dir}")


if __name__ == "__main__":
    app()

