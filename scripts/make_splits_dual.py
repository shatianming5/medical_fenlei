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


def _parse_int_set(value: str) -> set[int]:
    parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
    return {int(x) for x in parts} if parts else set()


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


def _write_stats(path: Path, *, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame | None = None) -> None:
    def _counts(df: pd.DataFrame) -> dict:
        counts = _side_label_counts(df)
        return {CLASS_ID_TO_NAME[int(k)]: int(v) for k, v in counts.items()}

    stats = {"train_side_counts": _counts(train_df), "val_side_counts": _counts(val_df)}
    # Optional 3-way split support (check.md 5.1: train/val/test = 70/10/20).
    if test_df is not None:
        stats["test_side_counts"] = _counts(test_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _apply_code_filters(
    df: pd.DataFrame,
    *,
    keep_codes: set[int] | None = None,
    drop_codes: set[int] | None = None,
) -> pd.DataFrame:
    """
    Filter left_code/right_code in-place by setting unwanted codes to NaN.

    Codes are 1-based (same as the original XLSX labels and check.md "Label N").
    Rows with both sides empty after filtering are dropped.
    """
    keep = set(int(x) for x in (keep_codes or set()))
    drop = set(int(x) for x in (drop_codes or set()))

    for x in keep | drop:
        if x < 1 or x > len(CLASS_ID_TO_NAME):
            raise ValueError(f"code out of range: {x} (expected 1..{len(CLASS_ID_TO_NAME)})")

    out = df.copy()
    for col in ("left_code", "right_code"):
        if col not in out.columns:
            continue

        def _f(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            try:
                c = int(v)
            except Exception:
                return np.nan
            if c < 1 or c > len(CLASS_ID_TO_NAME):
                return np.nan
            if keep and c not in keep:
                return np.nan
            if drop and c in drop:
                return np.nan
            return float(c)

        out[col] = out[col].map(_f)

    left_ok = out.get("left_code", pd.Series([], dtype=float)).notna() if "left_code" in out.columns else pd.Series(False, index=out.index)
    right_ok = out.get("right_code", pd.Series([], dtype=float)).notna() if "right_code" in out.columns else pd.Series(False, index=out.index)
    out = out[left_ok | right_ok].reset_index(drop=True)
    return out


def _drop_exams_with_codes(df: pd.DataFrame, *, drop_codes: set[int]) -> pd.DataFrame:
    """
    Drop whole exams that contain any of the given label codes (1..6) on either side.

    This matches the Dual input granularity (one sample == one exam with both ears),
    and is the recommended way to construct check.md's zero-shot Setting B.
    """
    drop = set(int(x) for x in (drop_codes or set()))
    if not drop:
        return df
    for x in drop:
        if x < 1 or x > len(CLASS_ID_TO_NAME):
            raise ValueError(f"code out of range: {x} (expected 1..{len(CLASS_ID_TO_NAME)})")

    out = df.copy()
    for col in ("left_code", "right_code"):
        if col not in out.columns:
            continue
        codes = pd.to_numeric(out[col], errors="coerce")
        out[col] = codes
    m = pd.Series(False, index=out.index)
    if "left_code" in out.columns:
        m |= out["left_code"].isin(sorted(drop))
    if "right_code" in out.columns:
        m |= out["right_code"].isin(sorted(drop))
    out = out.loc[~m].reset_index(drop=True)
    return out


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


def _patient_primary_label(exams: pd.DataFrame) -> int:
    side_df = _expand_side_rows(exams)
    if side_df.empty:
        return -1
    vc = side_df["label"].value_counts()
    try:
        return int(vc.index[0])
    except Exception:
        return -1


def _attach_patient_key(
    df: pd.DataFrame,
    *,
    manifest_csv: Path,
    patient_key_col: str,
) -> pd.DataFrame:
    man = pd.read_csv(manifest_csv, usecols=["exam_id", patient_key_col])
    man[patient_key_col] = man[patient_key_col].fillna("").astype(str)
    man = man[man[patient_key_col].str.len() > 0].copy()
    exam_to_patient = man.groupby("exam_id")[patient_key_col].first().to_dict()

    out = df.copy()
    out["patient_key"] = out["exam_id"].astype(int).map(lambda x: exam_to_patient.get(int(x), ""))
    out["patient_key"] = out["patient_key"].fillna("").astype(str)
    # fallback: use exam_id as unique key when patient key missing
    m = out["patient_key"].str.len() <= 0
    out.loc[m, "patient_key"] = out.loc[m, "exam_id"].astype(int).map(lambda x: f"EXAM_{int(x)}")
    return out


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
    val_ratio: float = typer.Option(0.1, help="验证集比例（按 exam_id 切分，避免左右耳泄漏）"),
    test_ratio: float = typer.Option(0.2, help="测试集比例（按 exam_id 切分，避免左右耳泄漏；0=不生成 test.csv）"),
    keep_codes: str = typer.Option("", help="仅保留这些 label_code（1..6；逗号分隔；空=不过滤），用于设置A/控制训练-测试类别范围"),
    keep_mode: str = typer.Option(
        "mask_side",
        help="keep_codes 的行为：drop_exam（删掉包含其他类别的整条 exam 样本；更严格） | mask_side（仅将该侧 label 置空）",
    ),
    drop_train_codes: str = typer.Option("", help="仅对 train.csv 生效：移除这些 label_code（1..6；逗号分隔），用于设置B/零样本模拟"),
    drop_train_mode: str = typer.Option(
        "drop_exam",
        help="drop_train_codes 的行为：drop_exam（删掉包含该类的整条 exam 样本；推荐） | mask_side（仅将该侧 label 置空）",
    ),
    ratios: str = typer.Option("0.01,0.2,1.0", help="训练集抽样比例（逗号分隔）"),
    seed: int = typer.Option(42),
    min_per_class: int = typer.Option(1, help="每类最少保留样本数（按耳朵计）"),
    manifest_csv: Path | None = typer.Option(
        None,
        help="可选：耳朵级 manifest（由 scripts/build_manifest_ears.py 生成），用于 patient-level split",
    ),
    patient_key_col: str = typer.Option("patient_key_hash", help="manifest 中的 patient key 列名（建议 hash）"),
    patient_split: bool = typer.Option(False, "--patient-split/--no-patient-split", help="按 patient_key 切分 train/val，避免同一病人跨集合"),
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

    val_ratio_v = float(val_ratio)
    test_ratio_v = float(test_ratio)
    if val_ratio_v < 0 or test_ratio_v < 0 or (val_ratio_v + test_ratio_v) >= 1.0:
        raise ValueError(f"invalid split ratios: val_ratio={val_ratio_v:g} test_ratio={test_ratio_v:g} (need val>=0, test>=0, val+test<1)")

    def _split_train_holdout(
        df_in: pd.DataFrame,
        *,
        holdout_ratio: float,
        stratify: pd.Series | None,
        stratify_fallback: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if holdout_ratio <= 0:
            return df_in.sample(frac=1.0, random_state=seed).reset_index(drop=True), df_in.iloc[0:0].copy()
        for strat in (stratify, stratify_fallback, None):
            try:
                a, b = train_test_split(
                    df_in,
                    test_size=float(holdout_ratio),
                    random_state=seed,
                    shuffle=True,
                    stratify=strat,
                )
                return a.reset_index(drop=True), b.reset_index(drop=True)
            except ValueError:
                continue
        raise RuntimeError("split_train_holdout: unable to split (unexpected stratify failure)")

    def _split_holdout_to_val_test(
        df_in: pd.DataFrame,
        *,
        test_ratio_in_holdout: float,
        stratify: pd.Series | None,
        stratify_fallback: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df_in.empty:
            return df_in.iloc[0:0].copy(), df_in.iloc[0:0].copy()
        if test_ratio_in_holdout <= 0:
            return df_in.reset_index(drop=True), df_in.iloc[0:0].copy()
        if test_ratio_in_holdout >= 1.0:
            return df_in.iloc[0:0].copy(), df_in.reset_index(drop=True)
        for strat in (stratify, stratify_fallback, None):
            try:
                a, b = train_test_split(
                    df_in,
                    test_size=float(test_ratio_in_holdout),
                    random_state=seed,
                    shuffle=True,
                    stratify=strat,
                )
                return a.reset_index(drop=True), b.reset_index(drop=True)
            except ValueError:
                continue
        raise RuntimeError("split_holdout_to_val_test: unable to split (unexpected stratify failure)")

    if patient_split and manifest_csv is not None:
        df2 = _attach_patient_key(df, manifest_csv=Path(manifest_csv), patient_key_col=str(patient_key_col))

        # patient-level table
        patient_rows: list[dict] = []
        for pk, g in df2.groupby("patient_key"):
            patient_rows.append({"patient_key": str(pk), "primary_label": _patient_primary_label(g), "n_exams": int(len(g))})
        p_df = pd.DataFrame(patient_rows)
        if p_df.empty:
            raise typer.Exit(code=2)

        holdout = float(val_ratio_v + test_ratio_v)
        train_p, hold_p = _split_train_holdout(p_df, holdout_ratio=holdout, stratify=p_df["primary_label"], stratify_fallback=None)
        val_p, test_p = _split_holdout_to_val_test(
            hold_p,
            test_ratio_in_holdout=(float(test_ratio_v) / holdout) if holdout > 0 and test_ratio_v > 0 else 0.0,
            stratify=hold_p["primary_label"] if "primary_label" in hold_p.columns else None,
            stratify_fallback=None,
        )

        train_keys = set(train_p["patient_key"].astype(str).tolist())
        val_keys = set(val_p["patient_key"].astype(str).tolist())
        test_keys = set(test_p["patient_key"].astype(str).tolist())
        train_df = df2[df2["patient_key"].astype(str).isin(train_keys)].reset_index(drop=True)
        val_df = df2[df2["patient_key"].astype(str).isin(val_keys)].reset_index(drop=True)
        test_df = df2[df2["patient_key"].astype(str).isin(test_keys)].reset_index(drop=True)
        typer.echo(f"patient_split: patients train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}")
    else:
        holdout = float(val_ratio_v + test_ratio_v)
        key = _stratify_key(df)
        # First split: train vs holdout (val+test).
        train_df, hold_df = _split_train_holdout(df, holdout_ratio=holdout, stratify=key, stratify_fallback=_primary_label(df))

        # Second split: val vs test (default stratify by primary label).
        val_df, test_df = _split_holdout_to_val_test(
            hold_df,
            test_ratio_in_holdout=(float(test_ratio_v) / holdout) if holdout > 0 and test_ratio_v > 0 else 0.0,
            stratify=_primary_label(hold_df) if not hold_df.empty else None,
            stratify_fallback=None,
        )

    keep_set = _parse_int_set(keep_codes)
    drop_train_set = _parse_int_set(drop_train_codes)

    if keep_set:
        mode = str(keep_mode or "mask_side").strip().lower()
        if mode not in {"drop_exam", "mask_side"}:
            raise ValueError(f"unknown keep_mode: {keep_mode!r} (expected drop_exam|mask_side)")
        if mode == "drop_exam":
            all_codes = set(range(1, len(CLASS_ID_TO_NAME) + 1))
            drop_other = all_codes - keep_set
            train_df = _drop_exams_with_codes(train_df, drop_codes=drop_other)
            val_df = _drop_exams_with_codes(val_df, drop_codes=drop_other)
            test_df = _drop_exams_with_codes(test_df, drop_codes=drop_other)
        else:
            train_df = _apply_code_filters(train_df, keep_codes=keep_set, drop_codes=None)
            val_df = _apply_code_filters(val_df, keep_codes=keep_set, drop_codes=None)
            test_df = _apply_code_filters(test_df, keep_codes=keep_set, drop_codes=None)

    if drop_train_set:
        # Only drop from training to simulate an "unseen" class (check.md Setting B).
        mode = str(drop_train_mode or "drop_exam").strip().lower()
        if mode not in {"drop_exam", "mask_side"}:
            raise ValueError(f"unknown drop_train_mode: {drop_train_mode!r} (expected drop_exam|mask_side)")
        if mode == "drop_exam":
            train_df = _drop_exams_with_codes(train_df, drop_codes=drop_train_set)
        else:
            train_df = _apply_code_filters(train_df, keep_codes=None, drop_codes=drop_train_set)

    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_list = [float(x.strip()) for x in ratios.split(",") if x.strip()]
    for ratio in ratio_list:
        pct = int(round(ratio * 100))
        split_dir = out_dir / f"{pct}pct"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_sub = _select_balanced_exams(train_df, ratio=ratio, seed=seed, min_per_class=min_per_class)

        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"
        test_path = split_dir / "test.csv"
        stats_path = split_dir / "stats.json"

        train_sub.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        if test_ratio_v > 0:
            test_df.to_csv(test_path, index=False)
        else:
            # keep empty placeholder for downstream tooling consistency
            test_df.iloc[0:0].to_csv(test_path, index=False)
        _write_stats(stats_path, train_df=train_sub, val_df=val_df, test_df=test_df)

        typer.echo(f"{pct}%: train_exams={len(train_sub)} val_exams={len(val_df)} test_exams={len(test_df)} -> {split_dir}")


if __name__ == "__main__":
    app()
