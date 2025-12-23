from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _load_exam_ids(path: Path) -> set[int]:
    df = pd.read_csv(path)
    if "exam_id" not in df.columns:
        raise ValueError(f"missing exam_id in {path}")
    return set(df["exam_id"].astype(int).tolist())


@app.command()
def main(
    manifest_csv: Path = typer.Option(Path("artifacts/manifest_ears.csv"), exists=True, help="由 scripts/build_manifest_ears.py 生成（不入库）"),
    train_csv: Path = typer.Option(Path("artifacts/splits_dual/100pct/train.csv"), exists=True),
    val_csv: Path = typer.Option(Path("artifacts/splits_dual/100pct/val.csv"), exists=True),
    patient_key_col: str = typer.Option("patient_key_hash", help="manifest 里的病人标识列（建议 hash）"),
    out_csv: Path = typer.Option(Path("artifacts/leakage_patient_ids.csv"), help="泄漏明细输出（不入库）"),
) -> None:
    man = pd.read_csv(manifest_csv)
    if man.empty:
        raise typer.Exit(code=2)
    if patient_key_col not in man.columns:
        raise ValueError(f"missing column in manifest: {patient_key_col}")

    train_exam_ids = _load_exam_ids(train_csv)
    val_exam_ids = _load_exam_ids(val_csv)

    cols = ["exam_id", "side", patient_key_col]
    df = man.loc[man["has_label"].fillna(False), cols].copy()
    df[patient_key_col] = df[patient_key_col].astype(str).replace({"None": "", "nan": ""})
    df = df[df[patient_key_col].astype(str).str.len() > 0]
    if df.empty:
        typer.echo("no patient key available; cannot check patient-level leakage")
        raise typer.Exit(code=0)

    train_patients = set(df.loc[df["exam_id"].isin(train_exam_ids), patient_key_col].tolist())
    val_patients = set(df.loc[df["exam_id"].isin(val_exam_ids), patient_key_col].tolist())
    leaked = sorted(train_patients & val_patients)

    typer.echo(f"train exams: {len(train_exam_ids)}  val exams: {len(val_exam_ids)}")
    typer.echo(f"patients(train): {len(train_patients)}  patients(val): {len(val_patients)}")
    typer.echo(f"leaked patients: {len(leaked)}")

    if not leaked:
        raise typer.Exit(code=0)

    leak_df = df[df[patient_key_col].isin(leaked)].copy()
    leak_df["split"] = leak_df["exam_id"].apply(lambda x: "train" if int(x) in train_exam_ids else ("val" if int(x) in val_exam_ids else "other"))
    leak_df = leak_df.sort_values([patient_key_col, "split", "exam_id", "side"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    leak_df.to_csv(out_csv, index=False)
    typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()

