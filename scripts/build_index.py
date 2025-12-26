from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from medical_fenlei.cli_defaults import default_dicom_base, default_labels_xlsx
from medical_fenlei.indexing import build_dataset_index
from medical_fenlei.labels import load_labels_xlsx
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dicom_base: Path = typer.Option(
        default_dicom_base(),
        help="DICOM 数据基目录（会自动向下推断真正的 dicom_root）",
    ),
    xlsx_path: Path = typer.Option(
        default_labels_xlsx(),
        help="标注表 XLSX（本地文件，不入库）",
    ),
    limit: int | None = typer.Option(None, help="仅处理前 N 条（用于快速验证）"),
    out_csv: Path = typer.Option(
        Path("artifacts/dataset_index.csv"),
        help="输出索引 CSV（不入库）",
    ),
) -> None:
    dicom_root = infer_dicom_root(dicom_base)
    labels = load_labels_xlsx(xlsx_path)
    if limit is not None:
        labels = labels.head(int(limit))
    index = build_dataset_index(labels, dicom_root=dicom_root)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_csv, index=False)

    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"labels: {len(labels)} rows")
    typer.echo(f"labels unique exam_id: {labels['exam_id'].nunique()}")
    typer.echo(f"matched: {len(index)} rows")
    if not index.empty and "date_match" in index.columns:
        n_date_match = int(index["date_match"].fillna(False).sum())
        n_mismatch = int(len(index) - n_date_match)
        typer.echo(f"date_match: {n_date_match}  date_mismatch: {n_mismatch}")
    if not index.empty and "ambiguous_match" in index.columns:
        typer.echo(f"ambiguous_match: {int(index['ambiguous_match'].fillna(False).sum())}")
    if not index.empty:
        label_exam_ids = set(labels["exam_id"].astype("int64").tolist())
        matched_exam_ids = set(index["exam_id"].astype("int64").tolist())
        typer.echo(f"labels missing in index: {len(label_exam_ids - matched_exam_ids)}")
    typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()
