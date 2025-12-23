from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from medical_fenlei.indexing import build_dataset_index
from medical_fenlei.labels import load_labels_xlsx
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dicom_base: Path = typer.Option(
        Path("data/medical_data_2"),
        help="DICOM 数据基目录（会自动向下推断真正的 dicom_root）",
    ),
    xlsx_path: Path = typer.Option(
        Path("metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx"),
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
    typer.echo(f"matched: {len(index)} rows")
    typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()
