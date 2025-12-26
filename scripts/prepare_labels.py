from __future__ import annotations

from pathlib import Path

import typer

from medical_fenlei.cli_defaults import default_labels_xlsx
from medical_fenlei.labels import load_labels_xlsx

app = typer.Typer(add_completion=False)


@app.command()
def main(
    xlsx_path: Path = typer.Option(
        default_labels_xlsx(),
        exists=True,
        help="标注表 XLSX（本地文件，不入库）",
    ),
    out_csv: Path = typer.Option(
        Path("artifacts/labels_clean.csv"),
        help="输出的去标识化 CSV（不入库）",
    ),
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = load_labels_xlsx(xlsx_path)
    df.to_csv(out_csv, index=False)

    typer.echo(f"labels: {len(df)} rows")
    typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()
