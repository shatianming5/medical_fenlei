from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


@app.command()
def main(
    dicom_base: Path = typer.Option(default_dicom_base(), help="DICOM 数据基目录"),
) -> None:
    dicom_root = infer_dicom_root(dicom_base)
    date_dirs = _run(["bash", "-lc", f"find '{dicom_root}' -maxdepth 1 -mindepth 1 -type d | wc -l"])
    exam_dirs = _run(["bash", "-lc", f"find '{dicom_root}' -maxdepth 2 -mindepth 2 -type d | wc -l"])
    series_dirs = _run(["bash", "-lc", f"find '{dicom_root}' -maxdepth 3 -mindepth 3 -type d | wc -l"])
    dcm_files = _run(["bash", "-lc", f"find '{dicom_root}' -type f -iname '*.dcm' | wc -l"])

    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"date_dirs: {date_dirs}")
    typer.echo(f"exam_dirs: {exam_dirs}")
    typer.echo(f"series_dirs: {series_dirs}")
    typer.echo(f"dcm_files: {dcm_files}")


if __name__ == "__main__":
    app()
