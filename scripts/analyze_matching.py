from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

import pandas as pd
import typer

from medical_fenlei.labels import load_labels_xlsx
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _scan_disk_exams(dicom_root: Path) -> dict[int, list[str]]:
    exam_map: dict[int, list[str]] = defaultdict(list)
    for d in dicom_root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        if not _DATE_RE.match(d.name):
            continue
        for e in d.iterdir():
            if not e.is_dir() or e.name.startswith("."):
                continue
            try:
                exam_id = int(e.name)
            except Exception:
                continue
            exam_map[exam_id].append(d.name)
    return dict(exam_map)


@app.command()
def main(
    dicom_base: Path = typer.Option(Path("data/medical_data_2"), help="DICOM 数据基目录（会自动向下推断真正的 dicom_root）"),
    xlsx_path: Path = typer.Option(
        Path("metadata/导出数据第1~4017条数据20240329-To模型训练团队.xlsx"),
        help="标注表 XLSX（本地文件，不入库）",
    ),
    limit: int | None = typer.Option(None, help="仅处理前 N 条（用于快速验证）"),
    show_examples: int = typer.Option(5, help="展示 date 不匹配示例数"),
) -> None:
    dicom_root = infer_dicom_root(dicom_base)
    labels = load_labels_xlsx(xlsx_path)
    if limit is not None:
        labels = labels.head(int(limit))

    exam_map = _scan_disk_exams(dicom_root)
    disk_exam_dirs = sum(len(v) for v in exam_map.values())
    disk_unique_exam = len(exam_map)

    typer.echo(f"dicom_root: {dicom_root}")
    typer.echo(f"labels: {len(labels)} rows  unique_exam_id={labels.exam_id.nunique()}")
    typer.echo(f"disk: exam_dirs={disk_exam_dirs}  unique_exam_id={disk_unique_exam}")

    multi = {k: v for k, v in exam_map.items() if len(v) > 1}
    typer.echo(f"disk duplicate exam_id (multiple dates): {len(multi)}")

    strict = 0
    id_only = 0
    id_only_date_mismatch = 0
    missing = 0
    mismatches: list[tuple[int, str, list[str]]] = []

    for r in labels.itertuples(index=False):
        dates = exam_map.get(int(r.exam_id))
        if not dates:
            missing += 1
            continue
        id_only += 1
        if r.date in dates:
            strict += 1
        else:
            id_only_date_mismatch += 1
            if len(mismatches) < show_examples:
                mismatches.append((int(r.exam_id), str(r.date), sorted(dates)))

    typer.echo(f"match by exam_id only: {id_only}")
    typer.echo(f"strict match by exam_id+date: {strict}")
    typer.echo(f"id matches but date mismatch: {id_only_date_mismatch}")
    typer.echo(f"labels exam_id missing on disk: {missing}")

    labels_on_disk = labels[labels["exam_id"].isin(exam_map.keys())].copy()
    typer.echo(
        "label sides on disk: "
        f"any_side={(labels_on_disk['left_code'].notna() | labels_on_disk['right_code'].notna()).sum()}  "
        f"left={(labels_on_disk['left_code'].notna()).sum()}  "
        f"right={(labels_on_disk['right_code'].notna()).sum()}  "
        f"sides_total={(labels_on_disk['left_code'].notna()).sum() + (labels_on_disk['right_code'].notna()).sum()}"
    )

    if mismatches:
        typer.echo("examples (exam_id, label_date -> disk_dates):")
        for exam_id, label_date, disk_dates in mismatches:
            typer.echo(f"  - {exam_id}: {label_date} -> {disk_dates[:5]}{'...' if len(disk_dates) > 5 else ''}")

    # quick year-level mismatch summary
    years = Counter()
    for r in labels.itertuples(index=False):
        dates = exam_map.get(int(r.exam_id))
        if not dates:
            continue
        if r.date in dates:
            continue
        years[(str(r.date)[:4], str(dates[0])[:4])] += 1
    if years:
        typer.echo(f"top year mismatches: {years.most_common(10)}")


if __name__ == "__main__":
    app()

