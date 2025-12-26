from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from medical_fenlei.text_triplets import extract_entity_attribute_triplets, triplets_to_strings

app = typer.Typer(add_completion=False)


def _norm_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).replace("\r\n", "\n").replace("\r", "\n").strip()


@app.command()
def main(
    index_csv: Path = typer.Option(Path("artifacts/dataset_index.csv"), exists=True, help="由 scripts/build_index.py 生成"),
    out_jsonl: Path = typer.Option(Path("artifacts/report_triplets.jsonl"), help="输出 JSONL（不入库）"),
    out_csv: Path | None = typer.Option(Path("artifacts/report_triplets_flat.csv"), help="可选：输出扁平 CSV（不入库）"),
    limit: int | None = typer.Option(None, help="调试：最多处理 N 条（None=全部）"),
    include_text: bool = typer.Option(False, help="JSONL 中是否包含原始 report_text（默认不包含，避免泄露）"),
    verbose: bool = typer.Option(True),
) -> None:
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    need_cols = {"exam_id", "report_text"}
    missing = [c for c in sorted(need_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"index_csv missing columns: {missing} (re-run scripts/build_index.py)")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_flat: list[dict[str, Any]] = []
    n_total = 0
    n_with_text = 0
    n_with_triplets = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            if limit is not None and n_total >= int(limit):
                break
            n_total += 1

            exam_id = int(getattr(r, "exam_id"))
            report_text = _norm_text(getattr(r, "report_text", ""))
            if report_text:
                n_with_text += 1

            triplets = extract_entity_attribute_triplets(report_text) if report_text else []
            if triplets:
                n_with_triplets += 1

            rec: dict[str, Any] = {
                "exam_id": exam_id,
                "n_triplets": int(len(triplets)),
                "triplets": [t.as_dict() for t in triplets],
            }
            if include_text:
                rec["report_text"] = report_text
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if out_csv is not None and triplets:
                for t in triplets:
                    rows_flat.append(
                        {
                            "exam_id": exam_id,
                            "location": t.location,
                            "entity": t.entity,
                            "attribute": t.attribute,
                            "negated": bool(t.negated),
                            "clause": t.clause,
                        }
                    )

            if verbose and n_total <= 3 and triplets:
                typer.echo(f"example exam_id={exam_id}: {triplets_to_strings(triplets)[:10]}")

    if out_csv is not None:
        pd.DataFrame(rows_flat).to_csv(out_csv, index=False)

    typer.echo(f"rows: {n_total}  with_text: {n_with_text}  with_triplets: {n_with_triplets}")
    typer.echo(f"saved: {out_jsonl}")
    if out_csv is not None:
        typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()

