from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pandas as pd

_DIAG_RE: Final = re.compile(r"^\s*(?P<code>\d+)\s*[：:]\s*(?P<name>.+?)\s*$")


def _parse_diag(value) -> tuple[int | None, str | None]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    m = _DIAG_RE.match(text)
    if not m:
        return None, text
    return int(m.group("code")), m.group("name").strip()


def load_labels_xlsx(
    xlsx_path: str | Path,
    *,
    sheet_name: str = "影像学检查",
    exam_col: str = "检查号",
    time_col: str = "检查时间",
    left_col: str = "左侧诊断-训练",
    right_col: str = "右侧诊断-训练",
    findings_col: str = "影像所见",
    conclusion_col: str = "检查结论",
    include_report_text: bool = True,
) -> pd.DataFrame:
    """
    Load and de-identify labels from the XLSX.

    Returns a dataframe with only:
      - exam_id (int)
      - date (YYYY-MM-DD)
      - left_code / left_name
      - right_code / right_name
      - (optional) findings / conclusion / report_text
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    exam = pd.to_numeric(df[exam_col], errors="coerce")
    dt = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    date = dt.dt.strftime("%Y-%m-%d")

    left_parsed = df[left_col].apply(_parse_diag)
    right_parsed = df[right_col].apply(_parse_diag)

    data = {
        "exam_id": exam,
        "date": date,
        "left_code": [c for c, _ in left_parsed],
        "left_name": [n for _, n in left_parsed],
        "right_code": [c for c, _ in right_parsed],
        "right_name": [n for _, n in right_parsed],
    }

    if bool(include_report_text):
        def _norm_text(v) -> str:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            s = str(v).replace("\r\n", "\n").replace("\r", "\n").strip()
            return s

        findings = df[findings_col].apply(_norm_text) if findings_col in df.columns else pd.Series([""] * len(df))
        conclusion = df[conclusion_col].apply(_norm_text) if conclusion_col in df.columns else pd.Series([""] * len(df))
        report_text = (findings + "\n" + conclusion).str.strip()
        data.update({"findings": findings, "conclusion": conclusion, "report_text": report_text})

    out = pd.DataFrame(data)

    out = out.dropna(subset=["exam_id", "date"])
    out["exam_id"] = out["exam_id"].astype("int64")
    out = out.drop_duplicates(subset=["exam_id", "date"]).reset_index(drop=True)
    return out
