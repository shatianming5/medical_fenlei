from __future__ import annotations

import re
from pathlib import Path

_DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$")


def _is_date_dir(name: str) -> bool:
    return bool(_DATE_RE.match(name))


def infer_dicom_root(base: Path, *, max_descend: int = 5) -> Path:
    """
    Infer the directory that contains date folders (YYYY-MM-DD).

    The extracted dataset sometimes has repeated top-level folders, so this
    function will descend into a single-child directory chain until it finds
    date folders.
    """
    current = base
    for _ in range(max_descend):
        try:
            children = [p for p in current.iterdir() if p.is_dir()]
        except FileNotFoundError:
            return base

        if any(_is_date_dir(p.name) for p in children):
            return current

        # Common case: nested single folder(s)
        non_hidden = [p for p in children if not p.name.startswith(".")]
        if len(non_hidden) == 1:
            current = non_hidden[0]
            continue

        # Fallback: try shallow search
        for cand in non_hidden:
            try:
                cand_children = [p for p in cand.iterdir() if p.is_dir()]
            except Exception:
                continue
            if any(_is_date_dir(p.name) for p in cand_children):
                return cand

        return current

    return current

