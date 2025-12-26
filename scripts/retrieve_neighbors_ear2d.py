from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _l2norm(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def _parse_int_list(s: str | None) -> list[int]:
    if s is None:
        return []
    out: list[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


@dataclass(frozen=True)
class Emb:
    emb: np.ndarray  # (N,D)
    exam_id: np.ndarray  # (N,)
    side: np.ndarray  # (N,)
    label_code: np.ndarray  # (N,)


def _load_npz(path: Path) -> Emb:
    data = np.load(path, allow_pickle=True)
    emb = np.asarray(data["embedding"], dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"expected embedding (N,D), got {emb.shape} in {path}")
    return Emb(
        emb=emb,
        exam_id=np.asarray(data["exam_id"], dtype=np.int64),
        side=np.asarray(data["side"], dtype=object),
        label_code=np.asarray(data["label_code"], dtype=np.int64),
    )


def _filter_label_codes(e: Emb, *, label_codes: list[int]) -> Emb:
    if not label_codes:
        return e
    codes = np.asarray(sorted(set(int(x) for x in label_codes)), dtype=np.int64)
    m = np.isin(e.label_code.astype(np.int64), codes)
    return Emb(emb=e.emb[m], exam_id=e.exam_id[m], side=e.side[m], label_code=e.label_code[m])


def _topk_cosine_neighbors(
    *,
    train: Emb,
    query: Emb,
    topk: int,
    block_size: int = 256,
) -> list[list[dict[str, Any]]]:
    topk = int(topk)
    if topk <= 0:
        return [[] for _ in range(int(query.emb.shape[0]))]
    if train.emb.size <= 0 or query.emb.size <= 0:
        return [[] for _ in range(int(query.emb.shape[0]))]

    x = _l2norm(train.emb.astype(np.float32))
    q = _l2norm(query.emb.astype(np.float32))

    n_train = int(x.shape[0])
    k = min(int(topk), int(n_train))

    out: list[list[dict[str, Any]]] = []
    for start in range(0, int(q.shape[0]), int(block_size)):
        qb = q[start : start + int(block_size)]
        sims = qb @ x.T  # (B, Nt)
        # partial top-k then sort
        idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for i in range(int(idx_part.shape[0])):
            idx = idx_part[i]
            idx = idx[np.argsort(-sims[i, idx])].astype(int).tolist()
            out.append(
                [
                    {
                        "exam_id": int(train.exam_id[j]),
                        "side": str(train.side[j]),
                        "label_code": int(train.label_code[j]),
                        "sim": float(sims[i, j]),
                    }
                    for j in idx
                ]
            )

    return out


@app.command()
def main(
    train_npz: Path = typer.Option(..., exists=True, help="由 scripts/extract_embeddings_ear2d.py 生成（train）"),
    query_npz: Path = typer.Option(..., exists=True, help="由 scripts/extract_embeddings_ear2d.py 生成（val 或自定义 query）"),
    train_label_codes: str | None = typer.Option(None, help="仅在训练库中检索这些 label_code，例如 '2' 或 '2,5'"),
    query_label_codes: str | None = typer.Option(None, help="仅对这些 query label_code 做检索（可选）"),
    topk: int = typer.Option(5, help="每个 query 返回的 neighbors 数"),
    block_size: int = typer.Option(256, help="分块计算的 block size（避免一次性占用太多内存）"),
    out_csv: Path = typer.Option(Path("artifacts/retrieval_neighbors.csv"), help="输出 CSV（不入库）"),
) -> None:
    train = _load_npz(train_npz)
    query = _load_npz(query_npz)

    train_codes = _parse_int_list(train_label_codes)
    query_codes = _parse_int_list(query_label_codes)
    train_f = _filter_label_codes(train, label_codes=train_codes)
    query_f = _filter_label_codes(query, label_codes=query_codes)

    neigh = _topk_cosine_neighbors(train=train_f, query=query_f, topk=int(topk), block_size=int(block_size))

    rows: list[dict[str, Any]] = []
    for i in range(int(query_f.emb.shape[0])):
        rows.append(
            {
                "query_exam_id": int(query_f.exam_id[i]),
                "query_side": str(query_f.side[i]),
                "query_label_code": int(query_f.label_code[i]),
                "neighbors": json.dumps(neigh[i], ensure_ascii=False),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    typer.echo(f"train: {int(train_f.emb.shape[0])}  query: {int(query_f.emb.shape[0])}")
    typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()

