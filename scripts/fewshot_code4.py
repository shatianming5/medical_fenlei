from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from medical_fenlei.metrics import binary_metrics, bootstrap_binary_metrics_by_exam
from medical_fenlei.tasks import resolve_task

app = typer.Typer(add_completion=False)


def _l2norm(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


@dataclass(frozen=True)
class Emb:
    emb: np.ndarray  # (N,D)
    exam_id: np.ndarray  # (N,)
    side: np.ndarray  # (N,)
    label_code: np.ndarray  # (N,)


def _load_npz(path: Path) -> Emb:
    data = np.load(path, allow_pickle=True)
    emb = np.asarray(data["embedding"], dtype=np.float32)
    return Emb(
        emb=emb,
        exam_id=np.asarray(data["exam_id"], dtype=np.int64),
        side=np.asarray(data["side"], dtype=object),
        label_code=np.asarray(data["label_code"], dtype=np.int64),
    )


def _filter_task(e: Emb, *, task_name: str) -> tuple[Emb, np.ndarray, np.ndarray]:
    task = resolve_task(task_name)
    if task.kind != "binary":
        raise ValueError(f"fewshot_code4 expects a binary task; got {task.kind}")
    rel = task.relevant_codes()
    pos = set(task.pos_codes)

    m = np.isin(e.label_code, np.asarray(sorted(rel), dtype=np.int64))
    emb = e.emb[m]
    exam_id = e.exam_id[m]
    side = e.side[m]
    label_code = e.label_code[m]
    y = np.asarray([1 if int(c) in pos else 0 for c in label_code], dtype=np.int64)
    return Emb(emb=emb, exam_id=exam_id, side=side, label_code=label_code), y, exam_id


def _prototype_scores(train: Emb, y_train: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    x = _l2norm(train.emb.astype(np.float32))
    q = _l2norm(query_emb.astype(np.float32))

    pos = x[y_train == 1]
    neg = x[y_train == 0]
    if pos.shape[0] <= 0 or neg.shape[0] <= 0:
        raise ValueError("need both pos and neg samples for prototype")

    proto_pos = pos.mean(axis=0, keepdims=True)
    proto_neg = neg.mean(axis=0, keepdims=True)
    proto_pos = _l2norm(proto_pos)[0]
    proto_neg = _l2norm(proto_neg)[0]

    s_pos = (q * proto_pos[None, :]).sum(axis=1)
    s_neg = (q * proto_neg[None, :]).sum(axis=1)
    return (s_pos - s_neg).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _knn_neighbors(train: Emb, query: Emb, *, topk: int) -> list[list[dict[str, Any]]]:
    x = _l2norm(train.emb.astype(np.float32))
    q = _l2norm(query.emb.astype(np.float32))
    sims = q @ x.T  # (Nq, Nt)
    out: list[list[dict[str, Any]]] = []
    for i in range(sims.shape[0]):
        idx = np.argsort(-sims[i])[: int(topk)].astype(int).tolist()
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
    train_npz: Path = typer.Option(..., exists=True, help="由 scripts/extract_embeddings_ear2d.py 生成"),
    val_npz: Path = typer.Option(..., exists=True),
    task: str = typer.Option("normal_vs_cholesterol_granuloma", help="必须是 pos_codes 含 code4 的二分类 task"),
    temperature: float = typer.Option(1.0, help="score -> prob 的温度缩放（越大越平）"),
    n_boot: int = typer.Option(1000, help="bootstrap 次数（按 exam_id）"),
    seed: int = typer.Option(42),
    knn_topk: int = typer.Option(5),
    out_dir: Path = typer.Option(Path("artifacts/fewshot_code4"), help="输出目录（不入库）"),
) -> None:
    train = _load_npz(train_npz)
    val = _load_npz(val_npz)

    train_f, y_tr, exam_tr = _filter_task(train, task_name=str(task))
    val_f, y_va, exam_va = _filter_task(val, task_name=str(task))
    if train_f.emb.size <= 0 or val_f.emb.size <= 0:
        raise typer.Exit(code=2)

    # prototype score
    score = _prototype_scores(train_f, y_tr, val_f.emb) / float(max(1e-6, float(temperature)))
    prob = _sigmoid(score)

    m = binary_metrics(y_va, prob, threshold=0.5, specificity_target=0.95)
    ci = bootstrap_binary_metrics_by_exam(y_va, prob, exam_va, n_boot=int(n_boot), seed=int(seed))

    # kNN neighbors for case-based explanation
    neigh = _knn_neighbors(train_f, val_f, topk=int(knn_topk))

    rows: list[dict[str, Any]] = []
    for i in range(val_f.emb.shape[0]):
        rows.append(
            {
                "exam_id": int(val_f.exam_id[i]),
                "side": str(val_f.side[i]),
                "label_code": int(val_f.label_code[i]),
                "y_true": int(y_va[i]),
                "score": float(score[i]),
                "y_prob": float(prob[i]),
                "y_pred": int(prob[i] >= 0.5),
                "neighbors": json.dumps(neigh[i], ensure_ascii=False),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / f"pred_{task}.csv"
    pd.DataFrame(rows).to_csv(pred_csv, index=False)

    report = {
        "method": "prototype_cosine",
        "task": str(task),
        "train_npz": str(train_npz),
        "val_npz": str(val_npz),
        "n_train": int(train_f.emb.shape[0]),
        "n_val": int(val_f.emb.shape[0]),
        "train_pos": int((y_tr == 1).sum()),
        "train_neg": int((y_tr == 0).sum()),
        "val_pos": int((y_va == 1).sum()),
        "val_neg": int((y_va == 0).sum()),
        "metrics": m,
        "bootstrap_ci_by_exam": ci,
        "temperature": float(temperature),
        "knn_topk": int(knn_topk),
    }
    rep_path = out_dir / f"report_{task}.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"saved: {pred_csv}")
    typer.echo(f"saved: {rep_path}")


if __name__ == "__main__":
    app()

