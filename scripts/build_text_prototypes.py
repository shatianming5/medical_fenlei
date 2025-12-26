from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.feature_extraction.text import TfidfVectorizer

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.text_prompts import get_default_class_prompts_zh

app = typer.Typer(add_completion=False)


def _code_to_class_id(code) -> int | None:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    try:
        code_int = int(code)
    except Exception:
        return None
    # xlsx codes are 1..6; map to 0..5
    cid = code_int - 1
    if cid not in CLASS_ID_TO_NAME:
        return None
    return int(cid)


def _norm_text(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).replace("\r\n", "\n").replace("\r", "\n").strip()


@dataclass(frozen=True)
class _EarTextRow:
    exam_id: int
    class_id: int
    report_text: str


def _expand_ear_rows(df: pd.DataFrame) -> list[_EarTextRow]:
    if "exam_id" not in df.columns:
        raise ValueError("index_csv missing column: exam_id")
    if "report_text" not in df.columns:
        raise ValueError("index_csv missing column: report_text (re-run scripts/build_index.py)")
    if "left_code" not in df.columns or "right_code" not in df.columns:
        raise ValueError("index_csv missing columns: left_code/right_code")

    out: list[_EarTextRow] = []
    for r in df.itertuples(index=False):
        exam_id = int(getattr(r, "exam_id"))
        text = _norm_text(getattr(r, "report_text", ""))
        for col in ("left_code", "right_code"):
            cid = _code_to_class_id(getattr(r, col, None))
            if cid is None:
                continue
            if not text:
                continue
            out.append(_EarTextRow(exam_id=exam_id, class_id=int(cid), report_text=text))
    return out


def _l2norm(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


@app.command()
def main(
    index_csv: Path = typer.Option(Path("artifacts/dataset_index.csv"), exists=True, help="由 scripts/build_index.py 生成"),
    out_npz: Path = typer.Option(Path("artifacts/text_prototypes_tfidf.npz"), help="输出（不入库）"),
    max_features: int = typer.Option(20000, help="TF-IDF vocab 上限"),
    ngram_max: int = typer.Option(2, help="TF-IDF ngram 上限（1=unigram,2=bigram）"),
    final: str = typer.Option("mix", help="最终原型：prompt | data | mix（有数据用 data，没数据用 prompt）"),
    min_samples: int = typer.Option(20, help="data prototype 至少样本数（按耳样本计）"),
    normalize: bool = typer.Option(True, help="是否对原型做 L2 normalize（推荐 True）"),
    save_vectorizer: bool = typer.Option(False, help="保存 TF-IDF vectorizer（joblib；不入库）"),
    vectorizer_path: Path = typer.Option(Path("artifacts/text_tfidf_vectorizer.joblib"), help="vectorizer 保存路径"),
) -> None:
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    ear_rows = _expand_ear_rows(df)
    if not ear_rows:
        typer.echo("no usable (label, report_text) rows found; check report_text and labels")
        raise typer.Exit(code=2)

    prompts = get_default_class_prompts_zh()
    class_ids = np.asarray(sorted(CLASS_ID_TO_NAME.keys()), dtype=np.int64)
    class_names = np.asarray([str(CLASS_ID_TO_NAME[int(cid)]) for cid in class_ids], dtype=object)
    prompt_texts = np.asarray([str(prompts.get(int(cid), "")) for cid in class_ids], dtype=object)

    corpus = [r.report_text for r in ear_rows] + [str(t) for t in prompt_texts.tolist() if str(t)]
    vec = TfidfVectorizer(
        max_features=int(max_features) if int(max_features) > 0 else None,
        ngram_range=(1, int(max(1, ngram_max))),
        lowercase=False,
        analyzer="char",  # robust for Chinese without extra tokenizers
    )
    mat = vec.fit_transform(corpus)  # (N, V) sparse

    n_ear = len(ear_rows)
    ear_mat = mat[:n_ear]
    prompt_mat = mat[n_ear:]

    prompt_emb = prompt_mat.toarray().astype(np.float32)
    if prompt_emb.shape[0] != int(class_ids.size):
        raise RuntimeError("unexpected prompt embedding shape")

    # data prototypes (mean over ear-level samples with that class_id)
    data_emb = np.full((int(class_ids.size), int(prompt_emb.shape[1])), np.nan, dtype=np.float32)
    data_counts = np.zeros((int(class_ids.size),), dtype=np.int64)
    cid_to_idx = {int(cid): int(i) for i, cid in enumerate(class_ids.tolist())}
    ear_cids = np.asarray([int(r.class_id) for r in ear_rows], dtype=np.int64)
    ear_dense = ear_mat.toarray().astype(np.float32)
    for cid in np.unique(ear_cids):
        idx = cid_to_idx.get(int(cid))
        if idx is None:
            continue
        m = ear_cids == int(cid)
        k = int(m.sum())
        data_counts[idx] = k
        if k <= 0:
            continue
        data_emb[idx] = ear_dense[m].mean(axis=0)

    final_s = str(final).strip().lower()
    if final_s not in {"prompt", "data", "mix"}:
        raise ValueError(f"unknown final={final!r} (expected prompt|data|mix)")

    final_emb = prompt_emb.copy()
    if final_s == "data":
        # fall back to prompt when data missing
        for i in range(int(class_ids.size)):
            if int(data_counts[i]) >= int(min_samples) and np.isfinite(data_emb[i]).all():
                final_emb[i] = data_emb[i]
    elif final_s == "mix":
        for i in range(int(class_ids.size)):
            if int(data_counts[i]) >= int(min_samples) and np.isfinite(data_emb[i]).all():
                final_emb[i] = data_emb[i]

    if bool(normalize):
        prompt_emb = _l2norm(prompt_emb)
        final_emb = _l2norm(final_emb)
        # data_emb may contain NaNs; normalize only finite rows for saving convenience
        data_emb2 = data_emb.copy()
        ok = np.isfinite(data_emb2).all(axis=1)
        data_emb2[ok] = _l2norm(data_emb2[ok])
        data_emb = data_emb2

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        class_ids=class_ids,
        class_names=class_names,
        prompt_texts=prompt_texts,
        prompt_emb=prompt_emb,
        data_emb=data_emb,
        data_counts=data_counts,
        final=final_s,
        final_emb=final_emb,
    )
    typer.echo(f"saved: {out_npz}")
    typer.echo(f"dim: {final_emb.shape[1]}  method: tfidf(char)  n_ear_samples: {n_ear}")
    typer.echo(f"data_counts: {[int(x) for x in data_counts.tolist()]}")

    if bool(save_vectorizer):
        import joblib

        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vec, vectorizer_path)
        typer.echo(f"saved vectorizer: {vectorizer_path}")


if __name__ == "__main__":
    app()

