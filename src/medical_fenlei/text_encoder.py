from __future__ import annotations

import math
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

TextPool = Literal["cls", "mean"]


def random_projection(x: torch.Tensor, *, out_dim: int, seed: int = 42) -> torch.Tensor:
    """
    Deterministic random projection to match embedding dims.

    Used to map text encoder hidden_size -> visual feat_dim while keeping the
    whole pipeline dependency-light (no extra projection head to train).
    """
    if x.ndim != 2:
        raise ValueError(f"expected x (N,D), got {tuple(x.shape)}")
    out_dim = int(out_dim)
    if out_dim <= 0:
        raise ValueError(f"out_dim must be > 0, got {out_dim}")
    in_dim = int(x.shape[1])
    if in_dim == out_dim:
        return x

    # Use a CPU generator for deterministic behavior across devices.
    g = torch.Generator()
    g.manual_seed(int(seed))
    w = torch.randn(in_dim, out_dim, generator=g, dtype=torch.float32)
    w = w / math.sqrt(float(in_dim))
    y = x.float() @ w.to(device=x.device)
    return y


@torch.no_grad()
def hf_encode_texts(
    texts: list[str],
    *,
    model_name_or_path: str,
    pool: TextPool = "cls",
    max_length: int = 256,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Encode Chinese radiology texts using a HuggingFace Transformer (e.g., CMBERT).

    Returns: (N, hidden_size) float32 tensor on CPU.
    """
    if not texts:
        return torch.zeros((0, 0), dtype=torch.float32)

    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("hf_encode_texts requires `transformers` (pip install transformers)") from e

    pool_s = str(pool).strip().lower()
    if pool_s not in {"cls", "mean"}:
        raise ValueError(f"unknown pool: {pool!r} (expected cls|mean)")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
    model = AutoModel.from_pretrained(str(model_name_or_path))
    model.eval()
    model.to(device)

    out: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    max_len = max(8, int(max_length))

    for i in range(0, len(texts), bs):
        chunk = [str(t) for t in texts[i : i + bs]]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        y = model(**enc)
        last = getattr(y, "last_hidden_state", None)
        if last is None:
            # Some models return tuples; try index 0 as a fallback.
            if isinstance(y, (tuple, list)) and y:
                last = y[0]
        if last is None or not torch.is_tensor(last):
            raise RuntimeError("unexpected HF model output; cannot find last_hidden_state")

        if pool_s == "cls":
            emb = last[:, 0]
        else:
            attn = enc.get("attention_mask")
            if attn is None:
                emb = last.mean(dim=1)
            else:
                m = attn.unsqueeze(-1).to(dtype=last.dtype)
                denom = m.sum(dim=1).clamp(min=1.0)
                emb = (last * m).sum(dim=1) / denom
        out.append(emb.detach().cpu().float())

    emb_all = torch.cat(out, dim=0)
    return emb_all


def encode_texts_to_dim(
    texts: list[str],
    *,
    dim: int,
    encoder: str = "hash",
    hf_model_name_or_path: str = "hfl/chinese-roberta-wwm-ext",
    hf_pool: TextPool = "cls",
    hf_max_length: int = 256,
    proj_seed: int = 42,
) -> torch.Tensor:
    """
    Encode texts to a fixed dim for use as prototypes or for retrieval eval.

    - encoder=hash: fast HashingVectorizer baseline (char ngrams, deterministic)
    - encoder=hf: HuggingFace model (e.g., CMBERT) + deterministic random projection
    """
    enc = str(encoder).strip().lower()
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")

    if enc == "hash":
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
        except Exception as e:  # pragma: no cover
            raise RuntimeError("encode_texts_to_dim(encoder=hash) requires scikit-learn") from e

        vec = HashingVectorizer(
            n_features=int(dim),
            analyzer="char",
            ngram_range=(1, 2),
            lowercase=False,
            alternate_sign=False,
            norm=None,
        )
        mat = vec.transform([str(t) for t in texts]).toarray().astype("float32")
        t = torch.from_numpy(mat)
        return F.normalize(t, dim=-1)

    if enc == "hf":
        emb = hf_encode_texts(
            [str(t) for t in texts],
            model_name_or_path=str(hf_model_name_or_path),
            pool=str(hf_pool),  # type: ignore[arg-type]
            max_length=int(hf_max_length),
            batch_size=16,
            device=torch.device("cpu"),
        )
        emb = random_projection(emb, out_dim=int(dim), seed=int(proj_seed))
        return F.normalize(emb, dim=-1)

    raise ValueError(f"unknown encoder: {encoder!r} (expected hash|hf)")


class HFTextVectorizer:
    """
    HuggingFace text encoder with `.transform(texts) -> np.ndarray` API.

    This mirrors the minimal interface used by the training loop (sklearn-like),
    but avoids re-loading the HF model every batch. It also supports a simple
    in-memory cache for repeated texts across epochs.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        out_dim: int,
        pool: TextPool = "cls",
        max_length: int = 256,
        batch_size: int = 16,
        device: str | torch.device = "cpu",
        proj_seed: int = 42,
        cache_size: int = 10000,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("HFTextVectorizer requires `transformers` (pip install transformers)") from e

        self.model_name_or_path = str(model_name_or_path)
        self.out_dim = int(out_dim)
        if self.out_dim <= 0:
            raise ValueError(f"out_dim must be > 0, got {self.out_dim}")
        self.pool = str(pool).strip().lower()  # type: ignore[assignment]
        if self.pool not in {"cls", "mean"}:
            raise ValueError(f"unknown pool: {pool!r} (expected cls|mean)")
        self.max_length = max(8, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.proj_seed = int(proj_seed)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

        self._proj_weight: torch.Tensor | None = None  # (hidden,out_dim) on self.device

    def _normalize_key(self, t: str) -> str:
        return str(t).replace("\r\n", "\n").replace("\r", "\n").strip()

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected x (N,D), got {tuple(x.shape)}")
        in_dim = int(x.shape[1])
        if in_dim == int(self.out_dim):
            return x
        if self._proj_weight is None or tuple(self._proj_weight.shape) != (in_dim, int(self.out_dim)):
            g = torch.Generator()
            g.manual_seed(int(self.proj_seed))
            w = torch.randn(in_dim, int(self.out_dim), generator=g, dtype=torch.float32)
            w = w / math.sqrt(float(in_dim))
            self._proj_weight = w.to(device=self.device)
        return x.float() @ self._proj_weight

    @torch.no_grad()
    def _encode_unique(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, int(self.out_dim)), dtype=np.float32)

        out: list[np.ndarray] = []
        for i in range(0, len(texts), int(self.batch_size)):
            chunk = [self._normalize_key(t) for t in texts[i : i + int(self.batch_size)]]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=int(self.max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            y = self.model(**enc)
            last = getattr(y, "last_hidden_state", None)
            if last is None:
                if isinstance(y, (tuple, list)) and y:
                    last = y[0]
            if last is None or not torch.is_tensor(last):
                raise RuntimeError("unexpected HF model output; cannot find last_hidden_state")

            if self.pool == "cls":
                emb = last[:, 0]
            else:
                attn = enc.get("attention_mask")
                if attn is None:
                    emb = last.mean(dim=1)
                else:
                    m = attn.unsqueeze(-1).to(dtype=last.dtype)
                    denom = m.sum(dim=1).clamp(min=1.0)
                    emb = (last * m).sum(dim=1) / denom

            emb = self._project(emb)
            emb = F.normalize(emb, dim=-1)
            out.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(out, axis=0).astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Return (N,out_dim) float32 array.
        """
        if texts is None:
            texts = []
        n = int(len(texts))
        out = np.zeros((n, int(self.out_dim)), dtype=np.float32)
        if n <= 0:
            return out

        # Gather missing unique texts.
        missing: list[str] = []
        missing_pos: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            key = self._normalize_key(str(t))
            if not key:
                continue
            cached = self._cache.get(key) if self.cache_size != 0 else None
            if cached is not None:
                out[int(i)] = cached
                self._cache.move_to_end(key, last=True)
            else:
                if key not in missing_pos:
                    missing.append(key)
                    missing_pos[key] = []
                missing_pos[key].append(int(i))

        if missing:
            emb = self._encode_unique(missing)  # (M,out_dim)
            if emb.shape[0] != len(missing) or emb.shape[1] != int(self.out_dim):
                raise RuntimeError(f"unexpected HF embedding shape: {emb.shape}, expected ({len(missing)},{self.out_dim})")
            for j, key in enumerate(missing):
                vec = emb[int(j)]
                for i in missing_pos.get(key, []):
                    out[int(i)] = vec

                if int(self.cache_size) != 0:
                    self._cache[key] = vec
                    self._cache.move_to_end(key, last=True)
                    if int(self.cache_size) > 0 and len(self._cache) > int(self.cache_size):
                        self._cache.popitem(last=False)

        return out
