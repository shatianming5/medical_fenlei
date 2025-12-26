from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from medical_fenlei.cli_defaults import default_dicom_base
from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.dual_dataset import DualPreprocessSpec, EarCTDualDataset
from medical_fenlei.models.dual_factory import make_dual_model
from medical_fenlei.paths import infer_dicom_root
from medical_fenlei.tasks import resolve_task

app = typer.Typer(add_completion=False)


def _default_out_dir(checkpoint: Path) -> Path:
    p = checkpoint.resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent
    return p.parent


def _strip_compile_prefix(state_dict: dict) -> dict:
    # torch.compile state_dict keys often start with "_orig_mod."
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if any(k.startswith("_orig_mod.") for k in keys):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


@torch.no_grad()
def _extract_embeddings(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_items: int | None,
) -> tuple[np.ndarray, list[str], list[int]]:
    model.eval()
    img_list: list[np.ndarray] = []
    text_list: list[str] = []
    exam_id_list: list[int] = []

    amp_enabled = bool(amp) and device.type == "cuda"
    autocast_ctx = (
        (torch.amp.autocast(device_type="cuda") if hasattr(torch, "amp") else torch.cuda.amp.autocast())
        if amp_enabled
        else None
    )

    fwf = getattr(model, "forward_with_features", None)
    if fwf is None and hasattr(model, "_orig_mod"):
        fwf = getattr(model._orig_mod, "forward_with_features", None)
    if not callable(fwf):
        raise ValueError("model does not expose forward_with_features(); retrieval eval expects *_proto models")

    n_seen = 0
    for batch in loader:
        if max_items is not None and n_seen >= int(max_items):
            break

        x = batch["image"].to(device, non_blocking=True)
        texts = batch.get("report_text")
        meta = batch.get("meta") or {}
        exam_ids = meta.get("exam_id")

        # Normalize exam_ids to a python list[int]
        if torch.is_tensor(exam_ids):
            exam_ids_l = [int(x) for x in exam_ids.detach().cpu().numpy().tolist()]
        elif isinstance(exam_ids, (list, tuple)):
            exam_ids_l = [int(x) for x in exam_ids]
        else:
            exam_ids_l = [int(exam_ids)] * int(x.shape[0])

        # Normalize report_text to a list[str]
        if isinstance(texts, (list, tuple)):
            texts_l = [str(t) for t in texts]
        else:
            texts_l = [str(texts)] * int(x.shape[0])

        if autocast_ctx is None:
            _, feat = fwf(x)
        else:
            with autocast_ctx:
                _, feat = fwf(x)

        if feat.ndim != 3 or int(feat.shape[1]) != 2:
            raise ValueError(f"expected feat (B,2,D), got {tuple(feat.shape)}")

        img_emb = feat.mean(dim=1).detach().float().cpu().numpy()  # (B,D)
        for eid, t, e in zip(exam_ids_l, texts_l, img_emb):
            if max_items is not None and n_seen >= int(max_items):
                break
            img_list.append(np.asarray(e, dtype=np.float32))
            text_list.append(str(t) if t is not None else "")
            exam_id_list.append(int(eid))
            n_seen += 1

    if not img_list:
        return np.zeros((0, 0), dtype=np.float32), [], []
    img = np.stack(img_list, axis=0).astype(np.float32)
    return img, text_list, exam_id_list


def _recall_at_k(sim: torch.Tensor, ks: list[int]) -> dict[str, float]:
    """
    sim: (N, N) similarity matrix where sim[i, j] is the score of (image i, text j).
    Correct match is diagonal i==j.
    """
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"sim must be square (N,N), got {tuple(sim.shape)}")
    n = int(sim.shape[0])
    if n <= 0:
        return {f"R@{k}": 0.0 for k in ks}

    max_k = max(int(k) for k in ks if int(k) > 0) if ks else 1
    max_k = min(max_k, n)
    topk = sim.topk(k=max_k, dim=1, largest=True, sorted=False).indices  # (N,max_k)
    tgt = torch.arange(n, device=sim.device).view(n, 1)

    out: dict[str, float] = {}
    for k in ks:
        kk = max(1, min(int(k), n))
        hit = (topk[:, :kk] == tgt).any(dim=1).float().mean().item()
        out[f"R@{kk}"] = float(hit)
    return out


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="outputs/.../checkpoints/best.pt 或 last.pt"),
    index_csv: Path = typer.Option(..., exists=True, help="artifacts/splits_dual/*pct/{val,test}.csv 或自定义索引"),
    dicom_base: Path = typer.Option(default_dicom_base()),
    out_json: Path | None = typer.Option(None, help="输出 json（默认保存到 run_dir/reports/）"),
    label_task: str | None = typer.Option(None, help="默认从 checkpoint 读取（没有则按 six_class）"),
    split_name: str = typer.Option("val", help="用于报告标识：val | test | custom"),
    text_encoder: str = typer.Option("hash", help="文本编码：hash（baseline） | hf（HuggingFace/CMBERT）"),
    text_model: str | None = typer.Option(None, help="hf: HF 模型 id/路径（默认从 checkpoint 读取；否则用 hfl/chinese-roberta-wwm-ext）"),
    text_pool: str | None = typer.Option(None, help="hf: pooling=cls|mean（默认从 checkpoint 读取）"),
    text_max_length: int | None = typer.Option(None, help="hf: tokenizer max_length（默认从 checkpoint 读取；否则 256）"),
    text_proj_seed: int | None = typer.Option(None, help="hf: hidden_size->dim 随机投影 seed（默认从 checkpoint 读取；否则 42）"),
    batch_size: int = typer.Option(1),
    num_workers: int = typer.Option(4),
    amp: bool = typer.Option(True),
    num_slices: int | None = typer.Option(None, help="默认从 checkpoint 读取"),
    image_size: int | None = typer.Option(None, help="默认从 checkpoint 读取"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="使用 cache/ 缓存体数据，提高吞吐"),
    cache_dir: Path = typer.Option(Path("cache/dual_volumes"), help="缓存目录（不入库）"),
    cache_dtype: str = typer.Option("float16", help="缓存数据类型：float16 | float32"),
    ks: str = typer.Option("1,5,10", help="Recall@K 列表（逗号分隔）"),
    max_items: int | None = typer.Option(None, help="调试：最多评估 N 条（None=全部）"),
) -> None:
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)

    ckpt = torch.load(checkpoint, map_location="cpu")
    model_name = str(ckpt.get("model_name", "dual_resnet10_3d"))
    model_kwargs = dict(ckpt.get("model_kwargs", {}) or {})
    num_classes = int(ckpt.get("num_classes", len(CLASS_ID_TO_NAME)))
    if num_slices is None:
        num_slices = int(ckpt.get("num_slices", 32))
    if image_size is None:
        image_size = int(ckpt.get("image_size", 224))

    inferred_task = label_task or ckpt.get("label_task")
    if inferred_task is None:
        inferred_task = "six_class"
    task_spec = resolve_task(str(inferred_task))
    if int(task_spec.num_classes) != int(num_classes):
        typer.echo(f"warning: task={task_spec.name} expects num_classes={task_spec.num_classes} but ckpt has {num_classes}")

    used_cache_dir = cache_dir / f"d{int(num_slices)}_s{int(image_size)}"
    if not cache:
        used_cache_dir = None

    pre = ckpt.get("preprocess")
    if isinstance(pre, dict):
        crop_size = int(pre.get("crop_size", 192) or 192)
        sampling = str(pre.get("sampling", "air_block") or "air_block")
        block_len = int(pre.get("block_len", 64) or 64)
        target_spacing = float(pre.get("target_spacing", 0.0) or 0.0)
        target_z_spacing = float(pre.get("target_z_spacing", 0.0) or 0.0)
        window_wl = float(pre.get("window_wl", 700.0) or 700.0)
        window_ww = float(pre.get("window_ww", 4000.0) or 4000.0)
        window2_wl = float(pre.get("window2_wl", 0.0) or 0.0)
        window2_ww = float(pre.get("window2_ww", 0.0) or 0.0)
        pair_features = str(pre.get("pair_features", "none") or "none").strip().lower()
    else:
        crop_size = 192
        sampling = "air_block"
        block_len = 64
        target_spacing = 0.7
        target_z_spacing = 0.8
        window_wl = 700.0
        window_ww = 4000.0
        window2_wl = 0.0
        window2_ww = 0.0
        pair_features = "none"

    w2_ww = float(window2_ww)
    w2_wl = float(window2_wl)
    window2_wl_v = w2_wl if w2_ww > 0 else None
    window2_ww_v = w2_ww if w2_ww > 0 else None
    base_channels = 2 if window2_ww_v is not None else 1
    pair_factor = 3 if str(pair_features) == "self_other_diff" else 1
    in_channels = int(base_channels) * int(pair_factor)

    preprocess_spec = DualPreprocessSpec(
        num_slices=int(num_slices),
        image_size=int(image_size),
        crop_size=int(crop_size),
        window_wl=float(window_wl),
        window_ww=float(window_ww),
        window2_wl=window2_wl_v,
        window2_ww=window2_ww_v,
        pair_features=str(pair_features),
        sampling=str(sampling),
        block_len=int(block_len),
        flip_right=True,
        target_spacing=float(target_spacing) if float(target_spacing) > 0 else None,
        target_z_spacing=float(target_z_spacing) if float(target_z_spacing) > 0 else None,
    )

    ds = EarCTDualDataset(
        index_df=df,
        dicom_root=dicom_root,
        spec=preprocess_spec,
        cache_dir=used_cache_dir,
        cache_dtype=cache_dtype,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (int(num_slices), int(image_size), int(image_size))

    model, _ = make_dual_model(
        model_name,
        num_classes=num_classes,
        in_channels=int(in_channels),
        img_size=img_size,
        vit_patch_size=tuple(model_kwargs.get("vit_patch_size", (4, 16, 16))),
        vit_pool=str(model_kwargs.get("vit_pool", "cls")),
        vit_hidden_size=int(model_kwargs.get("vit_hidden_size", 768)),
        vit_mlp_dim=int(model_kwargs.get("vit_mlp_dim", 3072)),
        vit_num_layers=int(model_kwargs.get("vit_num_layers", 12)),
        vit_num_heads=int(model_kwargs.get("vit_num_heads", 12)),
        unet_channels=tuple(model_kwargs.get("unet_channels", (16, 32, 64, 128, 256))),
        unet_strides=tuple(model_kwargs.get("unet_strides", (2, 2, 2, 2))),
        unet_num_res_units=int(model_kwargs.get("unet_num_res_units", 2)),
    )
    model = model.to(device)
    state_dict = _strip_compile_prefix(dict(ckpt.get("state_dict", {}) or {}))
    model.load_state_dict(state_dict, strict=True)

    img_emb, texts, exam_ids = _extract_embeddings(model=model, loader=loader, device=device, amp=bool(amp), max_items=max_items)
    if img_emb.size == 0:
        raise typer.Exit(code=2)

    keep = [i for i, t in enumerate(texts) if str(t).strip()]
    if len(keep) < 2:
        raise RuntimeError("not enough non-empty report_text rows for retrieval eval (need >=2)")

    img_emb = img_emb[keep]
    texts = [texts[i] for i in keep]
    exam_ids = [exam_ids[i] for i in keep]

    dim = int(img_emb.shape[1])
    enc = str(text_encoder).strip().lower()
    if enc == "hash":
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
        except Exception as e:
            raise RuntimeError("eval_dual_retrieval(text_encoder=hash) requires scikit-learn (HashingVectorizer)") from e

        vec = HashingVectorizer(
            n_features=int(dim),
            analyzer="char",
            ngram_range=(1, 2),
            lowercase=False,
            alternate_sign=False,
            norm=None,
        )
        txt_emb = vec.transform(texts).toarray().astype("float32")
        txt_t = torch.from_numpy(txt_emb).to(device=device, dtype=torch.float32)
        notes = "Semantic Consistency retrieval metrics (check.md 5.2.3); text embeddings use HashingVectorizer baseline."
    elif enc == "hf":
        from medical_fenlei.text_encoder import encode_texts_to_dim

        model_name = str(text_model or model_kwargs.get("proto_text_model") or "hfl/chinese-roberta-wwm-ext")
        pool = str(text_pool or model_kwargs.get("proto_text_pool") or "cls")
        max_len = int(text_max_length or model_kwargs.get("proto_text_max_length") or 256)
        proj_seed = int(text_proj_seed or model_kwargs.get("proto_text_proj_seed") or 42)

        txt_cpu = encode_texts_to_dim(
            texts,
            dim=int(dim),
            encoder="hf",
            hf_model_name_or_path=model_name,
            hf_pool=pool,
            hf_max_length=int(max_len),
            proj_seed=int(proj_seed),
        )
        txt_t = txt_cpu.to(device=device, dtype=torch.float32)
        notes = f"Semantic Consistency retrieval metrics (check.md 5.2.3); text embeddings use HF({model_name}) pool={pool}."
    else:
        raise ValueError(f"unknown text_encoder: {text_encoder!r} (expected hash|hf)")

    img_t = torch.from_numpy(img_emb).to(device=device, dtype=torch.float32)
    img_t = torch.nn.functional.normalize(img_t, dim=-1)
    txt_t = torch.nn.functional.normalize(txt_t, dim=-1)
    sim = img_t @ txt_t.t()  # (N,N)

    k_list = _parse_int_list(ks)
    if not k_list:
        k_list = [1, 5, 10]
    k_list = sorted({max(1, int(k)) for k in k_list})

    img2txt = _recall_at_k(sim, k_list)
    txt2img = _recall_at_k(sim.t(), k_list)

    report: dict[str, Any] = {
        "task": {"name": str(task_spec.name), "kind": str(task_spec.kind)},
        "checkpoint": str(checkpoint),
        "index_csv": str(index_csv),
        "split": str(split_name),
        "n": int(sim.shape[0]),
        "dim": int(dim),
        "recall": {"image_to_text": img2txt, "text_to_image": txt2img},
        "notes": notes,
    }

    out_dir = out_json.parent if out_json is not None else (_default_out_dir(checkpoint) / "reports")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_json or (out_dir / f"eval_retrieval_{str(split_name)}.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Also write the aligned ids for later plotting/debugging.
    pairs_df = pd.DataFrame({"i": np.arange(len(exam_ids), dtype=np.int64), "exam_id": np.asarray(exam_ids, dtype=np.int64), "report_text": texts})
    pairs_df.to_csv(out_path.with_suffix(".pairs.csv"), index=False)


if __name__ == "__main__":
    app()
