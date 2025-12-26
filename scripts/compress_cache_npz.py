from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)


def _convert_one(src_npy: Path, *, in_dir: Path, out_dir: Path) -> None:
    rel = src_npy.relative_to(in_dir)
    dst_npz = (out_dir / rel).with_suffix(".npz")
    dst_npz.parent.mkdir(parents=True, exist_ok=True)

    arr = np.load(src_npy)
    tmp = dst_npz.with_suffix(dst_npz.suffix + ".tmp")
    try:
        np.savez_compressed(tmp, arr=arr)
        tmp.replace(dst_npz)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    # Sidecar json (ear cache) — copy if present.
    src_json = src_npy.with_suffix(".json")
    if src_json.exists():
        dst_json = dst_npz.with_suffix(".json")
        try:
            shutil.copy2(src_json, dst_json)
        except Exception:
            pass


@app.command()
def main(
    in_dir: Path = typer.Option(..., exists=True, help="包含 .npy 的缓存目录（如 cache/ears_hu/d32_... 或 cache/dual_volumes/d32_...）"),
    out_dir: Path = typer.Option(..., help="输出 .npz 缓存目录（不入库）"),
    num_workers: int = typer.Option(16, help="并行线程数（压缩是 CPU 密集）"),
    overwrite: bool = typer.Option(False, help="如果目标已存在则覆盖"),
) -> None:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(in_dir.rglob("*.npy"))
    if not npy_files:
        typer.echo(f"no .npy found under: {in_dir}")
        raise typer.Exit(code=2)

    # Optional overwrite protection.
    if not overwrite:
        any_existing = False
        for src in npy_files[:2000]:  # cheap check
            rel = src.relative_to(in_dir)
            dst = (out_dir / rel).with_suffix(".npz")
            if dst.exists():
                any_existing = True
                break
        if any_existing:
            raise ValueError(f"out_dir already contains .npz; pass --overwrite to replace: {out_dir}")

    done = 0
    with ThreadPoolExecutor(max_workers=int(num_workers)) as ex:
        futs = [ex.submit(_convert_one, p, in_dir=in_dir, out_dir=out_dir) for p in npy_files]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 500 == 0:
                typer.echo(f"compressed: {done}/{len(npy_files)}")

    typer.echo(f"done: {done} files")
    typer.echo(f"out_dir: {out_dir}")


if __name__ == "__main__":
    app()

