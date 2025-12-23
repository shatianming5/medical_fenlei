from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.side_dataset import EarCTSideDataset
from medical_fenlei.models.resnet3d import ResNet10_3D
from medical_fenlei.models.slice_resnet import SliceMeanResNet
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


def _make_model(name: str, *, num_classes: int) -> torch.nn.Module:
    if name == "slice_mean_resnet18":
        return SliceMeanResNet(num_classes=num_classes, in_channels=1, pretrained=False)
    if name == "resnet10_3d":
        return ResNet10_3D(num_classes=num_classes, in_channels=1)
    raise ValueError(f"unknown model: {name}")


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="outputs/.../checkpoints/best.pt 或 last.pt"),
    index_csv: Path = typer.Option(..., exists=True, help="artifacts/splits/*pct/val.csv 或自定义索引"),
    dicom_base: Path = typer.Option(Path("data/medical_data_2")),
    out_csv: Path = typer.Option(Path("artifacts/predictions_side.csv")),
    batch_size: int = typer.Option(1),
    num_workers: int = typer.Option(4),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
) -> None:
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    dicom_root = infer_dicom_root(dicom_base)
    ds = EarCTSideDataset(index_df=df, dicom_root=dicom_root, num_slices=num_slices, image_size=image_size, flip_right=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    ckpt = torch.load(checkpoint, map_location="cpu")
    model_name = str(ckpt.get("model_name", "slice_mean_resnet18"))
    num_classes = int(ckpt.get("num_classes", len(CLASS_ID_TO_NAME)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].cpu().numpy()
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()

            meta = batch["meta"]
            exam_ids = meta["exam_id"].cpu().numpy().tolist() if torch.is_tensor(meta["exam_id"]) else meta["exam_id"]
            dates = meta["date"]
            sides = meta["side"]
            series_relpaths = meta["series_relpath"]

            for exam_id, date, side, series_relpath, gt, pr in zip(exam_ids, dates, sides, series_relpaths, y, pred):
                rows.append(
                    {
                        "exam_id": int(exam_id),
                        "date": str(date),
                        "side": str(side),
                        "series_relpath": str(series_relpath),
                        "label": int(gt),
                        "label_name": CLASS_ID_TO_NAME.get(int(gt)),
                        "pred": int(pr),
                        "pred_name": CLASS_ID_TO_NAME.get(int(pr)),
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()

