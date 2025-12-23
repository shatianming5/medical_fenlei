from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from medical_fenlei.constants import CLASS_ID_TO_NAME
from medical_fenlei.data.dataset import EarCTDataset
from medical_fenlei.models.slice_resnet import SliceMeanResNet
from medical_fenlei.paths import infer_dicom_root

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, help="训练输出的 best.pt/last.pt"),
    index_csv: Path = typer.Option(Path("artifacts/dataset_index.csv"), exists=True),
    dicom_base: Path = typer.Option(Path("data/medical_data_2")),
    out_csv: Path = typer.Option(Path("artifacts/predictions.csv")),
    batch_size: int = typer.Option(1),
    num_workers: int = typer.Option(4),
    num_slices: int = typer.Option(32),
    image_size: int = typer.Option(224),
) -> None:
    dicom_root = infer_dicom_root(dicom_base)
    df = pd.read_csv(index_csv)
    if df.empty:
        raise typer.Exit(code=2)

    ckpt = torch.load(checkpoint, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", len(CLASS_ID_TO_NAME)))

    ds = EarCTDataset(index_df=df, dicom_root=dicom_root, num_slices=num_slices, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceMeanResNet(num_classes=num_classes, in_channels=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            left_logits = model(left)
            right_logits = model(right)
            left_pred = left_logits.argmax(dim=1).cpu().numpy()
            right_pred = right_logits.argmax(dim=1).cpu().numpy()

            meta = batch["meta"]
            exam_ids = meta["exam_id"].cpu().numpy().tolist() if torch.is_tensor(meta["exam_id"]) else meta["exam_id"]
            dates = meta["date"]
            series_relpaths = meta["series_relpath"]
            for exam_id, date, series_relpath, lp, rp in zip(exam_ids, dates, series_relpaths, left_pred, right_pred):
                rows.append(
                    {
                        "exam_id": int(exam_id),
                        "date": str(date),
                        "series_relpath": str(series_relpath),
                        "left_pred": int(lp),
                        "right_pred": int(rp),
                        "left_pred_name": CLASS_ID_TO_NAME.get(int(lp)),
                        "right_pred_name": CLASS_ID_TO_NAME.get(int(rp)),
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    typer.echo(f"saved: {out_csv}")


if __name__ == "__main__":
    app()
