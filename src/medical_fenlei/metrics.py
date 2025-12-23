from __future__ import annotations

from typing import Any

import numpy as np
import torch


def classification_report_from_confusion(
    cm: torch.Tensor | np.ndarray,
    *,
    class_id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Compute common multi-class classification metrics from a confusion matrix.

    Confusion matrix definition:
      cm[i, j] = count(true=i, pred=j)
    """
    if torch.is_tensor(cm):
        cm_np = cm.detach().cpu().numpy()
    else:
        cm_np = np.asarray(cm)

    if cm_np.ndim != 2 or cm_np.shape[0] != cm_np.shape[1]:
        raise ValueError(f"cm must be square 2D, got shape={cm_np.shape}")

    k = int(cm_np.shape[0])
    total = int(cm_np.sum())
    diag = np.diag(cm_np).astype(np.float64)
    row_sum = cm_np.sum(axis=1).astype(np.float64)
    col_sum = cm_np.sum(axis=0).astype(np.float64)

    supports = row_sum
    tp = diag
    fn = row_sum - tp
    fp = col_sum - tp
    tn = total - tp - fn - fp

    def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        out = np.zeros_like(num, dtype=np.float64)
        m = den > 0
        out[m] = num[m] / den[m]
        return out

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # sensitivity
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    accuracy = float(diag.sum() / total) if total > 0 else 0.0

    macro_precision = float(precision.mean()) if k > 0 else 0.0
    macro_recall = float(recall.mean()) if k > 0 else 0.0
    macro_specificity = float(specificity.mean()) if k > 0 else 0.0
    macro_f1 = float(f1.mean()) if k > 0 else 0.0

    support_sum = float(supports.sum())
    if support_sum > 0:
        weight = supports / support_sum
        weighted_precision = float((precision * weight).sum())
        weighted_recall = float((recall * weight).sum())
        weighted_specificity = float((specificity * weight).sum())
        weighted_f1 = float((f1 * weight).sum())
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_specificity = 0.0
        weighted_f1 = 0.0

    per_class: list[dict[str, Any]] = []
    for i in range(k):
        per_class.append(
            {
                "id": int(i),
                "name": class_id_to_name.get(int(i)) if class_id_to_name else str(i),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "sensitivity": float(recall[i]),
                "specificity": float(specificity[i]),
                "f1": float(f1[i]),
                "support": int(supports[i]),
            }
        )

    return {
        "total": total,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_sensitivity": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_sensitivity": weighted_recall,
        "weighted_specificity": weighted_specificity,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm_np.astype(int).tolist(),
    }
