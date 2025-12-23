from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Rank data with average ranks for ties (1-based ranks), NumPy-only.
    """
    x = np.asarray(x)
    n = int(x.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average rank for [i..j] in 1-based indexing
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    AUROC via Mann–Whitney U (tie-aware), returns None if a class is missing.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(np.float64)
    m_pos = y_true == 1
    m_neg = y_true == 0
    n_pos = int(m_pos.sum())
    n_neg = int(m_neg.sum())
    if n_pos <= 0 or n_neg <= 0:
        return None

    ranks = _rankdata_average_ties(y_score)
    sum_pos = float(ranks[m_pos].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    Average precision (AUPRC) with the standard definition:
      AP = mean(precision@k for each positive in score-sorted order)
    Returns None if no positive.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(np.float64)
    m_pos = y_true == 1
    n_pos = int(m_pos.sum())
    if n_pos <= 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = 0
    precisions: list[float] = []
    for i, y in enumerate(y_sorted, start=1):
        if int(y) == 1:
            tp += 1
            precisions.append(tp / float(i))
    if not precisions:
        return 0.0
    return float(np.mean(np.asarray(precisions, dtype=np.float64)))


def binary_confusion(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float64)
    pred = (y_prob >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    specificity_target: float = 0.95,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float64)

    out: dict[str, Any] = {}
    out["n"] = int(y_true.size)
    out["pos"] = int((y_true == 1).sum())
    out["neg"] = int((y_true == 0).sum())

    out["auroc"] = binary_auroc(y_true, y_prob)
    out["auprc"] = binary_average_precision(y_true, y_prob)

    cm = binary_confusion(y_true, y_prob, threshold=float(threshold))
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    out.update({f"t{threshold:.3f}_{k}": int(v) for k, v in cm.items()})

    out["threshold"] = float(threshold)
    out["accuracy"] = _safe_div(tp + tn, tp + tn + fp + fn)
    out["precision"] = _safe_div(tp, tp + fp)
    out["recall"] = _safe_div(tp, tp + fn)
    out["sensitivity"] = out["recall"]
    out["specificity"] = _safe_div(tn, tn + fp)
    out["f1"] = _safe_div(2 * out["precision"] * out["recall"], out["precision"] + out["recall"])

    # Sensitivity@target specificity.
    neg_scores = y_prob[y_true == 0]
    if neg_scores.size <= 0:
        out["spec_target"] = float(specificity_target)
        out["threshold_at_spec"] = None
        out["sensitivity_at_spec"] = None
    else:
        n_neg = int(neg_scores.size)
        allowed_fp = int(np.floor((1.0 - float(specificity_target)) * n_neg + 1e-9))
        neg_sorted = np.sort(neg_scores)[::-1]  # desc
        if allowed_fp <= 0:
            base = float(neg_sorted[0])
            thr = float(np.nextafter(base, 1.0))
        elif allowed_fp >= n_neg:
            base = float(neg_sorted[-1])
            thr = float(np.nextafter(base, 0.0))
        else:
            base = float(neg_sorted[allowed_fp - 1])
            thr = float(np.nextafter(base, 1.0))

        cm2 = binary_confusion(y_true, y_prob, threshold=thr)
        tp2, tn2, fp2, fn2 = cm2["tp"], cm2["tn"], cm2["fp"], cm2["fn"]
        out["spec_target"] = float(specificity_target)
        out["threshold_at_spec"] = float(thr)
        out["specificity_at_spec"] = _safe_div(tn2, tn2 + fp2)
        out["sensitivity_at_spec"] = _safe_div(tp2, tp2 + fn2)

    return out


def bootstrap_binary_metrics_by_exam(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    exam_ids: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
    specificity_target: float = 0.95,
) -> dict[str, Any]:
    """
    Bootstrap metrics by exam_id (group resampling) to avoid overly optimistic CI
    due to left/right correlation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float64)
    exam_ids = np.asarray(exam_ids).astype(np.int64)
    if y_true.size != y_prob.size or y_true.size != exam_ids.size:
        raise ValueError("y_true/y_prob/exam_ids must have same length")

    uniq = np.unique(exam_ids)
    if uniq.size <= 0:
        return {"n_boot": 0, "ci": {}}

    rng = np.random.default_rng(int(seed))
    metrics_list: list[dict[str, Any]] = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(uniq, size=int(uniq.size), replace=True)
        mask = np.isin(exam_ids, sampled)
        m = binary_metrics(y_true[mask], y_prob[mask], threshold=float(threshold), specificity_target=float(specificity_target))
        metrics_list.append(m)

    def _ci(key: str) -> dict[str, float | None]:
        vals: list[float] = []
        for m in metrics_list:
            v = m.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            return {"mean": None, "p2.5": None, "p97.5": None}
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "p2.5": float(np.quantile(arr, 0.025)), "p97.5": float(np.quantile(arr, 0.975))}

    keys = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1", "sensitivity_at_spec", "specificity_at_spec"]
    return {"n_boot": int(n_boot), "seed": int(seed), "ci": {k: _ci(k) for k in keys}}


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
