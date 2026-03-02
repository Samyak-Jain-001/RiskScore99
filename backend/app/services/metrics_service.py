from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
from sklearn import metrics

from app.config import settings


def compute_core_metrics(y_true, y_scores) -> Dict[str, Any]:
    roc_auc = metrics.roc_auc_score(y_true, y_scores)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
    pr_auc = metrics.auc(recall, precision)
    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


def threshold_sweep(
    y_true,
    y_scores,
    thresholds: Iterable[float] | None = None,
) -> Dict[str, Any]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 11)

    rows = []
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        prec = metrics.precision_score(y_true, y_pred, zero_division=0)
        rec = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

        avg_amt = 1.0
        cost_fn = settings.fraud_loss_multiplier * avg_amt
        cost_fp = settings.friction_cost
        expected_cost = fn * cost_fn + fp * cost_fp

        rows.append(
            {
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "expected_cost": float(expected_cost),
            }
        )

    return {"thresholds": rows}

