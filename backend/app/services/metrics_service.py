from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

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
    avg_amt: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Sweep probability thresholds and compute confusion + cost metrics.

    Args:
        avg_amt: Real average transaction amount from training data.
                 Falls back to settings.avg_transaction_amt if not provided.
                 Previously hardcoded to 1.0 which made cost analysis meaningless.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 11)

    # FIX: use real avg_amt instead of hardcoded 1.0
    if avg_amt is None:
        avg_amt = settings.avg_transaction_amt

    rows = []
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        cm = metrics.confusion_matrix(y_true, y_pred)
        # Handle edge cases where confusion matrix may not be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # All predictions are the same class
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if (y_pred == 0).all():
                    tn = int(cm[0, 0])
                    fn = int((y_true == 1).sum())
                else:
                    tp = int(cm[0, 0])
                    fp = int((y_true == 0).sum())

        prec = metrics.precision_score(y_true, y_pred, zero_division=0)
        rec = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

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
                "avg_amt_used": float(avg_amt),
            }
        )

    return {"thresholds": rows}
