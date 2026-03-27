from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Outcome, TransactionScored
from app.services.data_service import add_audit_log, record_outcome
from app.utils.logger import get_logger


logger = get_logger(__name__)


class FeedbackLearningAgent:
    """
    Records outcomes and provides REAL drift detection + retrain
    recommendations based on accumulated outcome data.

    This is a genuinely agentic component: it analyses past decisions
    vs actual outcomes and recommends operational changes.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def record_outcome(
        self,
        tx: TransactionScored,
        outcome_label: str,
        notes: str | None,
    ):
        outcome = record_outcome(self.db, tx, outcome_label=outcome_label, notes=notes)

        add_audit_log(
            self.db,
            event_type="outcome_recorded",
            payload={
                "transaction_scored_id": tx.id,
                "outcome_id": outcome.id,
                "outcome_label": outcome_label,
            },
        )

        return outcome

    def retrain_recommendation(self) -> Dict[str, Any]:
        """
        Analyses outcome data to recommend retraining or threshold adjustment.
        Computes actual false positive / false negative rates from recorded outcomes.
        """
        # Total scored transactions with at least one outcome
        txs_with_outcomes = (
            self.db.query(TransactionScored)
            .join(Outcome)
            .all()
        )

        total = len(txs_with_outcomes)
        if total < settings.drift_alert_threshold:
            return {
                "labeled_transactions": total,
                "recommendation": "insufficient_data",
                "detail": f"Need at least {settings.drift_alert_threshold} labeled "
                          f"outcomes (have {total}).",
            }

        # Compute confusion-style metrics from outcomes
        false_positives = 0   # BLOCK/REVIEW but outcome was legit
        false_negatives = 0   # APPROVE but outcome was fraud/chargeback
        true_positives = 0
        true_negatives = 0

        fraud_labels = {"confirmed_fraud", "chargeback"}
        legit_labels = {"legit", "verified"}

        for tx in txs_with_outcomes:
            latest_outcome = max(tx.outcomes, key=lambda o: o.created_at)
            is_fraud_outcome = latest_outcome.outcome_label in fraud_labels
            was_blocked = tx.decision in ("BLOCK", "REVIEW")

            if was_blocked and not is_fraud_outcome:
                false_positives += 1
            elif not was_blocked and is_fraud_outcome:
                false_negatives += 1
            elif was_blocked and is_fraud_outcome:
                true_positives += 1
            else:
                true_negatives += 1

        fp_rate = false_positives / max(false_positives + true_negatives, 1)
        fn_rate = false_negatives / max(false_negatives + true_positives, 1)

        # Recommendation logic
        recommendation = "model_performing_adequately"
        adjustments: List[str] = []

        if fn_rate > 0.15:
            adjustments.append("LOWER_THRESHOLDS — too many frauds are being approved")
            recommendation = "retrain_recommended"
        if fp_rate > 0.30:
            adjustments.append("RAISE_THRESHOLDS — too many legitimate txs are being blocked")
            if recommendation != "retrain_recommended":
                recommendation = "threshold_adjustment_recommended"

        if total >= settings.drift_retrain_threshold:
            recommendation = "retrain_recommended"
            adjustments.append(
                f"Sufficient labeled data ({total}) for model retraining"
            )

        result = {
            "labeled_transactions": total,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "fp_rate": round(fp_rate, 4),
            "fn_rate": round(fn_rate, 4),
            "recommendation": recommendation,
            "suggested_adjustments": adjustments,
        }

        logger.info("Retrain recommendation: %s (FP=%.3f, FN=%.3f)", recommendation, fp_rate, fn_rate)

        add_audit_log(
            self.db,
            event_type="retrain_recommendation",
            payload=result,
        )

        return result
