from __future__ import annotations

from typing import Any, Dict

from sqlalchemy.orm import Session

from app.services.data_service import add_audit_log, record_outcome
from app.db.models import TransactionScored


class FeedbackLearningAgent:
    """
    Records outcomes and exposes a simple retrain-recommendation stub.
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
        MVP stub: counts how many outcomes are available and suggests retraining if large.
        """
        count = self.db.query(TransactionScored).count()
        recommendation = "insufficient_data"
        if count > 1000:
            recommendation = "consider_retraining"
        if count > 10000:
            recommendation = "retrain_recommended"

        return {"labeled_transactions": count, "recommendation": recommendation}

