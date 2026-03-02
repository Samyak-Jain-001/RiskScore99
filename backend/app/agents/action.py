from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.services.data_service import add_audit_log, create_scored_transaction


class ActionAgent:
    """
    Persists scored transactions and audit logs.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def persist(
        self,
        raw_record: Dict[str, Any],
        probability: float,
        risk_score: int,
        decision: str,
        reason_codes: List[str],
        model_version: str | None,
    ):
        tx = create_scored_transaction(
            self.db,
            raw_json=raw_record,
            probability=probability,
            score=risk_score,
            decision=decision,
            reason_codes=reason_codes,
            model_version=model_version,
        )

        add_audit_log(
            self.db,
            event_type="score_transaction",
            payload={
                "transaction_scored_id": tx.id,
                "decision": decision,
                "probability": probability,
                "score": risk_score,
                "model_version": model_version,
                "reason_codes": reason_codes,
            },
        )

        return tx

