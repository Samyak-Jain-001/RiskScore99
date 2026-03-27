from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.data_service import add_audit_log, create_scored_transaction
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ActionAgent:
    """
    Persists scored transactions and audit logs.
    Now tracks latency and whether the agentic re-evaluation loop fired.
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
        latency_ms: float | None = None,
        scoring_passes: int = 1,
    ):
        tx = create_scored_transaction(
            self.db,
            raw_json=raw_record,
            probability=probability,
            score=risk_score,
            decision=decision,
            reason_codes=reason_codes,
            model_version=model_version,
            latency_ms=latency_ms,
            scoring_passes=scoring_passes,
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
                "latency_ms": latency_ms,
                "scoring_passes": scoring_passes,
            },
        )

        return tx
