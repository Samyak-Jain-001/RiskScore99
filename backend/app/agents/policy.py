from __future__ import annotations

from typing import Any, Dict, List

from app.config import settings


class PolicyAgent:
    """
    Maps risk scores and simple business rules to decisions.
    """

    def decide(self, record: Dict[str, Any], probability: float, risk_score: int) -> str:
        t1 = int(round(99 * settings.threshold_t1))
        t2 = int(round(99 * settings.threshold_t2))
        t3 = int(round(99 * settings.threshold_t3))

        amount = float(record.get("TransactionAmt") or 0.0)
        missing_identity = any(
            key.startswith("id_") and record.get(key) is None for key in record.keys()
        )

        decision: str
        if risk_score < t1:
            decision = "APPROVE"
        elif risk_score < t2:
            decision = "CHALLENGE"
        elif risk_score < t3:
            decision = "REVIEW"
        else:
            decision = "BLOCK"

        if amount > 5000 and risk_score >= t2:
            decision = "BLOCK"

        if missing_identity and decision == "APPROVE":
            decision = "REVIEW"

        return decision

