from __future__ import annotations

from typing import Any, Dict

from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class PolicyAgent:
    """
    Maps risk scores and business rules to routing decisions.

    Returns both the decision string and an escalation flag that tells
    the orchestrator whether enhanced explanation is needed.
    """

    def decide(
        self, record: Dict[str, Any], probability: float, risk_score: int
    ) -> str:
        """Original interface — returns just the decision string."""
        result = self.decide_with_context(record, probability, risk_score)
        return result["decision"]

    def decide_with_context(
        self, record: Dict[str, Any], probability: float, risk_score: int
    ) -> Dict[str, Any]:
        """
        Full decisioning with context — returns decision + metadata
        that the orchestrator uses for agentic routing.
        """
        t1 = int(round(99 * settings.threshold_t1))
        t2 = int(round(99 * settings.threshold_t2))
        t3 = int(round(99 * settings.threshold_t3))

        amount = float(record.get("TransactionAmt") or 0.0)
        missing_identity = any(
            key.startswith("id_") and record.get(key) is None for key in record.keys()
        )

        decision: str
        applied_rules: list[str] = []

        if risk_score < t1:
            decision = "APPROVE"
        elif risk_score < t2:
            decision = "CHALLENGE"
        elif risk_score < t3:
            decision = "REVIEW"
        else:
            decision = "BLOCK"

        # Business rule overrides
        # Business rule overrides
        if amount > 5000 and risk_score >= t2:
            decision = "BLOCK"
            applied_rules.append("HIGH_VALUE_BLOCK")

        # Standalone high-amount safeguard — flag for review even if model score is low
        # Standalone high-amount safeguard — escalate low-confidence decisions on big transactions
        if amount > 5000 and decision in ("APPROVE", "CHALLENGE"):
            decision = "REVIEW"
            applied_rules.append("HIGH_VALUE_REVIEW_SAFEGUARD")

        if missing_identity and decision == "APPROVE":
            decision = "REVIEW"
            applied_rules.append("MISSING_IDENTITY_DOWNGRADE")

        # ── Escalation flag (agentic signal to orchestrator) ────────
        needs_enhanced_explanation = decision in ("BLOCK", "REVIEW")

        logger.debug(
            "Policy decision=%s for risk_score=%d (thresholds: t1=%d t2=%d t3=%d, rules=%s)",
            decision, risk_score, t1, t2, t3, applied_rules,
        )

        return {
            "decision": decision,
            "needs_enhanced_explanation": needs_enhanced_explanation,
            "applied_rules": applied_rules,
            "thresholds_used": {"t1": t1, "t2": t2, "t3": t3},
        }
