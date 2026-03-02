from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from app.utils.logger import get_logger


logger = get_logger(__name__)


class ExplanationAgent:
    """
    Generates lightweight, human-readable reason codes.
    For MVP, this uses simple heuristics on raw fields and risk score.
    """

    def explain(self, record: Dict[str, Any], probability: float, risk_score: int) -> Dict[str, Any]:
        reasons: List[str] = []

        amt = record.get("TransactionAmt")
        if amt is not None:
            try:
                amt_val = float(amt)
                if amt_val > 2000:
                    reasons.append("HIGH_TRANSACTION_AMOUNT")
                elif amt_val > 500:
                    reasons.append("MEDIUM_TRANSACTION_AMOUNT")
            except (TypeError, ValueError):
                pass

        p_email = (record.get("P_emaildomain") or "").lower()
        if p_email and not any(
            common in p_email for common in ["gmail", "yahoo", "hotmail", "outlook", "live"]
        ):
            reasons.append("UNUSUAL_EMAIL_DOMAIN")

        device_type = (record.get("DeviceType") or "").lower()
        device_info = (record.get("DeviceInfo") or "").lower()
        if not device_type or not device_info:
            reasons.append("DEVICE_RISK_SIGNAL")

        identity_missing = 0
        for key in record.keys():
            if key.startswith("id_") and record.get(key) in (None, np.nan):
                identity_missing += 1
        if identity_missing > 0:
            reasons.append("IDENTITY_MISMATCH_SIGNAL")

        if risk_score >= 80:
            reasons.append("MODEL_TOP_FEATURES")

        if not reasons:
            reasons.append("MODEL_BASELINE_RISK")

        explanation_text = (
            "RiskScore99 assessed this transaction with a calibrated fraud probability "
            f"of {probability:.3f} (risk score {risk_score}/99). "
            "Reason codes indicate high-level signals such as amount, email domain, "
            "device/identity patterns, and model-driven risk."
        )

        return {"reason_codes": reasons[:5], "explanation_text": explanation_text}

