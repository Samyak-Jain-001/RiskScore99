from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.utils.logger import get_logger
from app.utils.schema import TransactionInput
from app.utils.validators import normalize_email_domain, validate_transaction_payload


logger = get_logger(__name__)


class SignalCollectorAgent:
    """
    Collects, normalizes, and validates incoming transaction signals.

    Supports two modes:
      - collect():      standard first-pass signal collection
      - deep_collect():  enhanced second-pass when risk score is ambiguous
                         (agentic re-evaluation loop)
    """

    def collect(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Standard signal collection — validate, normalize, emit warnings."""
        tx = TransactionInput(**payload)
        data = tx.model_dump()

        data["P_emaildomain"] = normalize_email_domain(data.get("P_emaildomain"))
        data["R_emaildomain"] = normalize_email_domain(data.get("R_emaildomain"))

        warnings = validate_transaction_payload(data)

        logger.debug("Collected signals for TransactionID=%s", data.get("TransactionID"))
        return data, warnings

    def deep_collect(
        self, record: Dict[str, Any], initial_risk_score: int
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Enhanced signal collection triggered when the initial risk score
        falls in the ambiguous band.  Derives additional risk indicators
        that can shift the score on re-evaluation.

        This is the AGENTIC component: the orchestrator decides to loop back
        to signal collection based on an intermediate result, and the agent
        enriches the record with new signals that weren't computed on the
        first pass.
        """
        enriched = dict(record)
        extra_warnings: List[str] = []

        # ── 1. Amount anomaly signal ────────────────────────────────
        # Flag transactions with suspicious decimal patterns
        # (exact round numbers or .99 endings are fraud signals in real data)
        amt = float(record.get("TransactionAmt") or 0)
        decimal_part = amt % 1
        if decimal_part == 0 and amt > 100:
            enriched["_amount_is_round"] = True
            extra_warnings.append("DEEP_SIGNAL:ROUND_AMOUNT")

        # ── 2. Identity completeness score ──────────────────────────
        # Count how many id_* fields are populated vs missing
        id_keys = [k for k in record if k.startswith("id_")]
        if id_keys:
            populated = sum(1 for k in id_keys if record.get(k) is not None)
            completeness = populated / len(id_keys)
            enriched["_identity_completeness"] = completeness
            if completeness < 0.3:
                extra_warnings.append("DEEP_SIGNAL:LOW_IDENTITY_COMPLETENESS")
        else:
            enriched["_identity_completeness"] = 0.0
            extra_warnings.append("DEEP_SIGNAL:NO_IDENTITY_FIELDS")

        # ── 3. Device + email cross-signal ──────────────────────────
        # Missing device info combined with unusual email is higher risk
        device_type = record.get("DeviceType")
        device_info = record.get("DeviceInfo")
        p_email = (record.get("P_emaildomain") or "").lower()
        common_emails = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com"}

        if (not device_type or not device_info) and p_email and p_email not in common_emails:
            enriched["_device_email_cross_risk"] = True
            extra_warnings.append("DEEP_SIGNAL:DEVICE_EMAIL_CROSS_RISK")
        else:
            enriched["_device_email_cross_risk"] = False

        # ── 4. Time-of-day risk band ────────────────────────────────
        # Transactions between midnight and 5am are statistically riskier
        tx_dt = record.get("TransactionDT")
        if tx_dt is not None:
            try:
                hour = int((float(tx_dt) / 3600) % 24)
                if 0 <= hour < 5:
                    enriched["_nighttime_transaction"] = True
                    extra_warnings.append("DEEP_SIGNAL:NIGHTTIME_TRANSACTION")
                else:
                    enriched["_nighttime_transaction"] = False
            except (TypeError, ValueError):
                enriched["_nighttime_transaction"] = False
        else:
            enriched["_nighttime_transaction"] = False

        logger.info(
            "Deep collection for TransactionID=%s produced %d extra signals "
            "(initial_risk_score=%d)",
            record.get("TransactionID"),
            len(extra_warnings),
            initial_risk_score,
        )

        return enriched, extra_warnings
