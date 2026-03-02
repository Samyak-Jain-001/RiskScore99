from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from app.utils.logger import get_logger
from app.utils.schema import TransactionInput
from app.utils.validators import normalize_email_domain, validate_transaction_payload


logger = get_logger(__name__)


class SignalCollectorAgent:
    """
    Collects, normalizes, and lightly validates incoming transaction signals.
    """

    def collect(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        tx = TransactionInput(**payload)
        data = tx.model_dump()

        data["P_emaildomain"] = normalize_email_domain(data.get("P_emaildomain"))
        data["R_emaildomain"] = normalize_email_domain(data.get("R_emaildomain"))

        warnings = validate_transaction_payload(data)

        logger.debug("Collected signals for TransactionID=%s", data.get("TransactionID"))
        return data, warnings

