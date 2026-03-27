from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.model_service import ModelNotTrainedError, score_single, list_active_models
from app.utils.logger import get_logger


logger = get_logger(__name__)


class RiskScorerAgent:
    """
    Scores transactions using the ML pipeline.

    Agentic capabilities:
      - score():          standard scoring using latest/configured model
      - rescore():        re-score with enriched signals (agentic loop)
      - select_model():   dynamic model selection based on tx characteristics
                          (chooses specialized model if available)
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def score(
        self, record: Dict[str, Any], model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Score a transaction. Optionally override the model version."""
        try:
            result = score_single(self.db, record, model_version_override=model_version)
            logger.debug(
                "Scored transaction probability=%.4f risk_score=%d model=%s",
                result["probability"],
                result["risk_score"],
                result["model_version"],
            )
            return result
        except ModelNotTrainedError as exc:
            logger.warning("Model not trained: %s", exc)
            raise

    def rescore(
        self,
        enriched_record: Dict[str, Any],
        previous_score: int,
        deep_signals: List[str],
    ) -> Dict[str, Any]:
        """
        Re-score after deep signal collection.

        The enriched record may have additional derived fields.  The model
        itself only sees the standard feature columns (extras are ignored
        by the ColumnTransformer), BUT the deep signals can apply a
        heuristic adjustment to nudge ambiguous scores.
        """
        # First, get the model's raw re-score on the (possibly enriched) record
        result = self.score(enriched_record)
        model_risk_score = result["risk_score"]
        model_probability = result["probability"]

        # Apply heuristic nudge based on deep signals
        nudge = 0
        for signal in deep_signals:
            if "LOW_IDENTITY_COMPLETENESS" in signal:
                nudge += 5
            elif "NO_IDENTITY_FIELDS" in signal:
                nudge += 8
            elif "DEVICE_EMAIL_CROSS_RISK" in signal:
                nudge += 4
            elif "NIGHTTIME_TRANSACTION" in signal:
                nudge += 3
            elif "ROUND_AMOUNT" in signal:
                nudge += 2

        adjusted_score = min(99, max(0, model_risk_score + nudge))
        adjusted_probability = adjusted_score / 99.0

        logger.info(
            "Rescore: model_score=%d nudge=+%d adjusted_score=%d (previous=%d, %d deep_signals)",
            model_risk_score, nudge, adjusted_score, previous_score, len(deep_signals),
        )

        return {
            "probability": adjusted_probability,
            "risk_score": adjusted_score,
            "model_version": result["model_version"],
            "metadata": result.get("metadata", {}),
            "rescored": True,
            "original_risk_score": previous_score,
            "nudge_applied": nudge,
        }

    def select_model(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Dynamic model selection based on transaction characteristics.

        If multiple active models exist in the registry, this method
        chooses the most appropriate one.  For now: if the transaction
        amount is very high (>5000), prefer the most recent model
        (assumed to have better high-value coverage).  Otherwise, use
        the default (latest or fixed).

        Returns a model_version string or None (= use default).
        """
        active_models = list_active_models(self.db)
        if len(active_models) <= 1:
            return None  # only one model, no selection needed

        amt = float(record.get("TransactionAmt") or 0)
        if amt > 5000:
            # Use the most recent model for high-value transactions
            latest = max(active_models, key=lambda m: m.created_at)
            logger.info(
                "Dynamic model selection: high-value tx (%.2f) → model %s",
                amt, latest.model_version,
            )
            return latest.model_version

        return None  # default selection
