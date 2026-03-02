from __future__ import annotations

from typing import Any, Dict

from sqlalchemy.orm import Session

from app.services.model_service import ModelNotTrainedError, score_single
from app.utils.logger import get_logger


logger = get_logger(__name__)


class RiskScorerAgent:
    def __init__(self, db: Session) -> None:
        self.db = db

    def score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = score_single(self.db, record)
            logger.debug("Scored transaction with probability=%.4f", result["probability"])
            return result
        except ModelNotTrainedError as exc:
            logger.warning("Model not trained: %s", exc)
            raise

