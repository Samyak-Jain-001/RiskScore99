from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ModelRegistry
from app.services.feature_engineering import build_feature_matrix
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    pass


def _load_registry_entry(
    db: Session, fixed_version: Optional[str] = None
) -> Optional[ModelRegistry]:
    query = db.query(ModelRegistry)
    if fixed_version:
        return query.filter(ModelRegistry.model_version == fixed_version).first()
    return query.order_by(ModelRegistry.created_at.desc()).first()


def load_artifacts(db: Session) -> Tuple[Any, Dict[str, Any], str]:
    """Load preprocessor+model pipeline and metadata; raises if not available."""
    entry = _load_registry_entry(db, settings.fixed_model_version)
    if not entry:
        raise ModelNotTrainedError("No model registry entry found – run training first.")

    artifact_dir = Path(entry.artifact_path)
    pipeline_path = artifact_dir / "pipeline.joblib"
    metadata_path = artifact_dir / "metadata.json"

    if not pipeline_path.exists():
        raise ModelNotTrainedError(f"Missing pipeline artifact at {pipeline_path}")

    pipeline = joblib.load(pipeline_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return pipeline, metadata, entry.model_version


def score_single(db: Session, record: Dict[str, Any]) -> Dict[str, Any]:
    pipeline, metadata, model_version = load_artifacts(db)

    df = pd.DataFrame([record])
    X_df, _, _ = build_feature_matrix(df)

    proba: np.ndarray
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_df)[:, 1]
    elif isinstance(pipeline, CalibratedClassifierCV):
        proba = pipeline.predict_proba(X_df)[:, 1]
    else:
        preds = pipeline.predict(X_df)
        proba = preds.astype(float)

    p = float(np.clip(proba[0], 0.0, 1.0))
    risk_score = int(round(99 * p))

    return {
        "probability": p,
        "risk_score": risk_score,
        "model_version": model_version,
        "metadata": metadata,
    }

