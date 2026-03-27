from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ModelRegistry
from app.services.feature_engineering import build_feature_matrix
from app.utils.logger import get_logger

# ── FIX: Tell joblib to skip wmic CPU detection entirely ────────────
# This runs before joblib ever tries to detect cores, eliminating the
# warning at its source rather than silencing it.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    pass


# ═══════════════════════════════════════════════════════════════════
# MODEL CACHE — load once, reuse across requests
# ═══════════════════════════════════════════════════════════════════

class _ModelCache:
    """
    Thread-safe in-memory cache for the loaded pipeline.
    Avoids hitting disk + joblib deserialization on every request.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline = None
        self._metadata: Dict[str, Any] = {}
        self._model_version: Optional[str] = None
        self._artifact_path: Optional[str] = None

    def get(self, version: str, artifact_path: str) -> Tuple[Any, Dict[str, Any], str]:
        """Return cached pipeline if version matches, otherwise load fresh."""
        with self._lock:
            if self._pipeline is not None and self._model_version == version:
                return self._pipeline, self._metadata, self._model_version

        # Load outside the lock (I/O bound), then store under lock
        pipeline, metadata = self._load_from_disk(artifact_path)

        with self._lock:
            self._pipeline = pipeline
            self._metadata = metadata
            self._model_version = version
            self._artifact_path = artifact_path
            logger.info("Model cache loaded: version=%s from %s", version, artifact_path)

        return pipeline, metadata, version

    def invalidate(self):
        """Force reload on next request (used after rollback)."""
        with self._lock:
            self._pipeline = None
            self._model_version = None
            logger.info("Model cache invalidated")

    @staticmethod
    def _load_from_disk(artifact_path: str) -> Tuple[Any, Dict[str, Any]]:
        artifact_dir = Path(artifact_path)
        pipeline_path = artifact_dir / "pipeline.joblib"
        metadata_path = artifact_dir / "metadata.json"

        if not pipeline_path.exists():
            raise ModelNotTrainedError(f"Missing pipeline artifact at {pipeline_path}")

        pipeline = joblib.load(pipeline_path)
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        return pipeline, metadata


# Singleton cache instance
_cache = _ModelCache()


# ═══════════════════════════════════════════════════════════════════
# REGISTRY LOOKUP
# ═══════════════════════════════════════════════════════════════════

def _load_registry_entry(
    db: Session, fixed_version: Optional[str] = None
) -> Optional[ModelRegistry]:
    query = db.query(ModelRegistry).filter(ModelRegistry.is_active == True)  # noqa: E712
    if fixed_version:
        return query.filter(ModelRegistry.model_version == fixed_version).first()
    return query.order_by(ModelRegistry.created_at.desc()).first()


# ═══════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════

def load_artifacts(
    db: Session, model_version_override: Optional[str] = None
) -> Tuple[Any, Dict[str, Any], str]:
    """Load pipeline from cache (or disk on first call / version change)."""
    version = model_version_override or settings.fixed_model_version
    entry = _load_registry_entry(db, version)
    if not entry:
        raise ModelNotTrainedError("No active model registry entry found – run training first.")

    return _cache.get(entry.model_version, entry.artifact_path)


def score_single(
    db: Session, record: Dict[str, Any], model_version_override: Optional[str] = None
) -> Dict[str, Any]:
    pipeline, metadata, model_version = load_artifacts(db, model_version_override)

    df = pd.DataFrame([record])
    X_df, _, _ = build_feature_matrix(df)

    # ── OPTIMIZATION: Use n_jobs=1 for single-row prediction ────────
    # predict_proba on one row is faster single-threaded.
    # This also completely avoids joblib spawning worker processes.
    proba: np.ndarray
    if hasattr(pipeline, "predict_proba"):
        # Temporarily disable parallelism for the prediction
        proba = _predict_single_threaded(pipeline, X_df)
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


def _predict_single_threaded(pipeline, X_df) -> np.ndarray:
    """
    Run predict_proba without triggering joblib parallelism.
    
    CalibratedClassifierCV and ensemble models internally use n_jobs.
    For a single row, parallel overhead (process spawning, pickling)
    is far more expensive than just running sequentially.
    """
    # Walk through pipeline steps and temporarily set n_jobs=1
    original_njobs = []
    components = _get_all_estimators(pipeline)

    for comp in components:
        if hasattr(comp, "n_jobs"):
            original_njobs.append((comp, comp.n_jobs))
            comp.n_jobs = 1

    try:
        return pipeline.predict_proba(X_df)[:, 1]
    finally:
        # Restore original n_jobs values
        for comp, orig in original_njobs:
            comp.n_jobs = orig


def _get_all_estimators(pipeline) -> list:
    """Recursively find all estimator objects that might have n_jobs."""
    estimators = []

    if hasattr(pipeline, "steps"):
        for _, step in pipeline.steps:
            estimators.extend(_get_all_estimators(step))
    elif hasattr(pipeline, "estimator"):
        estimators.append(pipeline)
        estimators.extend(_get_all_estimators(pipeline.estimator))
    elif hasattr(pipeline, "calibrated_classifiers_"):
        estimators.append(pipeline)
        for cc in pipeline.calibrated_classifiers_:
            estimators.extend(_get_all_estimators(cc))
    elif hasattr(pipeline, "estimator_"):
        estimators.append(pipeline)
        estimators.extend(_get_all_estimators(pipeline.estimator_))

    if hasattr(pipeline, "n_jobs"):
        estimators.append(pipeline)

    return estimators


# ═══════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def list_active_models(db: Session) -> List[ModelRegistry]:
    """Return all active model registry entries."""
    return (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active == True)  # noqa: E712
        .order_by(ModelRegistry.created_at.desc())
        .all()
    )


def rollback_model(db: Session, target_version: str) -> Dict[str, Any]:
    """
    Rollback to a specific model version.
    Deactivates all models, activates the target, and invalidates cache.
    """
    target = db.query(ModelRegistry).filter(
        ModelRegistry.model_version == target_version
    ).first()

    if not target:
        raise ValueError(f"Model version '{target_version}' not found in registry.")

    artifact_dir = Path(target.artifact_path)
    pipeline_path = artifact_dir / "pipeline.joblib"
    if not pipeline_path.exists():
        raise ValueError(
            f"Model artifacts for version '{target_version}' not found at {artifact_dir}."
        )

    # Deactivate all, then activate target
    db.query(ModelRegistry).update({ModelRegistry.is_active: False})
    target.is_active = True
    db.commit()

    # Invalidate cache so next request loads the rolled-back model
    _cache.invalidate()

    logger.info("Rolled back to model version %s", target_version)

    return {
        "model_version": target.model_version,
        "created_at": str(target.created_at),
        "artifact_path": target.artifact_path,
        "metrics": target.metrics_json,
    }


def get_model_registry(db: Session) -> List[Dict[str, Any]]:
    """Return all model versions with their status."""
    models = (
        db.query(ModelRegistry)
        .order_by(ModelRegistry.created_at.desc())
        .all()
    )
    return [
        {
            "model_version": m.model_version,
            "created_at": str(m.created_at),
            "is_active": m.is_active,
            "metrics": m.metrics_json,
            "artifact_path": m.artifact_path,
            "description": m.description,
        }
        for m in models
    ]