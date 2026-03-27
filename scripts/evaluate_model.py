"""
Evaluate the latest trained RiskScore99 model.

- Loads processed IEEE-CIS data and latest artifacts.
- Computes ROC-AUC, PR-AUC, confusion matrices and threshold sweep.
- Saves ROC and PR curves as PNGs.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (BACKEND_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.db.base import Base
from app.db.models import ModelRegistry
from app.db.session import SessionLocal, engine
from app.services.feature_engineering import build_feature_matrix
from app.services.metrics_service import compute_core_metrics, threshold_sweep


DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"


def main() -> None:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    try:
        joined_path = DATA_PROCESSED / "ieee_train_joined.parquet"
        if not joined_path.exists():
            raise SystemExit(
                f"Processed file not found at {joined_path}. "
                "Run scripts/preprocess_ieee_cis.py first."
            )

        latest = session.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).first()
        if not latest:
            raise SystemExit("No model registry entry found. Train a model first.")

        artifact_dir = Path(latest.artifact_path)
        pipeline_path = artifact_dir / "pipeline.joblib"
        if not pipeline_path.exists():
            raise SystemExit(f"Pipeline artifact not found at {pipeline_path}")

        print(f"Loading data from {joined_path} ...")
        df = pd.read_parquet(joined_path)
        if "isFraud" not in df.columns:
            raise SystemExit("Column isFraud not found in dataset.")

        y = df["isFraud"].astype(int).values
        X_raw = df.drop(columns=["isFraud"])
        X_fe, _, _ = build_feature_matrix(X_raw)

        print(f"Loading pipeline from {pipeline_path} ...")
        pipeline = joblib.load(pipeline_path)

        print("Scoring full dataset...")
        y_scores = pipeline.predict_proba(X_fe)[:, 1]

        core = compute_core_metrics(y, y_scores)
        sweep = threshold_sweep(y, y_scores)

        print("Evaluation metrics:", core)

        fpr, tpr, _ = metrics.roc_curve(y, y_scores)
        precision, recall, _ = metrics.precision_recall_curve(y, y_scores)

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        roc_path = DOCS_DIR / "roc_curve.png"
        pr_path = DOCS_DIR / "pr_curve.png"

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {core['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("RiskScore99 ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_path)

        plt.figure()
        plt.plot(recall, precision, label=f"PR AUC = {core['pr_auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("RiskScore99 Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pr_path)

        latest.metrics_json.update(core)
        latest.metrics_json["threshold_sweep"] = sweep
        session.add(latest)
        session.commit()

        print(f"Saved ROC curve to {roc_path}")
        print(f"Saved PR curve to {pr_path}")

    finally:
        session.close()


if __name__ == "__main__":
    main()

