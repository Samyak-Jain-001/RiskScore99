"""
Train baseline IEEE-CIS fraud model for RiskScore99.

- Loads joined parquet produced by preprocess_ieee_cis.py
- Builds ColumnTransformer + classifier + CalibratedClassifierCV
- Saves pipeline + metadata under data/artifacts
- Registers model in SQLite model_registry
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for path in (BACKEND_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.config import settings
from app.db.base import Base
from app.db.models import ModelRegistry
from app.db.session import SessionLocal, engine
from app.services.feature_engineering import build_feature_matrix
from app.services.metrics_service import compute_core_metrics, threshold_sweep


DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_ROOT = PROJECT_ROOT / "data" / "artifacts"


def time_aware_split(df: pd.DataFrame, frac: float = 0.8):
    df_sorted = df.sort_values("TransactionDT")
    split_idx = int(len(df_sorted) * frac)
    train = df_sorted.iloc[:split_idx]
    val = df_sorted.iloc[split_idx:]
    return train, val


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

        print(f"Loading {joined_path} ...")
        df = pd.read_parquet(joined_path)

        if "isFraud" not in df.columns:
            raise SystemExit("Column isFraud not found in dataset.")

        y = df["isFraud"].astype(int).values
        X_raw = df.drop(columns=["isFraud"])

        X_fe, numeric_features, categorical_features = build_feature_matrix(X_raw)

        train_df, val_df = time_aware_split(
            pd.concat([X_fe.reset_index(drop=True), pd.Series(y, name="isFraud")], axis=1)
        )
        y_train = train_df["isFraud"].values
        y_val = val_df["isFraud"].values
        X_train = train_df.drop(columns=["isFraud"])
        X_val = val_df.drop(columns=["isFraud"])

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),              # ← ADD THIS LINE
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # ← add sparse_output=False
            ]
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        if settings.model_type == "hgb":
            base_estimator = HistGradientBoostingClassifier()
        elif settings.model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            base_estimator = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        else:
            base_estimator = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=-1,
            )

        clf = CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=3)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        print("Training model with calibration...")
        pipeline.fit(X_train, y_train)

        print("Evaluating on validation split...")
        y_scores = pipeline.predict_proba(X_val)[:, 1]
        core_metrics = compute_core_metrics(y_val, y_scores)
        sweep = threshold_sweep(y_val, y_scores)

        model_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        artifact_dir = ARTIFACTS_ROOT / f"model_{model_version}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = artifact_dir / "pipeline.joblib"
        metadata_path = artifact_dir / "metadata.json"

        print(f"Saving pipeline to {pipeline_path} ...")
        joblib.dump(pipeline, pipeline_path)

        metadata = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "model_type": settings.model_type,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        metrics_json = dict(core_metrics)
        metrics_json["threshold_sweep"] = sweep

        registry = ModelRegistry(
            model_version=model_version,
            created_at=datetime.utcnow(),
            metrics_json=metrics_json,
            data_hash="",
            artifact_path=str(artifact_dir),
        )
        session.add(registry)
        session.commit()

        print(f"Registered model version {model_version}")
        print("Core metrics:", core_metrics)

    finally:
        session.close()


if __name__ == "__main__":
    main()

