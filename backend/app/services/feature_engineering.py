from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


IEEE_NUMERIC_FEATURES: List[str] = [
    "TransactionAmt",
    "TransactionDT",
]

IEEE_CATEGORICAL_FEATURES: List[str] = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]


DERIVED_FEATURES: List[str] = [
    "TransactionAmt_log",
    "TransactionDT_hour",
    "P_emaildomain_group",
]


def derive_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log amount with safe handling for zeros
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].astype(float).clip(lower=0))
    else:
        df["TransactionAmt_log"] = 0.0

    # Approximate hour-of-day from TransactionDT seconds offset
    if "TransactionDT" in df.columns:
        hours = (df["TransactionDT"].astype(float) / 3600.0) % 24
        df["TransactionDT_hour"] = hours.astype(int)
    else:
        df["TransactionDT_hour"] = 0

    # Email domain grouping (simple)
    def _group_email(domain: str | float | None) -> str:
        if isinstance(domain, float) and np.isnan(domain):
            return "missing"
        if domain is None:
            return "missing"
        dom = str(domain).lower()
        if "gmail" in dom:
            return "gmail"
        if "yahoo" in dom:
            return "yahoo"
        if "hotmail" in dom or "outlook" in dom or "live" in dom:
            return "microsoft"
        return "other"

    df["P_emaildomain_group"] = df.get("P_emaildomain", pd.Series([None] * len(df))).map(_group_email)

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Returns a DataFrame ready for a ColumnTransformer-based pipeline,
    along with explicit lists of numeric and categorical feature names.
    """
    df = derive_basic_features(df)

    numeric = IEEE_NUMERIC_FEATURES + ["TransactionAmt_log"]
    categorical = IEEE_CATEGORICAL_FEATURES + ["TransactionDT_hour", "P_emaildomain_group"]

    # Ensure all expected columns exist
    for col in numeric + categorical:
        if col not in df.columns:
            df[col] = np.nan

    return df[numeric + categorical], numeric, categorical

