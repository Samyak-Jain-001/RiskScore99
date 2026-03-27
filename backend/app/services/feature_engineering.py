from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


# ── Original MVP features ───────────────────────────────────────────
IEEE_NUMERIC_FEATURES_BASE: List[str] = [
    "TransactionAmt",
    "TransactionDT",
]

IEEE_CATEGORICAL_FEATURES_BASE: List[str] = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

# ── NEW: Vesta engineered V features (top 20 by typical importance) ──
# These are anonymous but highly predictive in the IEEE-CIS dataset.
# Selected based on published competition analyses + feature importance.
V_FEATURES: List[str] = [
    "V1", "V2", "V3", "V4", "V5",
    "V12", "V13", "V14", "V15", "V16",
    "V17", "V19", "V20", "V23", "V24",
    "V25", "V26", "V29", "V30", "V34",
]

# ── NEW: Count features (C1–C14) ────────────────────────────────────
C_FEATURES: List[str] = [f"C{i}" for i in range(1, 15)]

# ── NEW: D (timedelta) features – commonly useful subset ────────────
D_FEATURES: List[str] = ["D1", "D2", "D3", "D4", "D10", "D15"]


# Combine all numeric features
IEEE_NUMERIC_FEATURES: List[str] = (
    IEEE_NUMERIC_FEATURES_BASE + V_FEATURES + C_FEATURES + D_FEATURES
)

IEEE_CATEGORICAL_FEATURES: List[str] = IEEE_CATEGORICAL_FEATURES_BASE


DERIVED_FEATURES: List[str] = [
    "TransactionAmt_log",
    "TransactionAmt_decimal",
    "TransactionDT_hour",
    "TransactionDT_dayofweek",
    "P_emaildomain_group",
    "email_domain_match",
]


def derive_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Log amount (unchanged) ──────────────────────────────────────
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].astype(float).clip(lower=0))
    else:
        df["TransactionAmt_log"] = 0.0

    # ── NEW: Decimal part of amount (fraud signal — round amounts vs cents) ─
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_decimal"] = (
            df["TransactionAmt"].astype(float) % 1
        ).round(4)
    else:
        df["TransactionAmt_decimal"] = 0.0

    # ── FIXED: Hour-of-day with proper NaN handling ─────────────────
    if "TransactionDT" in df.columns:
        dt_series = pd.to_numeric(df["TransactionDT"], errors="coerce")
        hours = (dt_series / 3600.0) % 24
        # FIX: fillna before astype(int) to avoid crash on missing TransactionDT
        df["TransactionDT_hour"] = hours.fillna(-1).astype(int).replace(-1, np.nan)
    else:
        df["TransactionDT_hour"] = np.nan

    # ── NEW: Day of week (cyclical transaction patterns) ────────────
    if "TransactionDT" in df.columns:
        dt_series = pd.to_numeric(df["TransactionDT"], errors="coerce")
        days = (dt_series / 86400.0) % 7
        df["TransactionDT_dayofweek"] = days.fillna(-1).astype(int).replace(-1, np.nan)
    else:
        df["TransactionDT_dayofweek"] = np.nan

    # ── Email domain grouping (unchanged logic) ─────────────────────
    def _group_email(domain) -> str:
        if domain is None:
            return "missing"
        if isinstance(domain, float) and np.isnan(domain):
            return "missing"
        dom = str(domain).lower()
        if "gmail" in dom:
            return "gmail"
        if "yahoo" in dom:
            return "yahoo"
        if "hotmail" in dom or "outlook" in dom or "live" in dom:
            return "microsoft"
        return "other"

    df["P_emaildomain_group"] = (
        df["P_emaildomain"] if "P_emaildomain" in df.columns
        else pd.Series([None] * len(df))
    ).map(_group_email)

    # ── NEW: Email domain match (P vs R — mismatch is a fraud signal) ─
    def _email_match(row):
        p = row.get("P_emaildomain")
        r = row.get("R_emaildomain")
        if p is None or r is None:
            return "missing"
        if (isinstance(p, float) and np.isnan(p)) or (isinstance(r, float) and np.isnan(r)):
            return "missing"
        return "match" if str(p).lower() == str(r).lower() else "mismatch"

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = df.apply(_email_match, axis=1)
    else:
        df["email_domain_match"] = "missing"

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Returns a DataFrame ready for a ColumnTransformer-based pipeline,
    along with explicit lists of numeric and categorical feature names.
    """
    df = derive_basic_features(df)

    numeric = (
        IEEE_NUMERIC_FEATURES
        + ["TransactionAmt_log", "TransactionAmt_decimal"]
    )
    categorical = (
        IEEE_CATEGORICAL_FEATURES
        + ["TransactionDT_hour", "TransactionDT_dayofweek",
           "P_emaildomain_group", "email_domain_match"]
    )

    # Ensure all expected columns exist (graceful if V/C/D columns missing in API payloads)
    for col in numeric + categorical:
        if col not in df.columns:
            df[col] = np.nan

    return df[numeric + categorical], numeric, categorical
