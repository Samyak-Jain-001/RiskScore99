"""
Preprocess IEEE-CIS Fraud Detection data for RiskScore99.

- Joins train_transaction.csv with train_identity.csv on TransactionID (left join).
- Handles high missingness by leaving NaNs (model pipeline will impute).
- Writes a processed parquet file under data/processed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    train_tx_path = DATA_EXTERNAL / "train_transaction.csv"
    train_id_path = DATA_EXTERNAL / "train_identity.csv"

    if not train_tx_path.exists() or not train_id_path.exists():
        raise SystemExit(
            f"Missing IEEE-CIS CSVs.\n"
            f"Expected at:\n- {train_tx_path}\n- {train_id_path}\n"
            "Download them with Kaggle and place under data/external "
            "(see scripts/download_kaggle_data.md and docs/dataset_notes.md)."
        )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Loading {train_tx_path} ...")
    df_tx = pd.read_csv(train_tx_path)
    print(f"Loading {train_id_path} ...")
    df_id = pd.read_csv(train_id_path)

    print("Joining on TransactionID (left join) ...")
    df = df_tx.merge(df_id, on="TransactionID", how="left", suffixes=("", "_id"))

    out_path = DATA_PROCESSED / "ieee_train_joined.parquet"
    print(f"Writing processed dataset to {out_path} ...")
    df.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

