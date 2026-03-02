IEEE-CIS Dataset Notes
======================

### How to Obtain the Dataset

- The dataset comes from the IEEE-CIS Fraud Detection competition on Kaggle.
- You need a Kaggle account and API credentials.
- Follow `scripts/download_kaggle_data.md` for detailed steps using `kaggle competitions download -c ieee-fraud-detection`.
- After download, place the extracted CSV files under `data/external`:
  - `train_transaction.csv`
  - `train_identity.csv`
  - `test_transaction.csv`
  - `test_identity.csv`

### Expected File Locations

- Raw Kaggle CSVs:
  - `data/external/train_transaction.csv`
  - `data/external/train_identity.csv`
  - `data/external/test_transaction.csv`
  - `data/external/test_identity.csv`
- Processed/joined training data (created by `scripts/preprocess_ieee_cis.py`):
  - `data/processed/ieee_train_joined.parquet`

### Schema and Join Details

- `train_transaction.csv`:
  - Contains `TransactionID` (identifier), label `isFraud`, and many transaction features:
    - Numeric: `TransactionAmt`, `TransactionDT`, `dist1`, `dist2`, `C1`..`C14`, `D1`..`D15`, `V1`..`V339`, etc.
    - Categorical: `ProductCD`, `card1`..`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, etc.
- `train_identity.csv`:
  - Contains identity-related features for a subset of transactions, joined via `TransactionID`.
  - Examples: `id_01`..`id_38`, `DeviceType`, `DeviceInfo`, and additional anonymized fields.

Join strategy:

- Left join `train_transaction` with `train_identity` on `TransactionID`.
- Keep all rows from `train_transaction` (primary label source).
- Allow `NaN` values where identity information is missing (common for many rows).

### Known Challenges

- **High Missingness**:
  - Many `identity` features are missing (identity data only for a subset of transactions).
  - Some categorical fields (e.g., `M1`..`M9`) are sparsely populated.
  - The pipeline handles this via imputers in the scikit-learn `ColumnTransformer`.

- **Anonymized Features**:
  - Most features (`V*`, `id_*`, etc.) are anonymized, so their semantic meaning is unknown.
  - This limits domain-specific feature engineering and interpretability.
  - RiskScore99 focuses on a interpretable subset plus generic transforms (e.g., log amount, time-of-day, email groups).

- **Class Imbalance**:
  - Fraud is relatively rare; the `isFraud` label is highly imbalanced.
  - Metrics like PR-AUC and threshold trade-offs are important; simply optimizing accuracy is misleading.

- **Feature Drift Risk**:
  - As a real-world system, these features may drift over time (e.g., new device types or email domains, change in amount distributions).
  - The design includes model versioning, calibration, and monitoring hooks for score distribution drift.

- **Privacy Considerations**:
  - Although the competition dataset is anonymized, real-world deployment would require strict handling of PII and device/identity signals.
  - RiskScore99 documentation emphasizes data minimization, limited retention, and clear governance controls.

