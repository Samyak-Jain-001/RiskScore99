Download IEEE-CIS Fraud Detection Dataset
=========================================

RiskScore99 uses the IEEE-CIS Fraud Detection (Vesta) dataset from Kaggle. You must download it yourself using the Kaggle API and place it under `data/external`.

### Prerequisites

- Kaggle account
- Kaggle API token configured locally (typically `~/.kaggle/kaggle.json`)
- `kaggle` Python package or the Kaggle CLI installed

### Steps

1. Install Kaggle CLI if needed:

```bash
pip install kaggle
```

2. Authenticate with Kaggle by placing your API token at `~/.kaggle/kaggle.json` with proper permissions.

3. From the `riskScore99` project root, create the external data directory (if it does not already exist):

```bash
mkdir -p data/external
cd data/external
```

4. Download the IEEE-CIS competition data:

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip
```

You should now have at least:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

5. The preprocessing script `scripts/preprocess_ieee_cis.py` expects these files under `data/external`. If they are missing, it will raise a clear error message with instructions.

