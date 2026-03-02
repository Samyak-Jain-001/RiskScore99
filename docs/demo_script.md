RiskScore99 Demo Script (5 Minutes)
===================================

This script outlines a 5-minute live demo of the RiskScore99 MVP.

### 1) Train the Model (Offline Step)

- Show the preprocessing and training commands:

```bash
cd riskScore99
python scripts/preprocess_ieee_cis.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

- Briefly summarize:
  - IEEE-CIS preprocessing: joining `train_transaction` + `train_identity` on `TransactionID`.
  - Model: scikit-learn pipeline with calibration and a 0–99 risk score.
  - Evaluation: ROC-AUC, PR-AUC, threshold sweep, and cost analysis.

### 2) Run the API

Start the FastAPI backend:

```bash
cd backend
uvicorn app.main:app --reload
```

or with Docker:

```bash
cd riskScore99
docker build -t riskscore99-backend . -f backend/Dockerfile
docker run -p 8000:8000 -v $(pwd)/data:/app/data riskscore99-backend
```

Navigate to `http://localhost:8000/docs` and show the OpenAPI UI.

### 3) Score a “Legit-like” Transaction (Low Score → APPROVE)

- Use the `/score_transaction` endpoint with a relatively small amount and common email domain:

```json
{
  "TransactionID": 123456789,
  "TransactionAmt": 25.50,
  "TransactionDT": 86400,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com",
  "DeviceType": "desktop"
}
```

- Highlight the response:
  - `fraud_probability` should be relatively low.
  - `risk_score_0_99` should fall below the APPROVE threshold.
  - `decision` = `APPROVE`.
  - `reason_codes` and `explanation_text` provide human-readable context.

### 4) Score a “Fraud-like” Transaction (High Score → REVIEW/BLOCK)

- Change inputs to simulate risk:
  - Very high `TransactionAmt`.
  - Unusual email domain.
  - Missing identity/device fields.

Example payload:

```json
{
  "TransactionID": 987654321,
  "TransactionAmt": 5000.00,
  "TransactionDT": 172800,
  "ProductCD": "C",
  "card4": "discover",
  "card6": "credit",
  "P_emaildomain": "suspicious-domain.biz",
  "R_emaildomain": "suspicious-domain.biz",
  "DeviceType": "",
  "DeviceInfo": ""
}
```

- Show the response:
  - Higher `fraud_probability` and `risk_score_0_99`.
  - `decision` likely `REVIEW` or `BLOCK` depending on thresholds.
  - Reason codes such as `HIGH_TRANSACTION_AMOUNT`, `UNUSUAL_EMAIL_DOMAIN`, `DEVICE_RISK_SIGNAL`.

### 5) Show Reason Codes and Audit Log

- Call `/transactions` to list recently scored transactions.
- Call `/transactions/{id}` for the high-risk example:
  - Show stored `reason_codes`, `decision`, `model_version`, and `raw_json`.
  - Mention that an audit log entry was created for the scoring event.

### 6) Override with Reviewer Action

- Use `/transactions/{id}/review_action`:

```json
{
  "action": "review",
  "notes": "Manual KYC requested due to unusual domain and high amount."
}
```

- Re-fetch `/transactions/{id}` to show `reviewer_action` and `reviewer_notes` now populated.

### 7) Record Outcome

- Once you “know” the ground truth, use `/outcomes/{id}`:

```json
{
  "outcome_label": "confirmed_fraud",
  "notes": "Chargeback filed by issuer."
}
```

- Highlight that outcomes are stored and associated with the scored transaction and logs.

### 8) Show Metrics Endpoint

- Call `/metrics`:
  - Display `roc_auc` and `pr_auc`.
  - Show threshold sweep snippet and operational stats (decision counts, score histogram).
  - Connect back to how thresholds T1/T2/T3 map probability → score → decision.

Conclude by tying the demo back to rubric criteria:

- Clear problem framing (fraud risk scoring).
- Realistic dataset and pipeline.
- Agentic workflow from signal collection to action.
- Quantitative evaluation and governance / auditability.

