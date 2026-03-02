RiskScore99 – Transaction Fraud Risk Scoring MVP
================================================

RiskScore99 is an MVP backend inspired by Visa-style AI transaction risk scoring, built on the IEEE-CIS Fraud Detection (Vesta) dataset. It exposes a FastAPI-based REST API and CLI-style scripts for preprocessing, training, evaluation, and scoring – with no frontend – and is designed to be demoable and rubric-friendly for a university project.

### Key Features

- FastAPI backend with clean OpenAPI docs.
- IEEE-CIS-specific preprocessing and model training pipeline.
- Calibrated fraud probabilities mapped to a 0–99 risk score.
- Agentic workflow for scoring and routing (APPROVE / CHALLENGE / REVIEW / BLOCK).
- SQLite persistence, audit logs, and model registry.
- Evaluation scripts with ROC-AUC, PR-AUC, threshold sweeps, and cost analysis.
- Dockerized backend that can start without local data.

### Quick Start (Local)

1. Create and activate a virtual environment (Python 3.10+ recommended).
2. Install backend requirements:

```bash
cd riskScore99/backend
pip install -r requirements.txt
```

3. Download the IEEE-CIS dataset from Kaggle into `data/external` (see `scripts/download_kaggle_data.md` and `docs/dataset_notes.md` for details).
4. Run preprocessing, training, and evaluation (once the scripts are implemented):

```bash
cd ..
python scripts/preprocess_ieee_cis.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

5. Start the API:

```bash
cd backend
uvicorn app.main:app --reload
```

6. Visit the interactive docs at `http://localhost:8000/docs`.

### Quick Start (Docker)

From the project root (`riskScore99` directory):

```bash
docker build -t riskscore99-backend . -f backend/Dockerfile
docker run -p 8000:8000 -v $(pwd)/data:/app/data riskscore99-backend
```

Then:

- Access API docs at `http://localhost:8000/docs`.
- Use `docker exec` (or a separate `docker run`) to call:

```bash
docker exec -it <container_id> python /app/scripts/preprocess_ieee_cis.py
docker exec -it <container_id> python /app/scripts/train_model.py
docker exec -it <container_id> python /app/scripts/evaluate_model.py
```

### Example cURL Commands

Score a transaction:

```bash
curl -X POST "http://localhost:8000/score_transaction" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 123456789,
    "TransactionAmt": 50.25,
    "TransactionDT": 86400,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "DeviceType": "desktop"
  }'
```

List scored transactions:

```bash
curl "http://localhost:8000/transactions?decision=APPROVE&min_score=0&max_score=30&limit=20"
```

Record a reviewer action:

```bash
curl -X POST "http://localhost:8000/transactions/1/review_action" \
  -H "Content-Type: application/json" \
  -d '{"action": "review", "notes": "Manual KYC requested"}'
```

Record an outcome:

```bash
curl -X POST "http://localhost:8000/outcomes/1" \
  -H "Content-Type: application/json" \
  -d '{"outcome_label": "confirmed_fraud", "notes": "Chargeback filed"}'
```

### Project Layout

The repo is structured as follows:

- `backend/` – FastAPI app, agents, services, DB models, tests, and Dockerfile.
- `data/` – raw/processed data and trained artifacts (not committed to git).
- `scripts/` – CLI-style Python scripts for preprocessing, training, evaluation, import, and streaming/CLI scoring.
- `docs/` – architecture notes, evaluation details, risk governance and demo script, dataset documentation.

See `docs/architecture.md` and `docs/demo_script.md` for more detail once those files are populated.

