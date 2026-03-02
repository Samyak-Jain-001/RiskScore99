Architecture Overview
=====================

RiskScore99 is a backend-only MVP implementing an IEEE-CIS-based fraud risk scoring system. It is organized into a FastAPI service, an agentic scoring pipeline, a model training/evaluation stack, and a lightweight SQLite-backed governance layer.

### High-Level Components

- **API Layer (FastAPI)**: Exposes REST endpoints for scoring, importing data, browsing transactions, recording reviews/outcomes, and inspecting metrics.
- **Agents**: Orchestrated classes that implement an “agentic workflow” for scoring and routing.
- **Services**: Reusable modules for feature engineering, model loading/scoring, data access, and metrics.
- **Database (SQLite + SQLAlchemy)**: Stores scored transactions, audit logs, outcomes, and model registry entries.
- **ML Pipeline (Scripts)**: Preprocessing, training, and evaluation scripts operating on IEEE-CIS CSVs.
- **Dockerized Runtime**: Container image that runs the FastAPI app and can mount data for training/evaluation.

### Agentic Workflow

Agents live under `backend/app/agents` and are wired together by the `/score_transaction` endpoint:

- `SignalCollectorAgent`: Validates and normalizes incoming transaction JSON; may enrich with simple derived features.
- `FeatureEngineeringAgent`: Converts cleaned input into model-ready feature vectors, including derived features such as log-amount and time-of-day.
- `RiskScorerAgent`: Uses the preprocessor + calibrated model artifacts to produce a fraud probability and corresponding 0–99 risk score.
- `ExplanationAgent`: Generates human-readable reason codes and explanation text for each transaction.
- `PolicyAgent`: Maps risk scores (and select business rules) into APPROVE / CHALLENGE / REVIEW / BLOCK decisions.
- `ActionAgent`: Persists the scored transaction, decision, and explanation into the database and writes audit logs.
- `FeedbackLearningAgent`: Records downstream outcomes and prepares a “retrain recommendation” stub.

### Data and Models

- **Dataset**: IEEE-CIS Fraud Detection (`train_transaction.csv` and `train_identity.csv`), joined on `TransactionID`.
- **Preprocessing**: Pandas-based join and cleaning, followed by scikit-learn `ColumnTransformer` for numeric and categorical features.
- **Models**: Baseline `LogisticRegression` or `HistGradientBoostingClassifier`, with optional LightGBM if available.
- **Calibration**: `CalibratedClassifierCV` to produce well-calibrated probabilities.
- **Artifacts**: Stored under `data/artifacts` (preprocessor, model, metadata, optionally calibration wrapper).
- **Model Registry**: SQLite table tracking `model_version`, metrics summary, data hash, and artifact path.

### Governance and Observability

- **Audit Logs**: Every scoring and outcome update emits an entry into `audit_logs`.
- **Outcomes**: Reviewer and downstream labels (e.g., `confirmed_fraud`, `legit`, `chargeback`) are tracked for later analysis.
- **Metrics**: Separate service and script compute ROC-AUC, PR-AUC, confusion matrices, and cost curves; the `/metrics` endpoint surfaces summary metrics and operational stats.
- **Config**: `backend/app/config.py` centralizes paths, thresholds, cost parameters, and model selection; environment variables can override defaults.

See `docs/evaluation.md` and `docs/risk_governance.md` for more detail on evaluation and responsible AI considerations.

