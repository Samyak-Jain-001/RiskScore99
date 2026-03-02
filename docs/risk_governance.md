Risk Governance and Responsible AI
==================================

RiskScore99 is an educational MVP inspired by production fraud systems. This document discusses major risks, controls, and responsible AI considerations.

### Key Risks

- **False Declines (False Positives)**:
  - Genuine customers may be blocked or challenged unnecessarily.
  - Impacts customer experience, revenue, and brand trust.

- **Missed Fraud (False Negatives)**:
  - Fraudulent transactions that slip through lead to direct financial losses.
  - Overly conservative thresholds may preserve experience but underperform on fraud prevention.

- **Bias and Fairness**:
  - Even anonymized features can proxy for sensitive attributes (e.g., geography via IP/device/email).
  - Thresholds or features that disproportionately impact certain groups can introduce unfair outcomes.

- **Privacy and Data Protection**:
  - Identity and device signals may be sensitive in real-world deployments.
  - Over-collection or long retention increases privacy risk.

- **Adversarial Adaptation**:
  - Fraudsters actively adapt to patterns, attempting to evade detection.
  - Static models without monitoring or retraining become less effective.

### Controls and Governance

- **Human-in-the-Loop**:
  - The policy distinguishes between APPROVE / CHALLENGE / REVIEW / BLOCK.
  - CHALLENGE and REVIEW provide hooks for manual or step-up verification (e.g., KYC, OTP).
  - Reviewer actions are recorded via `/transactions/{id}/review_action`.

- **Audit Logs**:
  - All scoring and outcome updates write structured entries to `audit_logs`.
  - Logs include decision, risk score, model version, and reason codes.
  - Enables post-hoc analysis, incident review, and regulatory reporting.

- **Model Versioning and Rollback**:
  - Each trained model has a `model_version` and `artifact_path` recorded in `model_registry`.
  - Metrics summary (ROC-AUC, PR-AUC, threshold sweep) is stored per version.
  - Rolling back to a prior version is as simple as changing configuration to load an earlier registry entry.

- **Monitoring and Drift Detection (Conceptual)**:
  - Score distributions and decision rates are exposed via `/metrics`.
  - Future extensions can include:
    - Population drift checks for key features (e.g., TransactionAmt, email domains, device types).
    - Calibration drift monitoring (e.g., observed fraud rates per score band).

### Data Minimization and Retention

- The IEEE-CIS dataset is anonymized; in real production:
  - Only collect features that materially improve fraud detection.
  - Avoid storing raw PII where possible; prefer hashed or categorical transforms.
  - Define and enforce retention windows (e.g., limited historical window for training and monitoring).
  - Ensure access controls around transaction, identity, and outcome data.

- In RiskScore99:
  - The `transactions_scored` table stores the raw JSON payload and derived scores.
  - This is justified for educational purposes and model debugging; in production, this would be carefully scoped.

### Explainability

- Limitations:
  - Many IEEE-CIS features are anonymized; we cannot provide user-facing narratives such as “IP mismatch with billing address.”
  - The model may rely on complex interactions that are not easily translated into plain language.

- Reason Codes:
  - The `ExplanationAgent` provides 3–5 high-level reason codes per transaction:
    - `HIGH_TRANSACTION_AMOUNT` / `MEDIUM_TRANSACTION_AMOUNT`
    - `UNUSUAL_EMAIL_DOMAIN`
    - `DEVICE_RISK_SIGNAL`
    - `IDENTITY_MISMATCH_SIGNAL`
    - `MODEL_TOP_FEATURES` / `MODEL_BASELINE_RISK`
  - These codes are not perfect explanations but provide a coarse view of why a transaction was considered risky.

- Transparency:
  - For internal reviewers, reason codes, scores, and metrics are available via API.
  - For end-users, any production deployment should pair reason codes with clear messaging and dispute/appeal processes.

### Bias and Fairness Checks (MVP)

- As an MVP, RiskScore99 does not perform full fairness audits.
- However, the design encourages:
  - Segment-level analysis via offline notebooks or scripts (e.g., performance by geography proxy features).
  - Reviewing decision rates and false positive rates across cohorts.
  - Treating thresholds and policy rules as adjustable levers, not fixed law.

### Summary

RiskScore99 demonstrates how an AI-driven fraud system might be structured with governance in mind:

- Explicit decision policies and thresholds.
- Model versioning and audit logging for traceability.
- Explainability via reason codes despite anonymized features.
- Hooks for monitoring, bias analysis, and retraining decisions.

The project is intentionally scoped as an educational MVP; any real-world deployment would require deeper privacy, security, compliance, and fairness reviews before going live.

