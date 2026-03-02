from app.agents.explainer import ExplanationAgent
from app.services.metrics_service import compute_core_metrics
import numpy as np


def test_risk_score_explanation_reason_codes():
    agent = ExplanationAgent()
    record = {
        "TransactionAmt": 5000.0,
        "P_emaildomain": "weird-domain.biz",
        "DeviceType": "",
        "DeviceInfo": "",
    }
    result = agent.explain(record, probability=0.9, risk_score=90)
    reasons = result["reason_codes"]
    assert any("HIGH_TRANSACTION_AMOUNT" in r for r in reasons)
    assert any("UNUSUAL_EMAIL_DOMAIN" in r for r in reasons)


def test_compute_core_metrics_shapes():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_core_metrics(y_true, y_scores)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

