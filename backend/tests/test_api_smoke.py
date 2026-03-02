from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "RiskScore99" in resp.json()["message"]


def test_score_transaction_without_model_returns_503():
    payload = {
        "TransactionAmt": 10.0,
        "TransactionDT": 3600,
    }
    resp = client.post("/score_transaction", json=payload)
    # Until a model is trained and registered, expect service-unavailable
    assert resp.status_code in (400, 503)

