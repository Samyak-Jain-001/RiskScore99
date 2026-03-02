"""
Score a single hard-coded sample transaction against a running RiskScore99 API.
"""

from __future__ import annotations

import httpx


def main() -> None:
    payload = {
        "TransactionID": 999999999,
        "TransactionAmt": 42.50,
        "TransactionDT": 86400,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "desktop",
    }
    url = "http://localhost:8000/score_transaction"
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, json=payload)
        print(resp.status_code, resp.json())


if __name__ == "__main__":
    main()

