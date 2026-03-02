"""
Simulate a streaming feed of transactions scored via the RiskScore99 API.

Reads a CSV and sends each row to /score_transaction with a small delay.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to CSV of transactions")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL of running RiskScore99 API"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5, help="Seconds to sleep between requests"
    )
    args = parser.parse_args()

    csv_path = Path(args.path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    url = f"{args.base_url.rstrip('/')}/score_transaction"
    with httpx.Client(timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = row.to_dict()
            resp = client.post(url, json=payload)
            print(resp.status_code, resp.json())
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()

