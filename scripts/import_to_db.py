"""
CLI helper to import a CSV of transactions into RiskScore99 via the /import_csv API.

Example:
  python scripts/import_to_db.py --path data/external/train_transaction.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to CSV file on server filesystem")
    parser.add_argument(
        "--limit", type=int, default=1000, help="Max rows to import for demo (default 1000)"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL of running RiskScore99 API"
    )
    args = parser.parse_args()

    csv_path = Path(args.path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found at {csv_path}")

    payload = {"path": str(csv_path), "limit": args.limit}
    url = f"{args.base_url.rstrip('/')}/import_csv"
    print(f"POST {url} with payload {payload}")

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        print(resp.json())


if __name__ == "__main__":
    main()

