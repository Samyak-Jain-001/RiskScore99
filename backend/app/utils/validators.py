from typing import Dict, List


def normalize_email_domain(domain: str | None) -> str | None:
    if domain is None:
        return None
    domain = domain.strip().lower()
    if domain == "":
        return None
    # Simple normalization for common webmail providers
    mapping = {
        "gmail.com": "gmail.com",
        "googlemail.com": "gmail.com",
        "yahoo.com": "yahoo.com",
        "ymail.com": "yahoo.com",
    }
    return mapping.get(domain, domain)


def validate_transaction_payload(payload: Dict) -> List[str]:
    """Lightweight validation beyond Pydantic, returning warnings."""
    warnings: List[str] = []
    amt = payload.get("TransactionAmt")
    if amt is not None and amt == 0:
        warnings.append("ZERO_AMOUNT")
    if payload.get("TransactionDT") is None:
        warnings.append("MISSING_TRANSACTION_TIME")
    return warnings

