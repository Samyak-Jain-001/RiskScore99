from typing import Any, Dict, List


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
        "rocketmail.com": "yahoo.com",
        "hotmail.com": "hotmail.com",
        "outlook.com": "outlook.com",
        "live.com": "live.com",
        "msn.com": "hotmail.com",
    }
    return mapping.get(domain, domain)


def validate_transaction_payload(payload: Dict[str, Any]) -> List[str]:
    """Lightweight validation beyond Pydantic, returning warnings."""
    warnings: List[str] = []

    amt = payload.get("TransactionAmt")
    if amt is not None:
        if amt == 0:
            warnings.append("ZERO_AMOUNT")
        elif amt < 0:
            warnings.append("NEGATIVE_AMOUNT")
        elif amt > 10000:
            warnings.append("VERY_HIGH_AMOUNT")

    if payload.get("TransactionDT") is None:
        warnings.append("MISSING_TRANSACTION_TIME")

    # Card type checks
    if payload.get("card4") is None and payload.get("card6") is None:
        warnings.append("MISSING_CARD_INFO")

    # Product code check
    if payload.get("ProductCD") is None:
        warnings.append("MISSING_PRODUCT_CODE")

    return warnings
