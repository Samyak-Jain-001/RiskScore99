from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import AuditLog, ModelRegistry, Outcome, TransactionScored


def create_scored_transaction(
    db: Session,
    raw_json: Dict[str, Any],
    probability: float,
    score: int,
    decision: str,
    reason_codes: List[str],
    model_version: str | None,
) -> TransactionScored:
    tx = TransactionScored(
        transaction_id=raw_json.get("TransactionID"),
        timestamp_scored=datetime.utcnow(),
        raw_json=raw_json,
        score=score,
        probability=probability,
        decision=decision,
        reason_codes_json=reason_codes,
        model_version=model_version,
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx


def add_audit_log(db: Session, event_type: str, payload: Dict[str, Any], actor: str = "system") -> AuditLog:
    log = AuditLog(event_type=event_type, actor=actor, payload_json=payload)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def list_transactions(
    db: Session,
    decision: Optional[str] = None,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[TransactionScored]:
    q = db.query(TransactionScored)
    if decision:
        q = q.filter(TransactionScored.decision == decision)
    if min_score is not None:
        q = q.filter(TransactionScored.score >= min_score)
    if max_score is not None:
        q = q.filter(TransactionScored.score <= max_score)
    return q.order_by(TransactionScored.timestamp_scored.desc()).offset(offset).limit(limit).all()


def get_transaction(db: Session, tx_id: int) -> Optional[TransactionScored]:
    return db.query(TransactionScored).filter(TransactionScored.id == tx_id).first()


def update_reviewer_action(
    db: Session, tx: TransactionScored, action: str, notes: Optional[str]
) -> TransactionScored:
    tx.reviewer_action = action
    tx.reviewer_notes = notes
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx


def record_outcome(
    db: Session, tx: TransactionScored, outcome_label: str, notes: Optional[str]
) -> Outcome:
    outcome = Outcome(transaction_scored_id=tx.id, outcome_label=outcome_label, notes=notes)
    db.add(outcome)
    db.commit()
    db.refresh(outcome)
    return outcome


def get_operational_stats(db: Session) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    # Counts by decision
    decision_counts = (
        db.query(TransactionScored.decision, func.count(TransactionScored.id))
        .group_by(TransactionScored.decision)
        .all()
    )
    result["decision_counts"] = {d: int(c) for d, c in decision_counts}

    # Simple histogram of scores (bins of 10)
    bins: Dict[str, int] = {}
    for bucket, count in (
        db.query((TransactionScored.score / 10).label("bucket"), func.count(TransactionScored.id))
        .group_by("bucket")
        .all()
    ):
        label = f"{int(bucket)*10}-{int(bucket)*10+9}"
        bins[label] = int(count)
    result["score_histogram"] = bins

    return result

