from app.db.base import Base
from app.db.models import AuditLog, TransactionScored
from app.db.session import SessionLocal, engine
from app.services.data_service import add_audit_log, create_scored_transaction


def setup_module(module):
    Base.metadata.create_all(bind=engine)


def test_create_scored_transaction_and_audit_log():
    db = SessionLocal()
    try:
        tx = create_scored_transaction(
            db,
            raw_json={"TransactionID": 1, "TransactionAmt": 100.0},
            probability=0.5,
            score=50,
            decision="APPROVE",
            reason_codes=["TEST_REASON"],
            model_version="test-version",
        )
        assert isinstance(tx.id, int)

        log = add_audit_log(
            db,
            event_type="unit_test_event",
            payload={"transaction_scored_id": tx.id},
        )
        assert isinstance(log.id, int)
    finally:
        db.close()

