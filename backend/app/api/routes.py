from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.agents.action import ActionAgent
from app.agents.explainer import ExplanationAgent
from app.agents.feedback import FeedbackLearningAgent
from app.agents.policy import PolicyAgent
from app.agents.risk_scorer import RiskScorerAgent
from app.agents.signal_collector import SignalCollectorAgent
from app.db.models import Outcome, TransactionScored
from app.db.session import get_db
from app.services import data_service
from app.services.metrics_service import compute_core_metrics, threshold_sweep
from app.services.model_service import ModelNotTrainedError
from app.utils.schema import (
    ImportCsvRequest,
    ImportSummary,
    MetricsResponse,
    OutcomeRequest,
    ReviewActionRequest,
    ScoreResponse,
    TransactionDetail,
    TransactionInput,
    TransactionSummary,
)
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get("/", tags=["system"])
async def root() -> dict:
    return {"message": "RiskScore99 API – ready"}


@router.post("/score_transaction", response_model=ScoreResponse, tags=["scoring"])
async def score_transaction(
    payload: TransactionInput,
    db: Session = Depends(get_db),
) -> ScoreResponse:
    collector = SignalCollectorAgent()
    scorer = RiskScorerAgent(db=db)
    explainer = ExplanationAgent()
    policy = PolicyAgent()
    action_agent = ActionAgent(db=db)

    record, warnings = collector.collect(payload.model_dump())

    try:
        score_result = scorer.score(record)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    probability = score_result["probability"]
    risk_score = score_result["risk_score"]
    model_version = score_result["model_version"]

    explanation = explainer.explain(record, probability=probability, risk_score=risk_score)
    reason_codes = explanation["reason_codes"]
    explanation_text = explanation["explanation_text"]

    decision = policy.decide(record, probability=probability, risk_score=risk_score)

    tx = action_agent.persist(
        raw_record=record,
        probability=probability,
        risk_score=risk_score,
        decision=decision,
        reason_codes=reason_codes + warnings,
        model_version=model_version,
    )

    return ScoreResponse(
        transaction_id=record.get("TransactionID"),
        fraud_probability=probability,
        risk_score_0_99=risk_score,
        decision=decision,
        reason_codes=reason_codes + warnings,
        explanation_text=explanation_text,
        model_version=model_version,
    )


@router.post("/import_csv", response_model=ImportSummary, tags=["ingest"])
async def import_csv(
    req: ImportCsvRequest,
    db: Session = Depends(get_db),
) -> ImportSummary:
    import pandas as pd

    try:
        df = pd.read_csv(req.path, nrows=req.limit)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"CSV file not found at {req.path}")

    collector = SignalCollectorAgent()
    scorer = RiskScorerAgent(db=db)
    explainer = ExplanationAgent()
    policy = PolicyAgent()
    action_agent = ActionAgent(db=db)

    rows_processed = 0
    rows_scored = 0
    errors: List[str] = []

    for _, row in df.iterrows():
        rows_processed += 1
        record = row.to_dict()
        try:
            cleaned, warnings = collector.collect(record)
            score_result = scorer.score(cleaned)
            probability = score_result["probability"]
            risk_score = score_result["risk_score"]
            model_version = score_result["model_version"]
            explanation = explainer.explain(cleaned, probability, risk_score)
            decision = policy.decide(cleaned, probability, risk_score)
            action_agent.persist(
                raw_record=cleaned,
                probability=probability,
                risk_score=risk_score,
                decision=decision,
                reason_codes=explanation["reason_codes"] + warnings,
                model_version=model_version,
            )
            rows_scored += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    return ImportSummary(rows_processed=rows_processed, rows_scored=rows_scored, errors=errors)


@router.get("/transactions", response_model=List[TransactionSummary], tags=["transactions"])
async def list_transactions_endpoint(
    decision: Optional[str] = Query(default=None),
    min_score: Optional[int] = Query(default=None, ge=0, le=99),
    max_score: Optional[int] = Query(default=None, ge=0, le=99),
    limit: int = Query(default=100, gt=0, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> List[TransactionSummary]:
    items = data_service.list_transactions(
        db, decision=decision, min_score=min_score, max_score=max_score, limit=limit, offset=offset
    )
    return [
        TransactionSummary(
            id=it.id,
            transaction_id=it.transaction_id,
            timestamp_scored=it.timestamp_scored,
            score=it.score,
            probability=it.probability,
            decision=it.decision,
            model_version=it.model_version,
        )
        for it in items
    ]


@router.get("/transactions/{tx_id}", response_model=TransactionDetail, tags=["transactions"])
async def get_transaction_endpoint(
    tx_id: int,
    db: Session = Depends(get_db),
) -> TransactionDetail:
    tx = data_service.get_transaction(db, tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")
    outcomes = [
        {
            "id": o.id,
            "outcome_label": o.outcome_label,
            "notes": o.notes,
            "created_at": o.created_at,
        }
        for o in tx.outcomes
    ]
    return TransactionDetail(
        id=tx.id,
        transaction_id=tx.transaction_id,
        timestamp_scored=tx.timestamp_scored,
        score=tx.score,
        probability=tx.probability,
        decision=tx.decision,
        model_version=tx.model_version,
        raw_json=tx.raw_json,
        reason_codes=tx.reason_codes_json,
        reviewer_action=tx.reviewer_action,
        reviewer_notes=tx.reviewer_notes,
        outcomes=outcomes,
    )


@router.post("/transactions/{tx_id}/review_action", response_model=TransactionDetail, tags=["transactions"])
async def review_action_endpoint(
    tx_id: int,
    req: ReviewActionRequest,
    db: Session = Depends(get_db),
) -> TransactionDetail:
    tx = data_service.get_transaction(db, tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    tx = data_service.update_reviewer_action(db, tx, action=req.action, notes=req.notes)
    data_service.add_audit_log(
        db,
        event_type="review_action",
        payload={
            "transaction_scored_id": tx.id,
            "action": req.action,
            "notes": req.notes,
        },
        actor="reviewer",
    )

    outcomes = [
        {
            "id": o.id,
            "outcome_label": o.outcome_label,
            "notes": o.notes,
            "created_at": o.created_at,
        }
        for o in tx.outcomes
    ]

    return TransactionDetail(
        id=tx.id,
        transaction_id=tx.transaction_id,
        timestamp_scored=tx.timestamp_scored,
        score=tx.score,
        probability=tx.probability,
        decision=tx.decision,
        model_version=tx.model_version,
        raw_json=tx.raw_json,
        reason_codes=tx.reason_codes_json,
        reviewer_action=tx.reviewer_action,
        reviewer_notes=tx.reviewer_notes,
        outcomes=outcomes,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["metrics"])
async def metrics_endpoint(db: Session = Depends(get_db)) -> MetricsResponse:
    from app.db.models import ModelRegistry

    latest = (
        db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).first()
    )
    operational = data_service.get_operational_stats(db)

    metrics_data: dict = {}
    if latest and latest.metrics_json:
        metrics_data = latest.metrics_json

    return MetricsResponse(
        model_version=latest.model_version if latest else None,
        roc_auc=metrics_data.get("roc_auc"),
        pr_auc=metrics_data.get("pr_auc"),
        threshold_metrics=metrics_data.get("threshold_sweep", {}),
        operational_stats=operational,
    )


@router.post("/outcomes/{tx_id}", tags=["feedback"])
async def record_outcome_endpoint(
    tx_id: int,
    req: OutcomeRequest,
    db: Session = Depends(get_db),
) -> dict:
    tx = data_service.get_transaction(db, tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    agent = FeedbackLearningAgent(db=db)
    outcome = agent.record_outcome(tx, outcome_label=req.outcome_label, notes=req.notes)

    return {
        "id": outcome.id,
        "transaction_scored_id": tx.id,
        "outcome_label": outcome.outcome_label,
        "notes": outcome.notes,
        "created_at": outcome.created_at,
    }

