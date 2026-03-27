from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.agents.action import ActionAgent
from app.agents.explainer import ExplanationAgent
from app.agents.explainer import ReasoningTraceAgent
from app.agents.feedback import FeedbackLearningAgent
from app.agents.policy import PolicyAgent
from app.agents.risk_scorer import RiskScorerAgent
from app.agents.signal_collector import SignalCollectorAgent
from app.config import settings
from app.db.models import Outcome, TransactionScored
from app.db.session import get_db
from app.services import data_service
from app.services.metrics_service import compute_core_metrics, threshold_sweep
from app.services.model_service import (
    ModelNotTrainedError,
    get_model_registry,
    rollback_model,
)
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


# ═══════════════════════════════════════════════════════════════════════
#  SYSTEM
# ═══════════════════════════════════════════════════════════════════════

@router.get("/", tags=["system"])
async def root() -> dict:
    return {"message": "RiskScore99 API – ready"}


# ═══════════════════════════════════════════════════════════════════════
#  CORE SCORING — AGENTIC ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
#  FILE 1: Changes to backend/app/utils/schema.py
#  Add these two fields to your ScoreResponse class:
# ═══════════════════════════════════════════════════════════════════

# In class ScoreResponse(BaseModel):
#   ... existing fields ...
#   explanation_source: Optional[str] = "heuristic"   # "llm" or "heuristic"
#   reasoning_trace: Optional[str] = None              # step-by-step agent narrative


# ═══════════════════════════════════════════════════════════════════
#  FILE 2: Full replacement for the score_transaction function
#  in backend/app/api/routes.py
#
#  Also add this import at the top of routes.py:
#    from app.agents.explainer import ExplanationAgent, ReasoningTraceAgent
# ═══════════════════════════════════════════════════════════════════

@router.post("/score_transaction", response_model=ScoreResponse, tags=["scoring"])
async def score_transaction(
    payload: TransactionInput,
    db: Session = Depends(get_db),
) -> ScoreResponse:
    """
    Agentic transaction scoring pipeline with LLM-powered explanations.

    Flow:
      1. SignalCollector.collect()         → normalize + validate
      2. RiskScorer.select_model()         → dynamic model selection
      3. RiskScorer.score()                → ML probability + risk score
      4. IF ambiguous/contradictory score:
         a. SignalCollector.deep_collect()  → enrich with additional signals
         b. RiskScorer.rescore()            → re-evaluate with deep signals
      5. PolicyAgent.decide_with_context()  → decision + escalation flag
      6. ExplanationAgent (LLM or heuristic) → explanation text
      7. ReasoningTraceAgent (LLM or template) → agent workflow narrative
      8. ActionAgent.persist()              → store + audit log
    """
    t_start = time.perf_counter()

    collector = SignalCollectorAgent()
    scorer = RiskScorerAgent(db=db)
    explainer = ExplanationAgent()
    policy = PolicyAgent()
    action_agent = ActionAgent(db=db)
    trace_agent = ReasoningTraceAgent()

    scoring_passes = 1
    deep_signals: list[str] = []

    # ── STEP 1: Standard signal collection ──────────────────────────
    record, warnings = collector.collect(payload.model_dump())

    # ── STEP 2: Initial scoring ─────────────────────────────────────
    try:
        selected_model = scorer.select_model(record)
        score_result = scorer.score(record, model_version=selected_model)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    probability = score_result["probability"]
    risk_score = score_result["risk_score"]
    model_version = score_result["model_version"]
    initial_risk_score = risk_score  # save for reasoning trace

    # ── STEP 3: Agentic re-evaluation loop ──────────────────────────
    high_risk_signals = (
        float(record.get("TransactionAmt") or 0) > 2000
        and (not record.get("DeviceType") or not record.get("DeviceInfo"))
    )

    if (
        settings.enable_adaptive_loop
        and (
            settings.ambiguous_score_low <= risk_score <= settings.ambiguous_score_high
            or (risk_score < settings.ambiguous_score_low and high_risk_signals)
        )
    ):
        logger.info(
            "Risk score %d triggered adaptive loop — deep collection starting",
            risk_score,
        )
        enriched_record, deep_signals = collector.deep_collect(record, risk_score)
        rescore_result = scorer.rescore(enriched_record, risk_score, deep_signals)

        probability = rescore_result["probability"]
        risk_score = rescore_result["risk_score"]
        model_version = rescore_result["model_version"]
        scoring_passes = 2
        record = enriched_record

    # ── STEP 4: Policy decision ─────────────────────────────────────
    policy_result = policy.decide_with_context(record, probability=probability, risk_score=risk_score)
    decision = policy_result["decision"]

    # ── STEP 5: Explanation (LLM-powered or heuristic fallback) ─────
    if (
        settings.enable_enhanced_explanations
        and policy_result["needs_enhanced_explanation"]
    ):
        explanation = explainer.explain_enhanced(
            record, probability=probability, risk_score=risk_score,
            decision=decision, deep_signals=deep_signals,
        )
    else:
        explanation = explainer.explain(record, probability=probability, risk_score=risk_score)

    reason_codes = explanation["reason_codes"]
    explanation_text = explanation["explanation_text"]
    explanation_source = explanation.get("explanation_source", "heuristic")

    # ── STEP 6: Reasoning trace (LLM narrates the pipeline) ────────
    latency_ms = (time.perf_counter() - t_start) * 1000

    reasoning_trace = trace_agent.generate_trace(
        record=record,
        initial_score=initial_risk_score,
        final_score=risk_score,
        probability=probability,
        decision=decision,
        reason_codes=reason_codes + warnings,
        scoring_passes=scoring_passes,
        deep_signals=deep_signals,
        applied_rules=policy_result.get("applied_rules", []),
        latency_ms=latency_ms,
    )

    # ── STEP 7: Persist ─────────────────────────────────────────────
    tx = action_agent.persist(
        raw_record=record,
        probability=probability,
        risk_score=risk_score,
        decision=decision,
        reason_codes=reason_codes + warnings,
        model_version=model_version,
        latency_ms=latency_ms,
        scoring_passes=scoring_passes,
    )

    return ScoreResponse(
        transaction_id=record.get("TransactionID"),
        fraud_probability=probability,
        risk_score_0_99=risk_score,
        decision=decision,
        reason_codes=reason_codes + warnings,
        explanation_text=explanation_text,
        model_version=model_version,
        explanation_source=explanation_source,
        reasoning_trace=reasoning_trace,
    )