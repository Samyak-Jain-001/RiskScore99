from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Subset of IEEE-CIS fields used for MVP scoring."""

    TransactionID: Optional[int] = None
    TransactionDT: Optional[float] = None
    TransactionAmt: float = Field(..., ge=0)
    ProductCD: Optional[str] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None

    class Config:
        extra = "allow"


class ScoreResponse(BaseModel):
    transaction_id: Optional[int]
    fraud_probability: Optional[float]
    risk_score_0_99: Optional[int]
    decision: str
    reason_codes: List[str]
    explanation_text: str
    model_version: Optional[str]


class ImportCsvRequest(BaseModel):
    path: str
    limit: Optional[int] = Field(default=1000, gt=0)


class ImportSummary(BaseModel):
    rows_processed: int
    rows_scored: int
    errors: List[str] = []


class TransactionSummary(BaseModel):
    id: int
    transaction_id: Optional[int]
    timestamp_scored: datetime
    score: int
    probability: float
    decision: str
    model_version: Optional[str]


class TransactionDetail(TransactionSummary):
    raw_json: Dict[str, Any]
    reason_codes: List[str]
    reviewer_action: Optional[str] = None
    reviewer_notes: Optional[str] = None
    outcomes: List[Dict[str, Any]] = []


class ReviewActionRequest(BaseModel):
    action: Literal["approve", "block", "challenge", "review"]
    notes: Optional[str] = None


class OutcomeRequest(BaseModel):
    outcome_label: Literal["confirmed_fraud", "legit", "chargeback", "verified"]
    notes: Optional[str] = None


class MetricsResponse(BaseModel):
    model_version: Optional[str] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    threshold_metrics: Dict[str, Any] = {}
    operational_stats: Dict[str, Any] = {}

