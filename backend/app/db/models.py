from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from .base import Base


class TransactionScored(Base):
    __tablename__ = "transactions_scored"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, nullable=True, index=True)
    timestamp_scored = Column(DateTime, default=datetime.utcnow, nullable=False)
    raw_json = Column(JSON, nullable=False)
    score = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    decision = Column(String(32), nullable=False)
    reason_codes_json = Column(JSON, nullable=False)
    model_version = Column(String(64), nullable=True)
    reviewer_action = Column(String(32), nullable=True)
    reviewer_notes = Column(Text, nullable=True)

    # ── NEW COLUMNS ─────────────────────────────────────────────────
    reviewer_id = Column(String(128), nullable=True)          # who reviewed
    latency_ms = Column(Float, nullable=True)                 # end-to-end scoring latency
    scoring_passes = Column(Integer, default=1, nullable=False)  # 1=normal, 2=re-evaluated (agentic loop)

    outcomes = relationship("Outcome", back_populates="transaction")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(64), nullable=False)
    actor = Column(String(64), nullable=False, default="system")
    payload_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Outcome(Base):
    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, index=True)
    transaction_scored_id = Column(
        Integer, ForeignKey("transactions_scored.id"), nullable=False, index=True
    )
    outcome_label = Column(String(64), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    transaction = relationship("TransactionScored", back_populates="outcomes")


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_version = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metrics_json = Column(JSON, nullable=False)
    data_hash = Column(String(128), nullable=True)
    artifact_path = Column(String(256), nullable=False)

    # ── NEW COLUMNS ─────────────────────────────────────────────────
    is_active = Column(Boolean, default=True, nullable=False)   # supports rollback
    description = Column(Text, nullable=True)                    # human note about this version
