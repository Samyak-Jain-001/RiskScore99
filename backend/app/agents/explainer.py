from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from app.utils.logger import get_logger


logger = get_logger(__name__)

# ── Ollama config ───────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def _call_ollama(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Optional[str]:
    """
    Call Ollama's local API. Returns generated text or None on failure.
    Uses urllib to avoid adding dependencies — Ollama is a simple REST API.
    """
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 500,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as exc:
        logger.warning("Ollama not reachable (%s) — using heuristic fallback", exc)
        return None
    except Exception as exc:
        logger.warning("Ollama call failed: %s — using heuristic fallback", exc)
        return None


def _check_ollama_available() -> bool:
    """Quick check if Ollama is running."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


class ExplanationAgent:
    """
    Generates human-readable reason codes and explanation text.

    Two modes:
      - LLM-powered (Ollama): Rich, contextual, analyst-facing explanations
      - Heuristic fallback: Template-based (if Ollama not running)

    CRITICAL: The LLM is called AFTER the decision is made.
    It explains and narrates — it does NOT decide.
    Reason codes are always generated deterministically.
    """

    def __init__(self):
        self._ollama_available = _check_ollama_available()
        if self._ollama_available:
            logger.info("Ollama detected at %s — LLM explanations enabled (model: %s)",
                        OLLAMA_BASE_URL, OLLAMA_MODEL)
        else:
            logger.info("Ollama not detected — using heuristic explanations")

    # ── Deterministic reason codes (always runs, never LLM) ─────────

    def _generate_reason_codes(
        self, record: Dict[str, Any], probability: float, risk_score: int
    ) -> List[str]:
        """Deterministic reason code extraction — no LLM involved."""
        reasons: List[str] = []

        amt = record.get("TransactionAmt")
        if amt is not None:
            try:
                amt_val = float(amt)
                if amt_val > 2000:
                    reasons.append("HIGH_TRANSACTION_AMOUNT")
                elif amt_val > 500:
                    reasons.append("MEDIUM_TRANSACTION_AMOUNT")
            except (TypeError, ValueError):
                pass

        p_email = (record.get("P_emaildomain") or "").lower()
        if p_email and not any(
            common in p_email for common in ["gmail", "yahoo", "hotmail", "outlook", "live"]
        ):
            reasons.append("UNUSUAL_EMAIL_DOMAIN")

        device_type = (record.get("DeviceType") or "").lower()
        device_info = (record.get("DeviceInfo") or "").lower()
        if not device_type or not device_info:
            reasons.append("DEVICE_RISK_SIGNAL")

        identity_missing = 0
        for key in record.keys():
            if key.startswith("id_") and record.get(key) in (None, np.nan):
                identity_missing += 1
        if identity_missing > 0:
            reasons.append("IDENTITY_MISMATCH_SIGNAL")

        r_email = (record.get("R_emaildomain") or "").lower()
        if p_email and r_email and p_email != r_email:
            reasons.append("EMAIL_DOMAIN_MISMATCH")

        if risk_score >= 80:
            reasons.append("MODEL_TOP_FEATURES")

        if not reasons:
            reasons.append("MODEL_BASELINE_RISK")

        return reasons[:5]

    # ── Standard explanation (heuristic OR LLM) ─────────────────────

    def explain(
        self, record: Dict[str, Any], probability: float, risk_score: int
    ) -> Dict[str, Any]:
        """Standard explanation. Uses LLM if available, heuristic otherwise."""
        reason_codes = self._generate_reason_codes(record, probability, risk_score)

        # Try LLM explanation
        if self._ollama_available:
            llm_text = self._llm_explain(record, probability, risk_score, reason_codes, decision=None)
            if llm_text:
                return {
                    "reason_codes": reason_codes,
                    "explanation_text": llm_text,
                    "explanation_source": "llm",
                }

        # Heuristic fallback
        explanation_text = (
            "RiskScore99 assessed this transaction with a calibrated fraud probability "
            f"of {probability:.3f} (risk score {risk_score}/99). "
            "Reason codes indicate high-level signals such as amount, email domain, "
            "device/identity patterns, and model-driven risk."
        )

        return {
            "reason_codes": reason_codes,
            "explanation_text": explanation_text,
            "explanation_source": "heuristic",
        }

    # ── Enhanced explanation (for BLOCK/REVIEW — inter-agent comm) ──

    def explain_enhanced(
        self,
        record: Dict[str, Any],
        probability: float,
        risk_score: int,
        decision: str,
        deep_signals: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Enhanced explanation triggered by PolicyAgent for BLOCK/REVIEW.
        This is inter-agent communication: PolicyAgent asks for deeper explanation.
        """
        reason_codes = self._generate_reason_codes(record, probability, risk_score)

        # Add deep signals to reason codes
        if deep_signals:
            for sig in deep_signals:
                tag = sig.replace("DEEP_SIGNAL:", "")
                if tag not in reason_codes and len(reason_codes) < 8:
                    reason_codes.append(tag)

        # Try LLM enhanced explanation
        if self._ollama_available:
            llm_text = self._llm_explain(
                record, probability, risk_score, reason_codes,
                decision=decision, deep_signals=deep_signals,
            )
            if llm_text:
                return {
                    "reason_codes": reason_codes,
                    "explanation_text": llm_text,
                    "explanation_source": "llm",
                    "enhanced": True,
                }

        # Heuristic enhanced fallback
        detail_parts: List[str] = []
        amt = float(record.get("TransactionAmt") or 0)

        if decision == "BLOCK":
            detail_parts.append(
                f"Transaction was BLOCKED (score {risk_score}/99, p={probability:.3f})."
            )
            if amt > 5000:
                detail_parts.append(
                    f"High-value transaction (${amt:,.2f}) combined with elevated risk "
                    "triggered automatic blocking per business rules."
                )
        elif decision == "REVIEW":
            detail_parts.append(
                f"Transaction routed for REVIEW (score {risk_score}/99, p={probability:.3f})."
            )

        if deep_signals:
            for sig in deep_signals:
                if "LOW_IDENTITY_COMPLETENESS" in sig:
                    detail_parts.append("Identity data is sparse — fewer than 30% of identity fields populated.")
                elif "DEVICE_EMAIL_CROSS_RISK" in sig:
                    detail_parts.append("Missing device info combined with uncommon email domain.")
                elif "NIGHTTIME_TRANSACTION" in sig:
                    detail_parts.append("Transaction occurred during high-risk hours (midnight–5 AM).")
                elif "NO_IDENTITY_FIELDS" in sig:
                    detail_parts.append("No identity verification fields provided.")

        p_email = (record.get("P_emaildomain") or "").lower()
        r_email = (record.get("R_emaildomain") or "").lower()
        if p_email and r_email and p_email != r_email:
            detail_parts.append(
                f"Purchaser email ({p_email}) differs from recipient email ({r_email})."
            )

        return {
            "reason_codes": reason_codes,
            "explanation_text": " ".join(detail_parts) if detail_parts else f"Transaction routed for {decision} (score {risk_score}/99, p={probability:.3f}).",
            "explanation_source": "heuristic",
            "enhanced": True,
        }

    # ── LLM explanation generation ──────────────────────────────────

    def _llm_explain(
        self,
        record: Dict[str, Any],
        probability: float,
        risk_score: int,
        reason_codes: List[str],
        decision: Optional[str] = None,
        deep_signals: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generate a natural language explanation using Ollama.
        Returns None if the call fails (triggers heuristic fallback).
        """
        system_prompt = (
            "You are a fraud risk analyst AI assistant embedded in a transaction scoring system called RiskScore99. "
            "Your job is to write clear, concise explanations of why a transaction received a specific risk score and decision. "
            "Write for a human fraud analyst who needs to quickly understand the risk signals. "
            "Be specific about the transaction details. Keep it to 2-4 sentences. "
            "Do NOT make up information not present in the data. "
            "Do NOT recommend actions — just explain the score and signals."
        )

        # Build context for the LLM
        tx_context = {
            "amount": record.get("TransactionAmt"),
            "product_code": record.get("ProductCD"),
            "card_network": record.get("card4"),
            "card_type": record.get("card6"),
            "purchaser_email": record.get("P_emaildomain"),
            "recipient_email": record.get("R_emaildomain"),
            "device_type": record.get("DeviceType"),
            "device_info": record.get("DeviceInfo"),
        }
        # Clean out None/empty values for cleaner prompt
        tx_context = {k: v for k, v in tx_context.items() if v}

        user_prompt = (
            f"Explain this transaction's risk assessment:\n\n"
            f"Risk Score: {risk_score}/99\n"
            f"Fraud Probability: {probability:.4f} ({probability*100:.2f}%)\n"
        )
        if decision:
            user_prompt += f"Decision: {decision}\n"

        user_prompt += f"Reason Codes: {', '.join(reason_codes)}\n"
        user_prompt += f"Transaction Details: {json.dumps(tx_context, indent=2)}\n"

        if deep_signals:
            user_prompt += f"Deep Analysis Signals: {', '.join(deep_signals)}\n"

        user_prompt += "\nWrite a clear 2-4 sentence explanation for the fraud analyst."

        return _call_ollama(system_prompt, user_prompt)


# ═══════════════════════════════════════════════════════════════════
# REASONING TRACE GENERATOR
# ═══════════════════════════════════════════════════════════════════

class ReasoningTraceAgent:
    """
    Generates a natural language narrative of the full agentic pipeline
    execution — what each agent did, why, and how they communicated.

    This is called AFTER all decisions are made. It narrates, it does not decide.
    """

    def __init__(self):
        self._ollama_available = _check_ollama_available()

    def generate_trace(
        self,
        record: Dict[str, Any],
        initial_score: int,
        final_score: int,
        probability: float,
        decision: str,
        reason_codes: List[str],
        scoring_passes: int,
        deep_signals: List[str] | None = None,
        applied_rules: List[str] | None = None,
        latency_ms: float | None = None,
    ) -> str:
        """
        Generate the reasoning trace. Uses LLM if available, structured template otherwise.
        """
        trace_data = {
            "initial_score": initial_score,
            "final_score": final_score,
            "probability": probability,
            "decision": decision,
            "reason_codes": reason_codes,
            "scoring_passes": scoring_passes,
            "deep_signals": deep_signals or [],
            "applied_rules": applied_rules or [],
            "latency_ms": latency_ms,
            "amount": record.get("TransactionAmt"),
            "device_type": record.get("DeviceType"),
            "device_info": record.get("DeviceInfo"),
            "p_email": record.get("P_emaildomain"),
        }

        if self._ollama_available:
            llm_trace = self._llm_trace(trace_data)
            if llm_trace:
                return llm_trace

        return self._heuristic_trace(trace_data)

    def _heuristic_trace(self, data: Dict[str, Any]) -> str:
        """Structured template-based trace."""
        parts = []

        parts.append(
            f"[1] SignalCollectorAgent collected and normalized transaction signals. "
            f"Amount: ${float(data.get('amount') or 0):,.2f}."
        )

        parts.append(
            f"[2] RiskScorerAgent scored the transaction. "
            f"Initial risk score: {data['initial_score']}/99 "
            f"(fraud probability: {data['probability']:.4f})."
        )

        if data["scoring_passes"] > 1:
            parts.append(
                f"[3] AGENTIC LOOP: Score {data['initial_score']} triggered deep signal collection. "
                f"SignalCollectorAgent found {len(data['deep_signals'])} additional signals: "
                f"{', '.join(data['deep_signals']) if data['deep_signals'] else 'none'}. "
                f"RiskScorerAgent re-scored: {data['initial_score']} → {data['final_score']}."
            )

        decision_step = 4 if data["scoring_passes"] > 1 else 3
        parts.append(
            f"[{decision_step}] PolicyAgent decided: {data['decision']}."
        )
        if data["applied_rules"]:
            parts.append(
                f"    Business rules applied: {', '.join(data['applied_rules'])}."
            )

        parts.append(
            f"[{decision_step + 1}] ExplanationAgent generated {len(data['reason_codes'])} reason codes."
        )

        parts.append(
            f"[{decision_step + 2}] ActionAgent persisted result to database."
        )

        if data.get("latency_ms"):
            parts.append(f"Total pipeline latency: {data['latency_ms']:.1f}ms.")

        return "\n".join(parts)

    def _llm_trace(self, data: Dict[str, Any]) -> Optional[str]:
        """LLM-generated narrative trace of the agentic pipeline."""
        system_prompt = (
            "You are narrating the execution of an agentic fraud scoring pipeline called RiskScore99. "
            "Describe what each agent did and why, in a clear step-by-step narrative. "
            "Emphasize any inter-agent communication (e.g., PolicyAgent requesting enhanced explanation, "
            "or the adaptive re-evaluation loop triggering). "
            "Write in present tense, technical but readable. 4-8 sentences. "
            "This is for a system audit log, so be precise about what happened."
        )

        user_prompt = f"""Narrate this pipeline execution:

Pipeline Data:
- Initial risk score: {data['initial_score']}/99
- Final risk score: {data['final_score']}/99  
- Fraud probability: {data['probability']:.4f}
- Decision: {data['decision']}
- Scoring passes: {data['scoring_passes']} ({'re-evaluation loop fired' if data['scoring_passes'] > 1 else 'single pass'})
- Deep signals found: {data['deep_signals'] if data['deep_signals'] else 'none'}
- Business rules applied: {data['applied_rules'] if data['applied_rules'] else 'none'}
- Reason codes: {data['reason_codes']}
- Transaction amount: ${float(data.get('amount') or 0):,.2f}
- Latency: {data.get('latency_ms', 'unknown')}ms

Write the step-by-step narrative."""

        return _call_ollama(system_prompt, user_prompt, temperature=0.2)