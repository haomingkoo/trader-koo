"""Deterministic semantic checks for the bounded setup-copy rewrite seam."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

EVALUATOR_VERSION = "setup-grounding-v1"
CACHE_VERSION = "setup-rewrite-cache-v2"
PROMPT_TEMPLATE_VERSION = "setup-rewrite-v2"

_INJECTION_MARKERS = (
    "ignore previous", "ignore all previous", "system prompt", "developer message",
    "jailbreak", "assistant:", "<script", "do not follow",
)
_PROHIBITED_CLAIMS = (
    "guaranteed", "guarantee", "will rise", "will fall", "outperform",
    "recommended", "size up", "priority allocation", "risk free",
)
_BULLISH_TERMS = ("bullish", "buy", "long", "breakout", "above resistance", "higher")
_BEARISH_TERMS = ("bearish", "sell", "short", "breakdown", "below support", "lower")
_INVALID_EVIDENCE = {"stale", "missing", "unknown", "mismatch", "unverified", "invalid"}


def _text_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [text for item in value.values() for text in _text_values(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _text_values(item)]
    return [str(value)] if value is not None else []


def _normal(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def cache_identity(context: dict[str, Any], runtime: dict[str, Any]) -> str:
    """Hash every input that can change the generated or accepted rewrite."""
    payload = {
        "cache_version": CACHE_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "context": context,
        "runtime": runtime,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def evaluate_setup_rewrite(
    output: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate grounding separately from schema and prose fluency."""
    errors: set[str] = set()
    output_text = " ".join(_text_values(output))
    output_lower = output_text.lower()
    context_texts = _text_values(context)
    fact_context = {
        key: value for key, value in context.items() if key != "constraints"
    }
    fact_texts = _text_values(fact_context)
    context_lower = " ".join(context_texts).lower()

    if any(marker in context_lower for marker in _INJECTION_MARKERS):
        errors.add("prompt_injection_in_evidence")
    if any(claim in output_lower for claim in _PROHIBITED_CLAIMS):
        errors.add("unsupported_causal_or_recommendation_claim")

    evidence_status = _normal(context.get("evidence_status"))
    if any(marker in evidence_status for marker in _INVALID_EVIDENCE):
        errors.add("evidence_unavailable")

    allowed_numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", " ".join(fact_texts)))
    output_numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", output_text))
    if output_numbers - allowed_numbers:
        errors.add("unsupported_numeric_claim")

    allowed_tickers = set(re.findall(r"\b[A-Z][A-Z0-9.^-]{1,9}\b", " ".join(fact_texts)))
    allowed_tickers.add(str(context.get("ticker") or "").upper().lstrip("^"))
    mentioned = set(re.findall(r"\b[A-Z][A-Z0-9.^-]{1,9}\b", output_text))
    if {ticker for ticker in mentioned if ticker not in allowed_tickers}:
        errors.add("unsupported_ticker_claim")

    bias = _normal(context.get("signal_bias"))
    if "bull" in bias and any(term in output_lower for term in _BEARISH_TERMS):
        errors.add("direction_action_contradiction")
    if "bear" in bias and any(term in output_lower for term in _BULLISH_TERMS):
        errors.add("direction_action_contradiction")
    if not any(direction in bias for direction in ("bull", "bear")) and any(
        term in output_lower for term in _BULLISH_TERMS + _BEARISH_TERMS
    ):
        errors.add("unsupported_directional_action")

    baseline = context.get("baseline") if isinstance(context.get("baseline"), dict) else {}
    exact_match = all(_normal(output.get(key)) == _normal(baseline.get(key)) for key in (
        "observation", "action", "risk_note",
    ))
    semantic_outcome = (
        "contradicted" if errors else "preserved" if exact_match else "rephrased"
    )
    return {
        "version": EVALUATOR_VERSION,
        "passed": not errors,
        "errors": sorted(errors),
        "semantic_outcome": semantic_outcome,
        "prose_quality_scored": False,
        "decision_scope": "narrative_only",
    }
