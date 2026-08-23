"""Deterministic fact and contract checks for bounded LLM setup analysis."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

EVALUATOR_VERSION = "setup-grounding-v12"
CACHE_VERSION = "setup-rewrite-cache-v13"
PROMPT_TEMPLATE_VERSION = "setup-rewrite-v3"

_INJECTION_MARKERS = (
    "ignore previous", "ignore all previous", "system prompt", "developer message",
    "jailbreak", "assistant:", "<script", "do not follow",
)
_PROHIBITED_CLAIMS = (
    "guaranteed", "guarantee", "will rise", "will fall", "outperform",
    "recommended", "size up", "priority allocation", "risk free",
)
_INVALID_EVIDENCE = {"stale", "missing", "unknown", "mismatch", "unverified", "invalid"}
_NUMBER = re.compile(
    r"(?<![\w])(?P<prefix>[+-]?\$?|\$[+-]?)(?P<number>\d[\d,]*(?:\.\d+)?)(?P<percent>%?)"
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_DATE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)


def _text_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [text for item in value.values() for text in _text_values(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _text_values(item)]
    return [str(value)] if value is not None else []


def _normal(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _unit(path: str, currency: bool, percent: bool) -> str:
    if percent or any(part in path for part in ("pct", "percent", "rate")):
        return "percent"
    if currency or any(part in path for part in (
        "price", "level", "close", "support", "resistance", "capital", "notional", "cost",
    )):
        return "price"
    return "plain"


def _numeric_text(text: str) -> str:
    normalized = text.translate(str.maketrans({
        "−": "-", "﹣": "-", "－": "-", "＋": "+", "–": "-", "—": "-",
    }))
    return re.sub(r"(?<=\d)\s*-\s*(?=\$?\d)", " ", normalized)


def _number_facts(value: Any, path: str = "") -> set[tuple[Decimal, str]]:
    if isinstance(value, dict):
        return {
            fact for key, item in value.items()
            for fact in _number_facts(item, f"{path}.{str(key).lower()}")
        }
    if isinstance(value, (list, tuple)):
        return {fact for item in value for fact in _number_facts(item, path)}
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float, Decimal)):
        try:
            return {(Decimal(str(value)).normalize(), _unit(path, False, False))}
        except InvalidOperation:
            return set()
    facts: set[tuple[Decimal, str]] = set()
    text = _numeric_text(_ISO_DATE.sub("", str(value)))
    for match in _NUMBER.finditer(text):
        try:
            sign = -1 if "-" in match.group("prefix") else 1
            number = (Decimal(match.group("number").replace(",", "")) * sign).normalize()
        except InvalidOperation:
            continue
        facts.add((number, _unit(path, "$" in match.group("prefix"), bool(match.group("percent")))))
    return facts


def _output_number_claims(text: str) -> set[tuple[Decimal, str]]:
    claims: set[tuple[Decimal, str]] = set()
    without_dates = _numeric_text(_MONTH_DATE.sub("", _ISO_DATE.sub("", text)))
    for match in _NUMBER.finditer(without_dates):
        try:
            sign = -1 if "-" in match.group("prefix") else 1
            number = (Decimal(match.group("number").replace(",", "")) * sign).normalize()
        except InvalidOperation:
            continue
        unit = "percent" if match.group("percent") else "price" if "$" in match.group("prefix") else "plain"
        claims.add((number, unit))
    return claims


def _dates(text: str) -> set[date]:
    values: set[date] = set()
    for raw in _ISO_DATE.findall(text):
        try:
            values.add(date.fromisoformat(raw))
        except ValueError:
            continue
    months = {
        name: index for index, names in enumerate((
            ("jan", "january"), ("feb", "february"), ("mar", "march"),
            ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
            ("aug", "august"), ("sep", "september"), ("oct", "october"),
            ("nov", "november"), ("dec", "december"),
        ), start=1) for name in names
    }
    for raw in _MONTH_DATE.findall(text):
        month_name, day_value, year_value = re.split(r"[\s,]+", raw)
        try:
            values.add(date(int(year_value), months[month_name.lower()], int(day_value)))
        except (KeyError, ValueError):
            continue
    return values


def intent_contract(context: dict[str, Any]) -> dict[str, str]:
    """Return caller-owned enums that the model must echo without mutation."""
    supplied = context.get("intent_contract")
    if isinstance(supplied, dict):
        return {
            "signal_bias": str(supplied.get("signal_bias") or "unspecified"),
            "actionability": str(supplied.get("actionability") or "unspecified"),
            "decision_delta": "none",
        }
    return {
        "signal_bias": str(context.get("signal_bias") or "unspecified").strip().lower(),
        "actionability": str(context.get("actionability") or "unspecified").strip().lower(),
        "decision_delta": "none",
    }


def baseline_candidate(context: dict[str, Any]) -> dict[str, Any]:
    baseline = context.get("baseline") if isinstance(context.get("baseline"), dict) else {}
    return {**baseline, "intent": intent_contract(context)}


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
    """Check immutable decision text, typed intent, and exact evidence facts."""
    errors: set[str] = set()
    output_text = " ".join(_text_values(output))
    output_lower = output_text.lower()
    fact_context = {key: value for key, value in context.items() if key != "constraints"}
    fact_texts = _text_values(fact_context)
    context_lower = " ".join(_text_values(context)).lower()

    if any(marker in context_lower for marker in _INJECTION_MARKERS):
        errors.add("prompt_injection_in_evidence")
    normalized_claims = re.sub(r"[-‐‑‒–—]+", " ", output_lower)
    if any(claim in normalized_claims for claim in _PROHIBITED_CLAIMS):
        errors.add("unsupported_causal_or_recommendation_claim")
    evidence_status = _normal(context.get("evidence_status"))
    if any(marker in evidence_status for marker in _INVALID_EVIDENCE):
        errors.add("evidence_unavailable")

    allowed_numbers = _number_facts(fact_context)
    unsupported_numbers = {
        (value, unit) for value, unit in _output_number_claims(output_text)
        if not any(
            value == allowed_value and (
                unit == allowed_unit
                or unit == "plain" and allowed_unit in {"plain", "price"}
                or unit == "price" and allowed_unit == "price"
            )
            for allowed_value, allowed_unit in allowed_numbers
        )
    }
    if unsupported_numbers:
        errors.add("unsupported_numeric_claim")
    if _dates(output_text) - _dates(" ".join(fact_texts)):
        errors.add("unsupported_date_claim")

    allowed_tickers = set(re.findall(r"\b[A-Z][A-Z0-9.^-]{1,9}\b", " ".join(fact_texts)))
    allowed_tickers.add(str(context.get("ticker") or "").upper().lstrip("^"))
    mentioned = set(re.findall(r"\b[A-Z][A-Z0-9.^-]{1,9}\b", output_text))
    if {ticker for ticker in mentioned if ticker not in allowed_tickers}:
        errors.add("unsupported_ticker_claim")

    baseline = context.get("baseline") if isinstance(context.get("baseline"), dict) else {}
    if str(output.get("action") or "").strip() != str(baseline.get("action") or "").strip():
        errors.add("action_changed")
    if str(output.get("risk_note") or "").strip() != str(baseline.get("risk_note") or "").strip():
        errors.add("risk_note_changed")
    if output.get("intent") != intent_contract(context):
        errors.add("intent_contract_mismatch")

    exact_match = all(_normal(output.get(key)) == _normal(baseline.get(key)) for key in (
        "observation", "action", "risk_note",
    ))
    semantic_outcome = "contradicted" if errors else "preserved" if exact_match else "rephrased"
    return {
        "version": EVALUATOR_VERSION,
        "passed": not errors,
        "errors": sorted(errors),
        "semantic_outcome": semantic_outcome,
        "prose_quality_scored": False,
        "decision_scope": "observation_narrative_only",
        "intent_contract": intent_contract(context),
    }
