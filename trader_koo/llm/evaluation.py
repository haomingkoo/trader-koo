"""Deterministic semantic checks for the bounded setup-copy rewrite seam."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

EVALUATOR_VERSION = "setup-grounding-v8"
CACHE_VERSION = "setup-rewrite-cache-v9"
PROMPT_TEMPLATE_VERSION = "setup-rewrite-v2"

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


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", text) is not None


def _affirmed(text: str, phrase: str) -> bool:
    for match in re.finditer(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", text):
        clause_prefix = re.split(r"[.;,]", text[:match.start()])[-1][-60:]
        if not re.search(
            r"\b(?:do\s+not|don't|never|avoid|not|no)\s+(?:\w+\s+){0,2}$",
            clause_prefix,
        ):
            return True
    return False


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


def _action_side(text: str) -> str | None:
    action = _normal(text)
    long_side = False
    short_side = False
    exit_targets: set[str] = set()
    for clause in re.split(r"[.;,]|\b(?:and\s+then|then|and|or|but)\b", action):
        clause = clause.strip()
        if not clause:
            continue
        targets = {
            match.group(1) for match in re.finditer(
            r"\b(?:exit|close|cover)\s+(?:the\s+)?(long|short)(?:\s+position)?\b",
            clause,
            )
        }
        if targets:
            exit_targets.update(targets)
            continue
        if re.search(r"\bbuy\s+to\s+cover\b", clause):
            exit_targets.add("short")
            continue
        if re.search(r"\bsell\s+to\s+close\b", clause):
            exit_targets.add("long")
            continue
        if re.match(r"^\s*(?:exit|close|cover)\b", clause):
            continue
        protective_exit = bool(re.search(
            r"\b(?:exit|close|cover)\s+(?:the\s+)?position\b|"
            r"\bstop\s+to\s+(?:buy|sell)\b|"
            r"^(?:use|place|set|keep)\b.*\bstop\b|"
            r"^(?:buy|sell)\b.*\bif\s+(?:invalidated|stopped)\b",
            clause,
        ))
        if protective_exit:
            continue
        negated_start = re.match(r"^(?:do\s+not|don't|never|avoid)\b", clause) is not None
        long_entry = not negated_start and bool(re.search(
            r"^(?:buy|go\s+long|long|enter(?:\s+a)?\s+long|open(?:\s+a)?\s+long)\b|"
            r"\b(?:to|should|could|may|can|consider)\s+(?:buy|go\s+long|enter\s+long)\b|"
            r"\b(?:consider|before|after)\s+buying\b|"
            r"\b(?:breakout|above\s+resistance)\b",
            clause,
        ))
        short_entry = not negated_start and bool(re.search(
            r"^(?:sell|go\s+short|short|enter(?:\s+a)?\s+short|open(?:\s+a)?\s+short)\b|"
            r"\b(?:to|should|could|may|can|consider)\s+(?:sell|go\s+short|enter\s+short)\b|"
            r"\b(?:consider|before|after)\s+selling\b|"
            r"\b(?:breakdown|below\s+support)\b",
            clause,
        ))
        long_side = long_side or long_entry
        short_side = short_side or short_entry
    if long_side and short_side or len(exit_targets) > 1 or exit_targets and (long_side or short_side):
        return "mixed"
    if long_side or short_side:
        return "long" if long_side else "short"
    if exit_targets:
        return f"exit_{next(iter(exit_targets))}"
    return None


def _action_mode(text: str) -> str:
    action = _normal(text)
    if any(_affirmed(action, phrase) for phrase in (
        "buy now", "sell now", "enter now", "act now", "open a position", "initiate",
    )):
        return "immediate"
    if any(_contains(action, marker) for marker in (
        "wait", "watch", "monitor", "confirmation", "if", "when", "after", "consider",
    )) or any(phrase in action for phrase in ("on a dip", "on dip", "above resistance", "below support")):
        return "conditional"
    if re.match(r"^(buy|sell|enter|short|long)\b", action):
        return "immediate"
    return "neutral"


def _risk_posture(text: str) -> tuple[set[str], bool]:
    risk = _normal(text)
    categories: set[str] = set()
    if any(_affirmed(risk, marker) for marker in (
        "stop", "stops", "invalidation", "protect", "protective",
    )):
        categories.add("stop")
    if any(_affirmed(risk, marker) for marker in ("size", "sizing", "limit")):
        categories.add("size")
    if any(_affirmed(risk, marker) for marker in ("risk", "bounded", "caution")):
        categories.add("generic")
    weakened = bool(re.search(
        r"\b(?:do not|don't|never|no|remove|loosen|ignore)\b.{0,30}"
        r"\b(?:stops?|risk|siz(?:e|ing)|limit|bounded|protect(?:ion|ive)?)\b",
        risk,
    )) or bool(re.search(
        r"\bstops?\b.{0,20}\b(?:never|not)\s+be\s+(?:used|required|necessary)\b|"
        r"\bstops?\b.{0,20}\b(?:is|are)\s+not\s+(?:required|necessary)\b|"
        r"\bstops?\b.{0,20}\b(?:is|are)\s+(?:optional|unnecessary)\b|"
        r"\bstops?\b.{0,20}\b(?:may|can|should)\s+be\s+(?:removed|loosened)\b|"
        r"\brisk\b.{0,20}\b(?:is|may|can|should|must)\s+(?:not\s+bounded|unbounded)\b|"
        r"\brisk\b.{0,20}\b(?:should|must)\s+not\s+be\s+(?:bounded|required)\b|"
        r"\bposition\s+siz(?:e|ing)\b.{0,20}\b(?:should|must)\s+not\s+be\s+(?:limited|bounded|constrained)\b|"
        r"\bstops?\b.{0,20}\b(?:need\s+not|(?:do|does)(?:n't|\s+not)\s+need\s+to)\s+be\s+(?:used|required)\b|"
        r"\bposition\s+siz(?:e|ing)\b.{0,20}\b(?:need\s+not|(?:do|does)(?:n't|\s+not)\s+need\s+to)\s+be\s+(?:limited|bounded|constrained)\b|"
        r"\brisk\b.{0,20}\b(?:need\s+not|(?:do|does)(?:n't|\s+not)\s+need\s+to)\s+be\s+bounded\b|"
        r"\bposition\s+siz(?:e|ing)\b.{0,20}\b(?:may|can|should)\s+be\s+increased\b",
        risk,
    )) or any(phrase in risk for phrase in (
        "unbounded", "increase position size", "increase size", "unlimited risk",
    ))
    return categories, weakened


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

    allowed_numbers = _number_facts(fact_context)
    output_numbers = _output_number_claims(output_text)
    unsupported_numbers = {
        (value, unit) for value, unit in output_numbers
        if not any(
            value == allowed_value and (
                unit == allowed_unit or unit == "plain" and allowed_unit in {"plain", "price"}
                or unit == "price" and allowed_unit == "price"
            )
            for allowed_value, allowed_unit in allowed_numbers
        )
    }
    if unsupported_numbers:
        errors.add("unsupported_numeric_claim")

    output_dates = _dates(output_text)
    if output_dates - _dates(" ".join(fact_texts)):
        errors.add("unsupported_date_claim")

    allowed_tickers = set(re.findall(r"\b[A-Z][A-Z0-9.^-]{1,9}\b", " ".join(fact_texts)))
    allowed_tickers.add(str(context.get("ticker") or "").upper().lstrip("^"))
    mentioned = set(re.findall(r"\b[A-Z][A-Z0-9.^-]{1,9}\b", output_text))
    if {ticker for ticker in mentioned if ticker not in allowed_tickers}:
        errors.add("unsupported_ticker_claim")

    baseline = context.get("baseline") if isinstance(context.get("baseline"), dict) else {}
    baseline_action = str(baseline.get("action") or "")
    output_action = str(output.get("action") or "")
    baseline_side = _action_side(baseline_action)
    output_side = _action_side(output_action)
    if baseline_side != output_side and (baseline_side is not None or output_side is not None):
        errors.add("action_type_changed")
    if output_side == "mixed":
        errors.add("ambiguous_directional_action")
    baseline_mode = _action_mode(baseline_action)
    output_mode = _action_mode(output_action)
    if baseline_mode in {"conditional", "immediate"} and output_mode != baseline_mode:
        errors.add("action_urgency_changed")
    baseline_controls, _ = _risk_posture(str(baseline.get("risk_note") or ""))
    output_controls, weakened_risk = _risk_posture(str(output.get("risk_note") or ""))
    missing_specific_control = any(
        category in baseline_controls and category not in output_controls
        for category in ("stop", "size")
    )
    missing_all_controls = bool(baseline_controls) and not output_controls
    if weakened_risk or missing_specific_control or missing_all_controls:
        errors.add("risk_posture_weakened")

    bias = _normal(context.get("signal_bias"))
    if "bull" in bias and output_side == "short":
        errors.add("direction_action_contradiction")
    if "bear" in bias and output_side == "long":
        errors.add("direction_action_contradiction")
    if not any(direction in bias for direction in ("bull", "bear")) and output_side:
        errors.add("unsupported_directional_action")

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
