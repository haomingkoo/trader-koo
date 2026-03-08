from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from trader_koo.llm_health import (
    llm_disable_seconds_on_fail,
    llm_health_summary,
    note_llm_failure,
    note_llm_success,
    note_llm_token_usage,
)
from trader_koo.llm.schemas import SetupRewrite
from trader_koo.llm.sanitizer import sanitize_llm_output
from trader_koo.llm.validator import generate_fallback_narrative, validate_llm_output

_TRUTHY = {"1", "true", "yes", "on"}
_CACHE_LOCK = threading.Lock()
_PROMPT_CACHE: dict[str, dict[str, str]] = {}
_RUNTIME_DISABLED_UNTIL: dt.datetime | None = None
LOG = logging.getLogger("trader_koo.llm")


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def llm_enabled() -> bool:
    return _as_bool(os.getenv("TRADER_KOO_LLM_ENABLED", "0"))


def _llm_provider() -> str:
    provider = str(os.getenv("TRADER_KOO_LLM_PROVIDER", "azure_openai") or "azure_openai").strip().lower()
    return provider if provider in {"azure_openai"} else "azure_openai"


def _llm_timeout_sec() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_TIMEOUT_SEC", "8") or "8").strip()
    try:
        return max(3, min(30, int(raw)))
    except ValueError:
        return 8


def _llm_temperature() -> float:
    raw = str(os.getenv("TRADER_KOO_LLM_TEMPERATURE", "0.2") or "0.2").strip()
    try:
        return max(0.0, min(0.7, float(raw)))
    except ValueError:
        return 0.2


def _llm_max_tokens() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_MAX_TOKENS", "220") or "220").strip()
    try:
        return max(80, min(600, int(raw)))
    except ValueError:
        return 220


def llm_max_setups() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_MAX_SETUPS", "8") or "8").strip()
    try:
        return max(0, min(20, int(raw)))
    except ValueError:
        return 8


def _azure_cfg() -> dict[str, Any]:
    return {
        "endpoint": str(os.getenv("AZURE_OPENAI_ENDPOINT", "") or "").strip().rstrip("/"),
        "api_key": str(os.getenv("AZURE_OPENAI_API_KEY", "") or "").strip(),
        "api_version": str(os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview") or "2024-12-01-preview").strip(),
        "deployment": str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "") or "").strip(),
    }


def _default_db_path() -> Path:
    primary = Path(os.getenv("TRADER_KOO_DB_PATH", str((Path(__file__).resolve().parents[0] / "data" / "trader_koo.db"))))
    return primary.resolve()


def _runtime_disabled_now(now: dt.datetime | None = None) -> bool:
    global _RUNTIME_DISABLED_UNTIL
    if _RUNTIME_DISABLED_UNTIL is None:
        return False
    ts = now or dt.datetime.now(dt.timezone.utc)
    if ts >= _RUNTIME_DISABLED_UNTIL:
        _RUNTIME_DISABLED_UNTIL = None
        return False
    return True


def _runtime_disable_remaining_sec(now: dt.datetime | None = None) -> int:
    if _RUNTIME_DISABLED_UNTIL is None:
        return 0
    ts = now or dt.datetime.now(dt.timezone.utc)
    remaining = int((_RUNTIME_DISABLED_UNTIL - ts).total_seconds())
    return max(0, remaining)


def _set_runtime_disable() -> None:
    global _RUNTIME_DISABLED_UNTIL
    disable_sec = llm_disable_seconds_on_fail()
    if disable_sec <= 0:
        return
    _RUNTIME_DISABLED_UNTIL = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=disable_sec)


def _safe_note_failure(
    db_path: Path,
    *,
    source: str,
    ticker: str | None,
    reason: str,
    error_class: str,
    details: str,
) -> None:
    try:
        note_llm_failure(
            db_path,
            source=source,
            ticker=ticker,
            reason=reason,
            error_class=error_class,
            details=details,
        )
    except Exception:
        LOG.debug("Failed to persist llm failure state", exc_info=True)


def _safe_note_success(
    db_path: Path,
    *,
    source: str,
    ticker: str | None,
) -> None:
    try:
        note_llm_success(
            db_path,
            source=source,
            ticker=ticker,
        )
    except Exception:
        LOG.debug("Failed to persist llm success state", exc_info=True)


def _safe_note_token_usage(
    db_path: Path,
    *,
    source: str,
    ticker: str | None,
    usage_meta: dict[str, Any],
) -> None:
    if not isinstance(usage_meta, dict):
        return
    try:
        note_llm_token_usage(
            db_path,
            source=source,
            ticker=ticker,
            model=usage_meta.get("model"),
            deployment=usage_meta.get("deployment"),
            prompt_tokens=usage_meta.get("prompt_tokens"),
            completion_tokens=usage_meta.get("completion_tokens"),
            total_tokens=usage_meta.get("total_tokens"),
            meta={
                "api_version": usage_meta.get("api_version"),
                "response_id": usage_meta.get("response_id"),
            },
        )
    except Exception:
        LOG.debug("Failed to persist llm token usage", exc_info=True)


def llm_ready() -> bool:
    if not llm_enabled():
        return False
    if _llm_provider() != "azure_openai":
        return False
    cfg = _azure_cfg()
    return bool(cfg["endpoint"] and cfg["api_key"] and cfg["deployment"] and cfg["api_version"])


def llm_status() -> dict[str, Any]:
    cfg = _azure_cfg()
    now = dt.datetime.now(dt.timezone.utc)
    runtime_disabled = _runtime_disabled_now(now)
    health = {}
    try:
        health = llm_health_summary(_default_db_path(), recent_limit=10)
    except Exception:
        health = {}
    return {
        "enabled": llm_enabled(),
        "provider": _llm_provider(),
        "ready": llm_ready(),
        "runtime_disabled": runtime_disabled,
        "runtime_disabled_remaining_sec": _runtime_disable_remaining_sec(now),
        "temperature": _llm_temperature(),
        "timeout_sec": _llm_timeout_sec(),
        "max_tokens": _llm_max_tokens(),
        "max_setups": llm_max_setups(),
        "has_endpoint": bool(cfg["endpoint"]),
        "has_api_key": bool(cfg["api_key"]),
        "has_deployment": bool(cfg["deployment"]),
        "api_version": cfg["api_version"],
        "health": health,
    }


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars + 1]
    if " " not in clipped:
        return text[:max_chars]
    return clipped.rsplit(" ", 1)[0].strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    content = str(raw or "").strip()
    if not content:
        return {}
    if "```" in content:
        fence = content.replace("```json", "```")
        parts = [p.strip() for p in fence.split("```") if p.strip()]
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                try:
                    return json.loads(part)
                except Exception:
                    continue
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        snippet = content[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


def _azure_chat_rewrite(prompt_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _azure_cfg()
    if not (cfg["endpoint"] and cfg["api_key"] and cfg["deployment"]):
        return ({}, {})
    url = (
        f"{cfg['endpoint']}/openai/deployments/{cfg['deployment']}"
        f"/chat/completions?api-version={cfg['api_version']}"
    )
    system_prompt = (
        "You rewrite stock setup copy for a dashboard. "
        "Use ONLY facts from INPUT. Do not fabricate indicators, events, or prices. "
        "Keep text concise, risk-first, and confirmation-first. "
        "Return STRICT JSON only with keys: observation, action, risk_note."
    )
    user_prompt = json.dumps(prompt_payload, ensure_ascii=True, separators=(",", ":"))
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": _llm_temperature(),
        "max_tokens": _llm_max_tokens(),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-key": cfg["api_key"],
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_llm_timeout_sec()) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    try:
        prompt_tokens = max(0, int(float((usage or {}).get("prompt_tokens") or 0)))
    except Exception:
        prompt_tokens = 0
    try:
        completion_tokens = max(0, int(float((usage or {}).get("completion_tokens") or 0)))
    except Exception:
        completion_tokens = 0
    try:
        total_tokens = max(0, int(float((usage or {}).get("total_tokens") or 0)))
    except Exception:
        total_tokens = 0
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    usage_meta = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model": str(payload.get("model") or "").strip() if isinstance(payload, dict) else "",
        "deployment": cfg["deployment"],
        "api_version": cfg["api_version"],
        "response_id": str(payload.get("id") or "").strip() if isinstance(payload, dict) else "",
    }
    choices = payload.get("choices") or []
    if not choices:
        return ({}, usage_meta)
    msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = msg.get("content") if isinstance(msg, dict) else ""
    return (_extract_json_object(str(content or "")), usage_meta)


def maybe_rewrite_setup_copy(row: dict[str, Any], *, source: str) -> dict[str, str]:
    db_path = _default_db_path()
    if _runtime_disabled_now():
        return {}
    if not llm_ready():
        return {}

    base_observation = _compact_text(row.get("observation"), max_chars=320)
    base_action = _compact_text(row.get("action"), max_chars=260)
    base_risk = _compact_text(row.get("risk_note"), max_chars=120)
    if not (base_observation or base_action):
        return {}

    context = {
        "source": str(source or "report"),
        "ticker": str(row.get("ticker") or "").upper(),
        "asof": row.get("asof"),
        "setup_score": row.get("score"),
        "setup_tier": row.get("setup_tier"),
        "setup_family": row.get("setup_family"),
        "signal_bias": row.get("signal_bias"),
        "trend_state": row.get("trend_state"),
        "level_context": row.get("level_context"),
        "level_event": row.get("level_event"),
        "breakout_state": row.get("breakout_state"),
        "structure_state": row.get("structure_state"),
        "stretch_state": row.get("stretch_state"),
        "candle_bias": row.get("candle_bias"),
        "yolo_pattern": row.get("yolo_pattern"),
        "yolo_recency": row.get("yolo_recency"),
        "yolo_age_days": row.get("yolo_age_days"),
        "yolo_timeframe": row.get("yolo_timeframe"),
        "support_level": row.get("support_level"),
        "resistance_level": row.get("resistance_level"),
        "pct_vs_ma20": row.get("pct_vs_ma20"),
        "pct_change": row.get("pct_change"),
        "baseline": {
            "observation": base_observation,
            "action": base_action,
            "risk_note": base_risk or "none",
        },
        "constraints": {
            "observation_max_chars": 260,
            "action_max_chars": 180,
            "risk_note_max_chars": 80,
            "style": "plain text, no markdown",
        },
    }

    prompt_hash = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    with _CACHE_LOCK:
        cached = _PROMPT_CACHE.get(prompt_hash)
    if cached is not None:
        return dict(cached)

    try:
        rewritten, usage_meta = _azure_chat_rewrite(context)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        error_class = "network_or_decode_error"
        _safe_note_failure(
            db_path,
            source=source,
            ticker=context.get("ticker"),
            reason="request_failed",
            error_class=error_class,
            details="LLM request failed due to network/timeout/decode error",
        )
        _set_runtime_disable()
        LOG.warning("LLM rewrite failed (%s); using fallback narrative", error_class)
        fallback = generate_fallback_narrative(context)
        return sanitize_llm_output(fallback, field_limits={"observation": 260, "action": 180, "risk_note": 80})
    except Exception as exc:
        _safe_note_failure(
            db_path,
            source=source,
            ticker=context.get("ticker"),
            reason="unexpected_exception",
            error_class=type(exc).__name__,
            details=str(exc),
        )
        _set_runtime_disable()
        LOG.warning("LLM rewrite failed (%s); using fallback narrative", type(exc).__name__)
        fallback = generate_fallback_narrative(context)
        return sanitize_llm_output(fallback, field_limits={"observation": 260, "action": 180, "risk_note": 80})

    _safe_note_token_usage(
        db_path,
        source=source,
        ticker=context.get("ticker"),
        usage_meta=usage_meta if isinstance(usage_meta, dict) else {},
    )

    # Validate LLM output against schema
    validation_ctx = {"source": source, "ticker": context.get("ticker")}
    validation_result = validate_llm_output(rewritten, SetupRewrite, context=validation_ctx)
    
    if not validation_result.is_valid:
        _safe_note_failure(
            db_path,
            source=source,
            ticker=context.get("ticker"),
            reason="schema_validation_failed",
            error_class="validation_error",
            details=f"Validation errors: {'; '.join(validation_result.errors)}",
        )
        LOG.warning(
            "LLM output validation failed; using fallback narrative. Errors: %s",
            "; ".join(validation_result.errors)
        )
        fallback = generate_fallback_narrative(context)
        out = sanitize_llm_output(fallback, field_limits={"observation": 260, "action": 180, "risk_note": 80})
    else:
        # Sanitize validated output
        validated_dict = validation_result.data.model_dump()
        out = sanitize_llm_output(validated_dict, field_limits={"observation": 260, "action": 180, "risk_note": 80})

    with _CACHE_LOCK:
        _PROMPT_CACHE[prompt_hash] = dict(out)
    _safe_note_success(
        db_path,
        source=source,
        ticker=context.get("ticker"),
    )
    return out
