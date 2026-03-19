"""Devil's advocate critic for paper trade qualification.

The critic's job is to KILL trades. It looks for reasons NOT to enter.
Only setups that survive the critic's scrutiny proceed to paper trading.

Philosophy: we want a small pool of A+ high-conviction trades, not a
scattered portfolio of 20 mediocre positions. The critic enforces
discipline by challenging every setup before it becomes a trade.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Critic checks — each returns (pass: bool, reason: str)
# ---------------------------------------------------------------------------

def _check_conviction_grade(
    row: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[bool, str]:
    """Only A-tier or A-equivalent setups with high scores pass."""
    tier = str(row.get("setup_tier") or "").upper().strip()
    score = float(row.get("score") or 0)

    if tier == "A" and score >= 75:
        return True, f"A-tier with score {score:.0f} — high conviction"
    if tier == "A" and score >= 65:
        return True, f"A-tier with score {score:.0f} — acceptable conviction"

    # B-tier can pass ONLY with exceptional score AND clean approval
    if tier == "B" and score >= 85 and evaluation.get("decision_state") == "approved":
        return True, f"B-tier but exceptional score {score:.0f} with clean approval"

    return False, f"Conviction too low: tier={tier}, score={score:.0f}. Need A-tier ≥65 or B-tier ≥85 with clean approval."


def _check_debate_strength(
    row: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[bool, str]:
    """Strong debate agreement required — no contested trades."""
    agreement = float(row.get("debate_agreement_score") or 0)

    if agreement >= 75:
        return True, f"Strong debate consensus ({agreement:.0f}%)"
    if agreement >= 60:
        return True, f"Adequate debate consensus ({agreement:.0f}%)"

    return False, f"Debate consensus too weak ({agreement:.0f}%). Critic requires ≥60% agreement."


def _check_risk_reward(
    plan: dict[str, Any],
) -> tuple[bool, str]:
    """Minimum 2R expected reward for a concentrated portfolio."""
    expected_r = float(plan.get("expected_r_multiple") or 0)

    if expected_r >= 2.5:
        return True, f"Excellent risk/reward ({expected_r:.1f}R)"
    if expected_r >= 2.0:
        return True, f"Good risk/reward ({expected_r:.1f}R)"

    return False, f"Risk/reward insufficient ({expected_r:.1f}R). Critic requires ≥2.0R for concentrated portfolio."


def _check_regime_alignment(
    row: dict[str, Any],
    evaluation: dict[str, Any],
    market_ctx: dict[str, Any],
) -> tuple[bool, str]:
    """Direction must align with regime — no counter-trend trades in strong trends."""
    direction = str(evaluation.get("direction") or "").lower()
    regime = str(market_ctx.get("regime_state_at_entry") or "").lower()

    if not regime or regime == "unknown":
        return True, "Regime unknown — no alignment check"

    is_bull = "bull" in regime
    is_bear = "bear" in regime

    # Long in bull regime = aligned
    if direction == "long" and is_bull:
        return True, f"Long in {regime} — aligned with trend"
    # Short in bear regime = aligned
    if direction == "short" and is_bear:
        return True, f"Short in {regime} — aligned with trend"

    # Counter-trend requires exceptional conviction
    tier = str(row.get("setup_tier") or "").upper()
    score = float(row.get("score") or 0)
    if tier == "A" and score >= 80:
        return True, f"Counter-trend ({direction} in {regime}) but A-tier with score {score:.0f} — allowed"

    return False, (
        f"Counter-trend: {direction} in {regime}. "
        "Critic rejects counter-trend trades unless A-tier with score ≥80."
    )


def _check_portfolio_concentration(
    conn: sqlite3.Connection,
    ticker: str,
    direction: str,
    row: dict[str, Any],
    *,
    max_open: int = 5,
) -> tuple[bool, str]:
    """Limit total open trades and avoid sector/family clustering."""
    try:
        open_trades = conn.execute(
            "SELECT ticker, direction, setup_family FROM paper_trades WHERE status = 'open'"
        ).fetchall()
    except Exception:
        return True, "Could not check portfolio — allowing"

    open_count = len(open_trades)

    if open_count >= max_open:
        return False, f"Portfolio full: {open_count}/{max_open} positions open. Wait for exits."

    # Check for duplicate ticker
    open_tickers = {str(r[0]).upper() for r in open_trades}
    if ticker.upper() in open_tickers:
        return False, f"Already holding {ticker}. No duplicate positions."

    # Check direction imbalance (max 4 in same direction)
    same_dir = sum(1 for r in open_trades if str(r[1]).lower() == direction.lower())
    if same_dir >= 4:
        return False, f"Direction overweight: {same_dir} {direction} trades already open. Add the other side."

    # Check family clustering (max 2 from same family)
    family = str(row.get("setup_family") or "").lower()
    if family:
        same_family = sum(1 for r in open_trades if str(r[2] or "").lower() == family)
        if same_family >= 2:
            return False, f"Family clustering: already {same_family} trades from '{row.get('setup_family')}'. Diversify."

    return True, f"Portfolio OK: {open_count}/{max_open} open, {same_dir} {direction}"


def _check_volatility_environment(
    market_ctx: dict[str, Any],
) -> tuple[bool, str]:
    """In extreme volatility, the critic blocks all new entries."""
    vix = market_ctx.get("vix_at_entry")
    if vix is None:
        return True, "VIX data unavailable — no vol check"

    if vix > 35:
        return False, f"VIX at {vix:.1f} — extreme volatility. Critic blocks all new entries."
    if vix > 30:
        return True, f"VIX at {vix:.1f} — elevated but allowing A-tier trades"

    return True, f"VIX at {vix:.1f} — environment acceptable"


def _check_caution_flags(
    evaluation: dict[str, Any],
) -> tuple[bool, str]:
    """Trades approved with too many flags are rejected in concentrated mode."""
    risk_flags = evaluation.get("risk_flags") or []
    decision_state = evaluation.get("decision_state", "")

    if decision_state == "approved":
        return True, "Clean approval — no flags"

    if len(risk_flags) >= 3:
        return False, f"Too many risk flags ({len(risk_flags)}): {', '.join(risk_flags[:3])}. Critic rejects."

    if len(risk_flags) >= 2:
        return True, f"Approved with {len(risk_flags)} flags — marginal, allowing"

    return True, f"Approved with flags: {len(risk_flags)} — acceptable"


# ---------------------------------------------------------------------------
# Main critic evaluation
# ---------------------------------------------------------------------------

def critic_review(
    conn: sqlite3.Connection,
    *,
    row: dict[str, Any],
    evaluation: dict[str, Any],
    plan: dict[str, Any],
    market_ctx: dict[str, Any],
    max_open: int = 5,
) -> dict[str, Any]:
    """Run all critic checks against a proposed trade.

    Returns::

        {
            "approved": bool,
            "checks_passed": int,
            "checks_total": int,
            "conviction_grade": str,     # "A+", "A", "B+", or "rejected"
            "critic_reasons": list[str], # why each check passed or failed
            "rejections": list[str],     # only the failures
            "summary": str,              # one-line verdict
        }
    """
    ticker = str(row.get("ticker") or "").upper()
    direction = str(evaluation.get("direction") or "")

    checks: list[tuple[str, bool, str]] = []

    # Run all critic checks
    name_fn_pairs = [
        ("conviction_grade", lambda: _check_conviction_grade(row, evaluation)),
        ("debate_strength", lambda: _check_debate_strength(row, evaluation)),
        ("risk_reward", lambda: _check_risk_reward(plan)),
        ("regime_alignment", lambda: _check_regime_alignment(row, evaluation, market_ctx)),
        ("portfolio_concentration", lambda: _check_portfolio_concentration(conn, ticker, direction, row, max_open=max_open)),
        ("volatility_environment", lambda: _check_volatility_environment(market_ctx)),
        ("caution_flags", lambda: _check_caution_flags(evaluation)),
    ]

    for name, fn in name_fn_pairs:
        try:
            passed, reason = fn()
            checks.append((name, passed, reason))
        except Exception as exc:
            checks.append((name, True, f"Check error (allowing): {exc}"))

    passed_count = sum(1 for _, p, _ in checks if p)
    total = len(checks)
    rejections = [reason for _, passed, reason in checks if not passed]
    all_reasons = [f"{'PASS' if p else 'FAIL'} [{name}]: {reason}" for name, p, reason in checks]

    approved = len(rejections) == 0

    # Assign conviction grade
    if not approved:
        grade = "rejected"
    elif passed_count == total:
        tier = str(row.get("setup_tier") or "").upper()
        score = float(row.get("score") or 0)
        if tier == "A" and score >= 80:
            grade = "A+"
        elif tier == "A":
            grade = "A"
        else:
            grade = "B+"
    else:
        grade = "B+"

    if approved:
        summary = f"APPROVED ({grade}): {ticker} {direction} passed {passed_count}/{total} critic checks"
    else:
        summary = f"REJECTED: {ticker} {direction} failed {len(rejections)} critic check(s): {rejections[0]}"

    LOG.info("Critic: %s", summary)

    return {
        "approved": approved,
        "checks_passed": passed_count,
        "checks_total": total,
        "conviction_grade": grade,
        "critic_reasons": all_reasons,
        "rejections": rejections,
        "summary": summary,
    }
