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
    """Filter by tier and score. B-tier is the workhorse of the pipeline."""
    tier = str(row.get("setup_tier") or "").upper().strip()
    score = float(row.get("score") or 0)

    if tier == "A" and score >= 75:
        return True, f"A-tier with score {score:.0f} — high conviction"
    if tier == "A" and score >= 65:
        return True, f"A-tier with score {score:.0f} — acceptable conviction"

    # B-tier passes with solid score (70+). Most actionable setups land here.
    if tier == "B" and score >= 75:
        return True, f"B-tier with score {score:.0f} — good conviction"
    if tier == "B" and score >= 70:
        decision = evaluation.get("decision_state", "")
        if decision in ("approved", "approved_with_flags"):
            return True, f"B-tier with score {score:.0f} — adequate conviction ({decision})"
        return False, f"B-tier score {score:.0f} OK but decision_state='{decision}' needs approval"

    return False, f"Conviction too low: tier={tier}, score={score:.0f}. Need A-tier ≥65 or B-tier ≥70 with approval."


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
    """Direction should align with regime. VIX regime matters:
    high_vol = favor shorts, low_vol = allow longs, normal = either.
    HMM directional regime is the tiebreaker.
    """
    direction = str(evaluation.get("direction") or "").lower()
    regime = str(market_ctx.get("regime_state_at_entry") or "").lower()
    vix = market_ctx.get("vix_at_entry")

    if not regime or regime == "unknown":
        return True, "Regime unknown — no alignment check"

    is_bull = "bull" in regime
    is_bear = "bear" in regime

    # VIX regime context — high vol favors shorts, low vol favors longs
    vix_regime = "normal"
    if isinstance(vix, (int, float)):
        if vix > 25:
            vix_regime = "high_vol"
        elif vix < 16:
            vix_regime = "low_vol"

    # Aligned trades always pass
    if direction == "long" and is_bull:
        return True, f"Long in {regime} — aligned with trend"
    if direction == "short" and is_bear:
        return True, f"Short in {regime} — aligned with trend"

    # VIX regime alignment (shorts in high vol, longs in low vol)
    if direction == "short" and vix_regime == "high_vol":
        return True, f"Short in high-vol VIX regime ({vix:.1f}) — regime-appropriate"
    if direction == "long" and vix_regime == "low_vol":
        return True, f"Long in low-vol VIX regime ({vix:.1f}) — regime-appropriate"

    # Check directional HMM regime (more precise than VIX-based regime)
    dir_regime = str(market_ctx.get("directional_regime_at_entry") or "").lower()
    if dir_regime:
        hmm_aligned = (
            (direction == "long" and dir_regime == "bullish")
            or (direction == "short" and dir_regime == "bearish")
        )
        hmm_counter = (
            (direction == "long" and dir_regime == "bearish")
            or (direction == "short" and dir_regime == "bullish")
        )
        if hmm_counter:
            # Hard block only for low-conviction counter-trend
            tier = str(row.get("setup_tier") or "").upper()
            score = float(row.get("score") or 0)
            if tier == "A" and score >= 75:
                return True, f"Counter-trend ({direction} vs HMM '{dir_regime}') but A-tier {score:.0f} — allowed"
            return False, (
                f"Counter-trend: {direction} in HMM directional regime '{dir_regime}'. "
                "Need A-tier ≥75 to override."
            )
        if hmm_aligned:
            return True, f"{direction.title()} aligned with HMM directional regime '{dir_regime}'"

    # Neutral/chop regime or no strong signal — allow with decent conviction
    tier = str(row.get("setup_tier") or "").upper()
    score = float(row.get("score") or 0)
    if score >= 70:
        return True, f"No strong regime signal, score {score:.0f} adequate — allowing"

    return False, (
        f"Counter-trend: {direction} in {regime} with score {score:.0f}. "
        "Need score ≥70 in ambiguous regime."
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

    # Check direction imbalance — allow up to half of max_open in same direction
    max_same_dir = max(max_open // 2, 4)
    same_dir = sum(1 for r in open_trades if str(r[1]).lower() == direction.lower())
    if same_dir >= max_same_dir:
        return False, f"Direction overweight: {same_dir}/{max_same_dir} {direction} trades open. Add the other side."

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


def _check_rolling_expectancy(
    conn: sqlite3.Connection,
) -> tuple[bool, str]:
    """Reject if last 20 closed trades have negative avg return."""
    rows = conn.execute(
        "SELECT pnl_pct FROM paper_trades "
        "WHERE status != 'open' AND pnl_pct IS NOT NULL "
        "ORDER BY exit_date DESC LIMIT 20",
    ).fetchall()
    if len(rows) < 5:
        return True, f"Insufficient history ({len(rows)} trades) for expectancy check"
    avg_pnl = sum(float(r[0]) for r in rows) / len(rows)
    if avg_pnl < -0.2:
        return False, f"Rolling expectancy negative ({avg_pnl:.2f}% avg over last {len(rows)} trades)"
    return True, f"Rolling expectancy OK ({avg_pnl:.2f}% avg over last {len(rows)} trades)"


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
        ("rolling_expectancy", lambda: _check_rolling_expectancy(conn)),
    ]

    # Checks that depend on external data availability — fail open on error
    data_dependent_checks = {"regime_alignment", "volatility_environment", "rolling_expectancy"}

    for name, fn in name_fn_pairs:
        try:
            passed, reason = fn()
            checks.append((name, passed, reason))
        except Exception as exc:
            if name in data_dependent_checks:
                LOG.warning("Critic check '%s' failed open (data unavailable): %s", name, exc)
                checks.append((name, True, f"Check error — data unavailable (allowing): {exc}"))
            else:
                LOG.error("Critic check '%s' failed closed (rejecting): %s", name, exc)
                checks.append((name, False, f"Check error (rejecting): {exc}"))

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
