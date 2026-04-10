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
# Critic thresholds — all bare score/tier numbers live here.
# Changing a threshold requires: (1) update the constant, (2) update the
# parametrized tests in tests/test_critic.py, (3) PR body must include
# trade count impact on last 20+ closed trades. Never mix threshold changes
# with logic changes in the same commit.
# ---------------------------------------------------------------------------

CONVICTION_A_HIGH: float = 75.0       # A-tier high conviction — passes outright
CONVICTION_A_MIN: float = 65.0        # A-tier minimum acceptable
CONVICTION_B_HIGH: float = 75.0       # B-tier strong conviction — passes outright
CONVICTION_B_MIN: float = 70.0        # B-tier minimum (requires approved decision_state)
DEBATE_CONSENSUS_STRONG: float = 75.0  # Strong debate agreement — passes outright
DEBATE_CONSENSUS_MIN: float = 60.0    # Minimum acceptable debate agreement
MIN_REWARD_R: float = 2.0             # Minimum R:R required by critic
GOOD_REWARD_R: float = 2.5            # R:R considered excellent
REGIME_VIX_HIGH_VOL: float = 25.0     # VIX above this = high-vol regime (block new longs)
REGIME_VIX_ELEVATED: float = 30.0     # VIX above this = elevated (log warning)
REGIME_VIX_EXTREME: float = 35.0      # VIX above this = extreme (block ALL entries)
REGIME_VIX_UNKNOWN_BLOCK: float = 22.0  # Block if regime unknown and VIX above this
REGIME_REVERSAL_MIN_SCORE: float = 80.0  # Min score for reversal long in non-bull regime
REGIME_AMBIGUOUS_MIN_SCORE: float = 70.0  # Min score to pass in ambiguous regime
FAMILY_EDGE_BLOCK_WINRATE: float = 0.0   # Win rate at or below this → block (0 = 0%)
FAMILY_EDGE_WEAK_WINRATE: float = 25.0   # Win rate below this → require A-tier
FAMILY_EDGE_MIN_SAMPLE: int = 4          # Minimum closed trades before edge check applies
FAMILY_EDGE_BLOCK_SAMPLE: int = 5        # Minimum trades required for 0% win-rate block
CAUTION_FLAGS_HARD_BLOCK: int = 3        # Block if risk_flags count >= this
CAUTION_FLAGS_WARN: int = 2              # Log warning if risk_flags count >= this
ROLLING_EXPECTANCY_MIN: float = -0.2     # Block new entries if avg PnL below this %
ROLLING_EXPECTANCY_MIN_SAMPLE: int = 5   # Minimum trades needed for expectancy check
CONVICTION_A_GRADE_SCORE: float = 80.0  # Score threshold for A+ vs A grade


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

    if tier == "A" and score >= CONVICTION_A_HIGH:
        return True, f"A-tier with score {score:.0f} — high conviction"
    if tier == "A" and score >= CONVICTION_A_MIN:
        return True, f"A-tier with score {score:.0f} — acceptable conviction"

    # B-tier passes with solid score. Most actionable setups land here.
    if tier == "B" and score >= CONVICTION_B_HIGH:
        return True, f"B-tier with score {score:.0f} — good conviction"
    if tier == "B" and score >= CONVICTION_B_MIN:
        decision = evaluation.get("decision_state", "")
        if decision in ("approved", "approved_with_flags"):
            return True, f"B-tier with score {score:.0f} — adequate conviction ({decision})"
        return False, f"B-tier score {score:.0f} OK but decision_state='{decision}' needs approval"

    return False, (
        f"Conviction too low: tier={tier}, score={score:.0f}. "
        f"Need A-tier ≥{CONVICTION_A_MIN:.0f} or B-tier ≥{CONVICTION_B_MIN:.0f} with approval."
    )


def _check_debate_strength(
    row: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[bool, str]:
    """Strong debate agreement required — no contested trades."""
    agreement = float(row.get("debate_agreement_score") or 0)

    if agreement >= DEBATE_CONSENSUS_STRONG:
        return True, f"Strong debate consensus ({agreement:.0f}%)"
    if agreement >= DEBATE_CONSENSUS_MIN:
        return True, f"Adequate debate consensus ({agreement:.0f}%)"

    return False, f"Debate consensus too weak ({agreement:.0f}%). Critic requires ≥{DEBATE_CONSENSUS_MIN:.0f}% agreement."


def _check_risk_reward(
    plan: dict[str, Any],
) -> tuple[bool, str]:
    """Minimum 2R expected reward for a concentrated portfolio."""
    expected_r = float(plan.get("expected_r_multiple") or 0)

    if expected_r >= GOOD_REWARD_R:
        return True, f"Excellent risk/reward ({expected_r:.1f}R)"
    if expected_r >= MIN_REWARD_R:
        return True, f"Good risk/reward ({expected_r:.1f}R)"

    return False, f"Risk/reward insufficient ({expected_r:.1f}R). Critic requires ≥{MIN_REWARD_R:.1f}R for concentrated portfolio."


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

    if not regime or "unknown" in regime:
        # Fail closed when VIX is elevated — don't let trades through without regime data
        if isinstance(vix, (int, float)) and vix > REGIME_VIX_UNKNOWN_BLOCK:
            return False, f"Regime unknown but VIX={vix:.1f} elevated. Blocking without regime data."
        return True, "Regime unknown, low-vol environment — allowing"

    is_bull = "bull" in regime
    is_bear = "bear" in regime

    # VIX regime context — high vol favors shorts, low vol favors longs
    vix_regime = "normal"
    if isinstance(vix, (int, float)):
        if vix > REGIME_VIX_HIGH_VOL:
            vix_regime = "high_vol"
        elif vix < 16:
            vix_regime = "low_vol"

    # Hard block: longs are forbidden in high-vol environment.
    # This check is unconditional — even bull-regime alignment does NOT override it.
    # A VIX spike inside a longer-term bull trend is a warning sign, not clearance.
    # No tier or score override — the regime must clear before taking longs.
    if direction == "long" and vix_regime == "high_vol":
        vix_str = f"{vix:.1f}" if isinstance(vix, (int, float)) else "elevated"
        return False, (
            f"Long blocked: VIX={vix_str} (high-vol regime). "
            f"No new longs until VIX drops below {REGIME_VIX_HIGH_VOL:.0f}."
        )

    # Aligned trades pass (VIX was checked above — safe to allow)
    if direction == "long" and is_bull:
        return True, f"Long in {regime} — aligned with trend"
    if direction == "short" and is_bear:
        return True, f"Short in {regime} — aligned with trend"

    # VIX regime alignment (shorts in high vol already handled above)
    if direction == "long" and vix_regime == "low_vol":
        return True, f"Long in low-vol VIX regime ({vix:.1f}) — regime-appropriate"

    # Check directional HMM regime (more precise than VIX-based regime).
    # Counter-trend trades are blocked unconditionally — no tier override.
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
            return False, (
                f"Counter-trend blocked: {direction} vs HMM directional regime '{dir_regime}'. "
                "Trade with the regime — no tier override permitted."
            )
        if hmm_aligned:
            return True, f"{direction.title()} aligned with HMM directional regime '{dir_regime}'"

    # Neutral/chop regime or no strong HMM signal
    score = float(row.get("score") or 0)

    # Longs in non-bull regime: only high-conviction reversals allowed.
    # Continuation longs in a non-bull tape = fighting the tape → blocked.
    if direction == "long" and not is_bull:
        family = str(row.get("setup_family") or "").lower()
        is_reversal = "reversal" in family
        if is_reversal and score >= REGIME_REVERSAL_MIN_SCORE:
            return True, (
                f"Reversal long in non-bull regime ({regime}) with score {score:.0f} — "
                f"high-conviction reversal family allowed at ≥{REGIME_REVERSAL_MIN_SCORE:.0f}"
            )
        return False, (
            f"Long blocked in non-bull regime ({regime}). "
            f"Score {score:.0f} "
            f"{'(reversal family — needs ≥' + str(int(REGIME_REVERSAL_MIN_SCORE)) + ')' if is_reversal else '(continuation longs forbidden in non-bull)'}."
        )

    if score >= REGIME_AMBIGUOUS_MIN_SCORE:
        return True, f"No strong regime signal, score {score:.0f} adequate — allowing"

    return False, (
        f"Counter-trend: {direction} in {regime} with score {score:.0f}. "
        f"Need score ≥{REGIME_AMBIGUOUS_MIN_SCORE:.0f} in ambiguous regime."
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

    # Check sector concentration (max 1 open position per sector)
    try:
        from trader_koo.ml.sector_rotation import build_sector_map_from_db

        sector_map = build_sector_map_from_db(conn)
        new_sector = sector_map.get(ticker.upper())
        if new_sector:
            open_sectors = [sector_map.get(str(r[0]).upper()) for r in open_trades]
            same_sector = sum(1 for s in open_sectors if s == new_sector)
            if same_sector >= 1:
                return False, (
                    f"Sector overweight: already holding a position in '{new_sector}'. "
                    "Diversify across sectors."
                )
    except Exception:
        pass  # Fail open — sector check is best-effort

    return True, f"Portfolio OK: {open_count}/{max_open} open, {same_dir} {direction}"


def _check_volatility_environment(
    market_ctx: dict[str, Any],
) -> tuple[bool, str]:
    """In extreme volatility, the critic blocks all new entries."""
    vix = market_ctx.get("vix_at_entry")
    if vix is None:
        return True, "VIX data unavailable — no vol check"

    if vix > REGIME_VIX_EXTREME:
        return False, f"VIX at {vix:.1f} — extreme volatility. Critic blocks all new entries."
    if vix > REGIME_VIX_ELEVATED:
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

    if len(risk_flags) >= CAUTION_FLAGS_HARD_BLOCK:
        return False, f"Too many risk flags ({len(risk_flags)}): {', '.join(risk_flags[:CAUTION_FLAGS_HARD_BLOCK])}. Critic rejects."

    if len(risk_flags) >= CAUTION_FLAGS_WARN:
        return True, f"Approved with {len(risk_flags)} flags — marginal, allowing"

    return True, f"Approved with flags: {len(risk_flags)} — acceptable"


def _check_family_edge(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[bool, str]:
    """Block families with proven negative edge.

    Two-layer check:
    1. calibration_state (persistent, updated every 3 sessions) — blocks on sustained
       negative expectancy over a combined eval + paper sample.
    2. paper_trades recent win rate — blocks on 0% WR over FAMILY_EDGE_BLOCK_SAMPLE trades.
    """
    family = str(row.get("setup_family") or "").lower().replace(" ", "_")
    direction = str(evaluation.get("direction") or "").lower()
    if not family or not direction:
        return True, "No family/direction — skipping family edge check"

    # Layer 1: check calibration_state (broad sample, pre-computed every 3 sessions)
    try:
        calib = conn.execute(
            "SELECT block_new_entries, score_adjustment, hit_rate_pct, expectancy_pct, "
            "combined_sample_count FROM calibration_state "
            "WHERE family = ? AND direction = ?",
            (family, direction),
        ).fetchone()
        if calib is not None and int(calib[0]) == 1:
            exp = calib[3]
            n = calib[4]
            hr = calib[2]
            return False, (
                f"Family '{family}' {direction} blocked by calibration pulse: "
                f"expectancy {exp:.1f}% | hit {hr:.0f}% | {n} combined samples. "
                "Will restore when edge recovers to ≥0%."
            )
    except Exception:
        pass  # Table not yet created — fail open and fall through

    # Layer 2: recent paper_trades win rate (real-time, smaller sample)
    rows = conn.execute(
        "SELECT pnl_pct FROM paper_trades "
        "WHERE status != 'open' AND pnl_pct IS NOT NULL "
        "AND LOWER(REPLACE(setup_family, ' ', '_')) = ? AND direction = ? "
        "ORDER BY exit_date DESC LIMIT 10",
        (family, direction),
    ).fetchall()

    if len(rows) < FAMILY_EDGE_MIN_SAMPLE:
        return True, f"Family '{family}' {direction}: only {len(rows)} trades — insufficient for edge check"

    wins = sum(1 for r in rows if float(r[0]) > 0)
    win_rate = wins / len(rows) * 100

    if win_rate <= FAMILY_EDGE_BLOCK_WINRATE and len(rows) >= FAMILY_EDGE_BLOCK_SAMPLE:
        return False, (
            f"Family '{family}' {direction} has 0% win rate over {len(rows)} trades. "
            "Blocking until edge recovers."
        )

    if win_rate < FAMILY_EDGE_WEAK_WINRATE:
        tier = str(row.get("setup_tier") or "").upper()
        if tier != "A":
            return False, (
                f"Family '{family}' {direction} win rate {win_rate:.0f}% "
                f"({wins}/{len(rows)}). Need A-tier to override weak edge."
            )
        return True, (
            f"Family '{family}' {direction} win rate {win_rate:.0f}% — weak but A-tier override"
        )

    return True, f"Family '{family}' {direction} win rate {win_rate:.0f}% ({wins}/{len(rows)}) — OK"


def _check_rolling_expectancy(
    conn: sqlite3.Connection,
) -> tuple[bool, str]:
    """Reject if last 20 closed trades have negative avg return."""
    rows = conn.execute(
        "SELECT pnl_pct FROM paper_trades "
        "WHERE status != 'open' AND pnl_pct IS NOT NULL "
        "ORDER BY exit_date DESC LIMIT 20",
    ).fetchall()
    if len(rows) < ROLLING_EXPECTANCY_MIN_SAMPLE:
        return True, f"Insufficient history ({len(rows)} trades) for expectancy check"
    avg_pnl = sum(float(r[0]) for r in rows) / len(rows)
    if avg_pnl < ROLLING_EXPECTANCY_MIN:
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
        ("family_edge", lambda: _check_family_edge(conn, row, evaluation)),
    ]

    # Checks that depend on external data availability — fail open on error
    data_dependent_checks = {"regime_alignment", "volatility_environment", "rolling_expectancy", "family_edge"}

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
        if tier == "A" and score >= CONVICTION_A_GRADE_SCORE:
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
