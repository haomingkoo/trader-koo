"""Setup ranking for one report run: score, enrich, guardrail, calibrate and order.

The generator calls four steps in this order:

1. ``score_setup`` builds one row per eligible mover.
2. ``rank_setups`` gives the preliminary order before candle patterns are known.
3. ``rerank_with_candles`` re-scores with the day's candles and sets the final order.
4. ``calibrate_setups`` applies setup-call reliability, after outcomes are scored.

The step functions live in ``setup_scoring``. This module owns their sequence,
the two sort orders, and the setup surfaces they write into the report signals.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

from trader_koo.report.ensemble_adjustment import apply_ensemble_adjustment as _apply_ensemble_adjustment
from trader_koo.report.setup_scoring import (
    OPTIONS_RESEARCH_ENABLED,
    _apply_agreement_tier_adjustment,
    _apply_debate_guardrails,
    _apply_debate_payload,
    _apply_news_research_context,
    _apply_options_research_context,
    _apply_setup_eval_fields,
    _describe_setup,
    _refresh_setup_eval_surfaces,
    _score_setup_from_confluence,
    _yolo_pattern_bias,
)
from trader_koo.report.suggestions import build_suggestions
from trader_koo.sec_events import build_setup_sec_events
from trader_koo.report.utils import _clamp, _setup_tier

LOG = logging.getLogger(__name__)

# Computed by market_context and read by setup_scoring, debate_engine and llm_narrative.
_SCORER_CONTEXT_FIELDS = ("ma_signal", "ma_major_signal", "ma_reclaim_state", "level_event", "recent_gap_state", "recent_gap_days")


def _preliminary_order(r: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(r.get("score") or 0.0),
        float(r.get("pct_change") or 0.0),
        float(r.get("discount_pct") or -999.0),
    )


def _final_order(r: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(r.get("score") or 0.0),
        float(r.get("confirmation_count") or 0.0),
        -float(r.get("contradiction_count") or 0.0),
        float(r.get("pct_change") or 0.0),
    )


def _options_research_surface(annotated: Any) -> dict[str, Any]:
    return {
        "enabled": OPTIONS_RESEARCH_ENABLED,
        "annotated": annotated,
        "source": "yfinance_options_iv",
        "note": (
            "Yahoo/yfinance option-chain snapshots; delayed/unofficial. "
            "Used for IV/OI rank context, not real-time sweeps or net premium."
        ),
    }


def _apply_debate_steps(setup_rows: list[dict[str, Any]]) -> None:
    _apply_debate_payload(setup_rows)
    # Apply agreement score tier adjustment after debate payload, before guardrails
    for row in setup_rows:
        if isinstance(row, dict):
            _apply_agreement_tier_adjustment(row)
    _apply_debate_guardrails(setup_rows)


def score_setup(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    pct_change: float,
    near_high: bool,
    near_low: bool,
    fund: dict[str, Any],
    sector: str,
    yolo: dict[str, Any] | None,
    vol: dict[str, Any],
    tech: dict[str, Any],
    vol_ctx: dict[str, Any],
) -> dict[str, Any]:
    """Score one mover: valuation + momentum + AI signal freshness, then the ensemble."""
    score = 50.0
    discount = fund.get("discount_pct")
    discount_component = 0.0
    if isinstance(discount, (int, float)):
        discount_component = _clamp(float(discount) * 0.8, -20.0, 20.0)
        score += discount_component

    peg = fund.get("peg")
    peg_component = 0.0
    if isinstance(peg, (int, float)) and float(peg) > 0:
        peg_v = float(peg)
        if peg_v <= 0.8:
            peg_component = 15.0
        elif peg_v <= 1.5:
            peg_component = 10.0
        elif peg_v <= 2.5:
            peg_component = 4.0
        elif peg_v <= 4.0:
            peg_component = -4.0
        else:
            peg_component = -8.0
        score += peg_component

    momentum_component = _clamp(pct_change * 1.5, -12.0, 12.0)
    score += momentum_component

    proximity_component = 0.0
    if near_high and pct_change > 0:
        proximity_component += 5.0
    if near_low and pct_change < 0:
        proximity_component -= 5.0
    score += proximity_component

    yolo_component = 0.0
    volatility_component = 0.0
    yolo_pattern = None
    yolo_confidence = None
    yolo_age_days = None
    yolo_timeframe = None
    yolo_first_seen_asof = None
    yolo_last_seen_asof = None
    yolo_snapshots_seen = None
    yolo_current_streak = None
    yolo_first_seen_days_ago = None
    signal_bias = "neutral"
    atr_pct_14 = vol.get("atr_pct_14")
    realized_vol_20 = vol.get("realized_vol_20")
    bb_width_20 = vol.get("bb_width_20")
    if yolo:
        yolo_pattern = yolo.get("pattern")
        yolo_confidence = yolo.get("confidence")
        yolo_age_days = yolo.get("age_days")
        yolo_timeframe = yolo.get("timeframe")
        yolo_first_seen_asof = yolo.get("first_seen_asof")
        yolo_last_seen_asof = yolo.get("last_seen_asof")
        yolo_snapshots_seen = yolo.get("snapshots_seen")
        yolo_current_streak = yolo.get("current_streak")
        yolo_first_seen_days_ago = yolo.get("first_seen_days_ago")
        signal_bias = _yolo_pattern_bias(yolo_pattern)
        conf = float(yolo_confidence or 0.0)
        yolo_component += _clamp(conf * 20.0, 0.0, 18.0)
        if str(yolo_timeframe) == "daily":
            yolo_component += 2.0
        if isinstance(yolo_age_days, int):
            if yolo_age_days <= 10:
                yolo_component += 3.0
            elif yolo_age_days <= 30:
                yolo_component += 1.0
        score += yolo_component

    if isinstance(atr_pct_14, (int, float)):
        atr_v = float(atr_pct_14)
        if atr_v < 1.0:
            volatility_component -= 2.0
        elif atr_v <= 4.5:
            volatility_component += 5.0
        elif atr_v <= 7.0:
            volatility_component += 2.0
        elif atr_v <= 10.0:
            volatility_component -= 2.0
        else:
            volatility_component -= 6.0

    if isinstance(bb_width_20, (int, float)):
        bb_v = float(bb_width_20)
        if bb_v <= 6.0 and abs(pct_change) >= 1.5:
            volatility_component += 3.0
        elif bb_v >= 18.0:
            volatility_component -= 3.0

    if isinstance(realized_vol_20, (int, float)):
        rv_v = float(realized_vol_20)
        if rv_v > 60.0:
            volatility_component -= 3.0
        elif rv_v >= 20.0:
            volatility_component += 2.0
        elif rv_v < 12.0:
            volatility_component -= 1.0

    vix_pctile = vol_ctx.get("vix_percentile_1y")
    if isinstance(vix_pctile, (int, float)):
        vix_p = float(vix_pctile)
        if vix_p >= 90.0:
            volatility_component -= 4.0
        elif vix_p >= 75.0:
            volatility_component -= 2.0
        elif vix_p <= 35.0:
            volatility_component += 1.0

    volatility_component = _clamp(volatility_component, -15.0, 15.0)
    score += volatility_component

    row = {
        "ticker": ticker,
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "confluence_score": round(_clamp(score, 0.0, 100.0), 1),
        "setup_tier": _setup_tier(round(_clamp(score, 0.0, 100.0), 1)),
        "sector": sector,
        "pct_change": round(pct_change, 2),
        "discount_pct": discount,
        "peg": peg,
        "atr_pct_14": atr_pct_14,
        "realized_vol_20": realized_vol_20,
        "bb_width_20": bb_width_20,
        "avg_volume_20": tech.get("avg_volume_20"),
        "volume_ratio_20": tech.get("volume_ratio_20"),
        "recent_range_pct_10": tech.get("recent_range_pct_10"),
        "recent_range_pct_20": tech.get("recent_range_pct_20"),
        "near_52w_high": near_high,
        "near_52w_low": near_low,
        "yolo_pattern": yolo_pattern,
        "yolo_confidence": yolo_confidence,
        "yolo_age_days": yolo_age_days,
        "yolo_timeframe": yolo_timeframe,
        "yolo_first_seen_asof": yolo_first_seen_asof,
        "yolo_last_seen_asof": yolo_last_seen_asof,
        "yolo_snapshots_seen": yolo_snapshots_seen,
        "yolo_current_streak": yolo_current_streak,
        "yolo_first_seen_days_ago": yolo_first_seen_days_ago,
        "signal_bias": signal_bias,
        "close": tech.get("close"),
        "ma20": tech.get("ma20"),
        "ma50": tech.get("ma50"),
        "pct_vs_ma20": tech.get("pct_vs_ma20"),
        "pct_vs_ma50": tech.get("pct_vs_ma50"),
        "pct_from_20d_high": tech.get("pct_from_20d_high"),
        "pct_from_20d_low": tech.get("pct_from_20d_low"),
        "trend_state": tech.get("trend_state") or "mixed",
        "level_context": tech.get("level_context") or "mid_range",
        "support_level": tech.get("support_level"),
        "support_zone_low": tech.get("support_zone_low"),
        "support_zone_high": tech.get("support_zone_high"),
        "support_tier": tech.get("support_tier"),
        "support_touches": tech.get("support_touches"),
        "resistance_level": tech.get("resistance_level"),
        "resistance_zone_low": tech.get("resistance_zone_low"),
        "resistance_zone_high": tech.get("resistance_zone_high"),
        "resistance_tier": tech.get("resistance_tier"),
        "resistance_touches": tech.get("resistance_touches"),
        "pct_to_support": tech.get("pct_to_support"),
        "pct_to_resistance": tech.get("pct_to_resistance"),
        "range_position": tech.get("range_position"),
        "stretch_state": tech.get("stretch_state") or "normal",
        "breakout_state": tech.get("breakout_state") or "none",
        "structure_state": tech.get("structure_state") or "normal",
        **{k: tech.get(k) for k in _SCORER_CONTEXT_FIELDS},
        "candle_pattern": None,
        "candle_bias": "neutral",
        "candle_confidence": None,
        "observation": "",
        "actionability": "watch-only",
        "action": "",
        "risk_note": "",
        "technical_read": "",
        "components": {
            "discount": round(discount_component, 2),
            "peg": round(peg_component, 2),
            "momentum": round(momentum_component, 2),
            "proximity": round(proximity_component, 2),
            "volatility": round(volatility_component, 2),
            "yolo": round(yolo_component, 2),
        },
    }
    row.update(_score_setup_from_confluence(row))

    # Enrich with 5-strategy technical ensemble
    try:
        from trader_koo.analysis.technical_ensemble import compute_technical_ensemble

        ticker_ohlcv = conn.execute(
            "SELECT date, open, high, low, close, volume FROM price_daily "
            "WHERE ticker = ? AND close IS NOT NULL ORDER BY date ASC",
            (row.get("ticker"),),
        ).fetchall()
        if len(ticker_ohlcv) >= 64:
            import pandas as pd

            ticker_df = pd.DataFrame(
                ticker_ohlcv,
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            ensemble = compute_technical_ensemble(ticker_df)
            row["ensemble_bias"] = ensemble["aggregate"]["bias"]
            row["ensemble_net_score"] = ensemble["aggregate"]["net_score"]
            row["ensemble_agreement_pct"] = ensemble["aggregate"]["agreement_pct"]
    except Exception as exc:
        LOG.debug("Ensemble enrichment skipped for %s: %s", row.get("ticker"), exc)

    # Adjust score/tier based on ensemble agreement with confluence bias
    _apply_ensemble_adjustment(row)
    return row


def rank_setups(conn: sqlite3.Connection, setup_rows: list[dict[str, Any]], signals: dict[str, Any]) -> None:
    """Preliminary pass: options and news context, debate guardrails, preliminary order.

    Sorts ``setup_rows`` in place and writes options_research, setup_news_context,
    setup_quality_top and a short watchlist_candidates into ``signals``.
    """
    options_annotated = _apply_options_research_context(conn, setup_rows)
    signals["options_research"] = _options_research_surface(options_annotated)

    setup_rows.sort(key=_preliminary_order, reverse=True)
    latest_price_row = conn.execute("SELECT MAX(date) FROM price_daily").fetchone()
    setup_news_as_of = str(latest_price_row[0]) if latest_price_row and latest_price_row[0] else None
    signals["setup_news_context"] = _apply_news_research_context(
        setup_rows,
        as_of_date=setup_news_as_of,
        conn=conn,
    )
    setup_rows.sort(key=_preliminary_order, reverse=True)
    _apply_debate_steps(setup_rows)
    signals["setup_quality_top"] = setup_rows[:40]
    signals["watchlist_candidates"] = [
        {
            "ticker": r.get("ticker"),
            "score": r.get("score"),
            "setup_tier": r.get("setup_tier"),
            "pct_change": r.get("pct_change"),
            "yolo_pattern": r.get("yolo_pattern"),
            "yolo_confidence": r.get("yolo_confidence"),
        }
        for r in setup_rows[:20]
    ]


def rerank_with_candles(
    conn: sqlite3.Connection,
    setup_rows: list[dict[str, Any]],
    candle_by_ticker: dict[str, dict[str, Any]],
    signals: dict[str, Any],
    *,
    as_of_date: str | None,
) -> None:
    """Final pass: merge candles, re-score, re-run enrichment and guardrails, final order.

    Sorts ``setup_rows`` in place and writes every setup surface into ``signals``.
    """
    for row in setup_rows:
        ticker = str(row.get("ticker") or "").upper()
        candle = candle_by_ticker.get(ticker) or {}
        if candle:
            row["candle_pattern"] = candle.get("pattern")
            row["candle_bias"] = candle.get("bias") or "neutral"
            row["candle_confidence"] = candle.get("confidence")
        row.update(_score_setup_from_confluence(row))
        _apply_ensemble_adjustment(row)  # the re-score above resets score and tier
        readout = _describe_setup(row)
        row.update(readout)

    options_annotated = _apply_options_research_context(conn, setup_rows)
    signals["options_research"] = _options_research_surface(options_annotated)
    signals["setup_news_context"] = _apply_news_research_context(
        setup_rows,
        as_of_date=as_of_date,
        conn=conn,
    )

    _apply_debate_steps(setup_rows)

    setup_rows.sort(key=_final_order, reverse=True)

    signals["setup_quality_all"] = setup_rows
    signals["setup_quality_top"] = setup_rows[:40]
    signals["setup_sec_events"] = build_setup_sec_events([r.get("ticker") for r in setup_rows], as_of_date=as_of_date)
    signals["suggestions"] = build_suggestions(signals["setup_quality_top"])
    signals["watchlist_candidates"] = [
        {
            "ticker": r.get("ticker"),
            "score": r.get("score"),
            "setup_tier": r.get("setup_tier"),
            "pct_change": r.get("pct_change"),
            "yolo_pattern": r.get("yolo_pattern"),
            "yolo_confidence": r.get("yolo_confidence"),
            "signal_bias": r.get("signal_bias"),
            "observation": r.get("observation"),
            "actionability": r.get("actionability"),
            "action": r.get("action"),
            "risk_note": r.get("risk_note"),
            "technical_read": r.get("technical_read"),
            "candle_pattern": r.get("candle_pattern"),
            "candle_bias": r.get("candle_bias"),
            "candle_confidence": r.get("candle_confidence"),
        }
        for r in setup_rows[:20]
    ]
    signals["setup_quality_lookup"] = {
        str(row.get("ticker") or "").upper(): {
            key: row.get(key)
            for key in (
                "ticker",
                "score",
                "setup_tier",
                "setup_family",
                "sector",
                "pct_change",
                "discount_pct",
                "peg",
                "atr_pct_14",
                "realized_vol_20",
                "bb_width_20",
                "signal_bias",
                "actionability",
                "observation",
                "action",
                "risk_note",
                "technical_read",
                "confirmation_count",
                "contradiction_count",
                "valuation_bias",
                "candle_pattern",
                "candle_bias",
                "candle_confidence",
                "yolo_pattern",
                "yolo_bias",
                "yolo_confidence",
                "yolo_age_days",
                "yolo_timeframe",
                "yolo_recency",
                "yolo_score_eligible",
                "yolo_direction_conflict",
                "yolo_conflict_strength",
                "debate_v1",
                "debate_consensus_state",
                "debate_consensus_bias",
                "debate_agreement_score",
                "debate_disagreement_count",
                "debate_safety_adjustment",
                "calibrated_hit_prob",
                "probability_label",
                "probability_source",
                "probability_sample_size",
                "probability_baseline_pct",
                "trend_state",
                "level_context",
                "support_level",
                "resistance_level",
                "pct_to_support",
                "pct_to_resistance",
                "candle_pattern",
                "candle_bias",
                "candle_confidence",
                "news_context",
                "news_sentiment_score",
                "macro_news_score",
                "options_context",
                "options_iv_rank_pct",
                "options_oi_rank_pct",
                "options_put_call_oi_ratio",
                "options_underpriced_score",
                "options_positioning_signal",
                "options_positioning_skew",
            )
        }
        for row in setup_rows
        if isinstance(row, dict) and row.get("ticker")
    }


def calibrate_setups(
    signals: dict[str, Any],
    *,
    reliability_lookup: dict[str, Any],
    min_sample: int,
    baseline_hit_rate_pct: Any,
    calibration_state: dict[str, Any],
) -> dict[str, Any]:
    """Apply setup-call reliability to the selected setups, re-order them, refresh surfaces.

    Returns the calibration summary from ``_apply_setup_eval_fields``.
    """
    setup_rows = signals.get("setup_quality_top") if isinstance(signals.get("setup_quality_top"), list) else []
    calibration = _apply_setup_eval_fields(
        setup_rows,
        reliability_lookup=reliability_lookup,
        min_sample=min_sample,
        baseline_hit_rate_pct=baseline_hit_rate_pct,
        calibration_state=calibration_state,
    )
    setup_rows.sort(key=_final_order, reverse=True)
    _refresh_setup_eval_surfaces(signals)
    signals["suggestions"] = build_suggestions(setup_rows)
    return calibration
