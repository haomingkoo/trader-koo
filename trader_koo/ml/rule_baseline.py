"""Model-free current-rule baseline for validation runs.

This backtest is intentionally deterministic. It uses only point-in-time
technical features, selects trades with the app's current swing-trade bias
families, and simulates the same position accounting as the ML backtest. The
goal is not to create another alpha engine; it gives model promotion a real
playbook hurdle instead of comparing only against SPY.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from trader_koo.ml.features import ML_CONTEXT_TICKERS, extract_features_for_universe

LOG = logging.getLogger(__name__)

RULE_BASELINE_METHOD = "current_rule_technical_proxy"

DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_MAX_POSITIONS = 5
DEFAULT_POSITION_PCT = 15.0
DEFAULT_COMMISSION_PER_TRADE = 5.0
DEFAULT_SLIPPAGE_PCT = 0.05

RULE_BASELINE_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_21d",
    "atr_pct_14",
    "volume_ratio_20d",
    "dist_ma20_pct",
    "dist_ma50_pct",
    "xrank_ret_1d",
    "xrank_ret_5d",
    "xrank_ret_21d",
    "xrank_vol_21d",
    "xrank_atr_pct_14",
    "xrank_volume_ratio_20d",
    "xrank_dist_ma20_pct",
    "xrank_dist_ma50_pct",
    "xrank_mean_reversion_5d",
]


def _compute_exit(
    pos: dict[str, Any],
    exit_price: float,
    exit_date: str,
    exit_reason: str,
    trading_days_held: int,
) -> dict[str, Any]:
    direction = pos["direction"]
    entry_price = pos["entry_price"]
    shares = pos["shares"]
    if direction == "long":
        pnl = (exit_price - entry_price) * shares - DEFAULT_COMMISSION_PER_TRADE
        return_pct = (exit_price / entry_price - 1.0) * 100.0
    else:
        pnl = (entry_price - exit_price) * shares - DEFAULT_COMMISSION_PER_TRADE
        return_pct = (1.0 - exit_price / entry_price) * 100.0
    return {
        **pos,
        "exit_date": exit_date,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
        "days_held": trading_days_held,
    }


def _return_cash_on_close(pos: dict[str, Any], exit_price: float, pnl: float) -> float:
    del exit_price
    entry_commission = DEFAULT_COMMISSION_PER_TRADE / 2
    return pos["entry_price"] * pos["shares"] + pnl + entry_commission


def _apply_exit_slippage(price: float, direction: str, exit_reason: str) -> float:
    del exit_reason
    slip = DEFAULT_SLIPPAGE_PCT / 100.0
    if direction == "long":
        return price * (1.0 - slip)
    return price * (1.0 + slip)


def _mark_to_market(pos: dict[str, Any], current_price: float) -> float:
    shares = pos["shares"]
    entry_price = pos["entry_price"]
    if pos["direction"] == "long":
        return current_price * shares
    return entry_price * shares + (entry_price - current_price) * shares


def _hit_target(price: float, target: float, direction: str) -> bool:
    if direction == "long":
        return price >= target
    return price <= target


def _hit_stop(price: float, stop: float, direction: str) -> bool:
    if direction == "long":
        return price <= stop
    return price >= stop


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _tier(score: float) -> str:
    if score >= 82.0:
        return "A"
    if score >= 74.0:
        return "B"
    if score >= 68.0:
        return "C"
    return "D"


def _add_rank_score(
    score: float,
    value: float,
    *,
    high_is_good: bool,
    threshold: float,
    weight: float,
) -> float:
    if not np.isfinite(value):
        return score
    if high_is_good:
        if value >= threshold:
            return score + (value - threshold) / max(1.0 - threshold, 0.01) * weight
    elif value <= threshold:
        return score + (threshold - value) / max(threshold, 0.01) * weight
    return score


def _candidate_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Score one feature row with the current-rule technical proxy."""
    ret_5d = _to_float(row.get("ret_5d"))
    ret_21d = _to_float(row.get("ret_21d"))
    ret_63d = _to_float(row.get("ret_63d"))
    atr_pct = _to_float(row.get("atr_pct_14"))
    dist_ma50 = _to_float(row.get("dist_ma50_pct"))
    xret_21d = _to_float(row.get("xrank_ret_21d"))
    xatr = _to_float(row.get("xrank_atr_pct_14"))
    xvol = _to_float(row.get("xrank_vol_21d"))
    xdist_50 = _to_float(row.get("xrank_dist_ma50_pct"))
    xmean_rev = _to_float(row.get("xrank_mean_reversion_5d"))
    xvolume = _to_float(row.get("xrank_volume_ratio_20d"))

    candidates: list[dict[str, Any]] = []

    # The paper book is currently strongest in bullish reversals, so demand
    # pullback evidence plus tolerable volatility instead of buying every dip.
    score = 46.0
    reasons: list[str] = []
    if np.isfinite(xdist_50) and xdist_50 <= 0.40:
        score = _add_rank_score(score, xdist_50, high_is_good=False, threshold=0.40, weight=18.0)
        reasons.append("near lower trend rank")
    if np.isfinite(xmean_rev) and xmean_rev <= 0.35:
        score = _add_rank_score(score, xmean_rev, high_is_good=False, threshold=0.35, weight=14.0)
        reasons.append("short-term pullback")
    if np.isfinite(ret_5d) and ret_5d < 0:
        score += min(abs(ret_5d) * 220.0, 10.0)
        reasons.append("recent weakness to fade")
    if np.isfinite(ret_63d) and ret_63d >= -0.15:
        score += 8.0
        reasons.append("not a broken long-term trend")
    if np.isfinite(xatr) and 0.20 <= xatr <= 0.85:
        score += 7.0
        reasons.append("tradable volatility")
    if np.isfinite(ret_21d) and ret_21d < -0.18:
        score -= 9.0
        reasons.append("deep 21d drawdown penalty")
    if np.isfinite(dist_ma50) and dist_ma50 < -18.0:
        score -= 10.0
        reasons.append("too far below MA50")
    if np.isfinite(atr_pct) and atr_pct > 11.0:
        score -= 8.0
        reasons.append("high ATR penalty")
    candidates.append({
        "direction": "long",
        "setup_family": "bullish_reversal",
        "score": score,
        "rule_reasons": reasons,
    })

    # Keep bullish continuation available, but only when it is not already
    # stretched. This is deliberately stricter than reversal.
    score = 42.0
    reasons = []
    if np.isfinite(xret_21d) and xret_21d >= 0.66:
        score = _add_rank_score(score, xret_21d, high_is_good=True, threshold=0.66, weight=18.0)
        reasons.append("upper-quartile 21d momentum")
    if np.isfinite(xdist_50) and 0.52 <= xdist_50 <= 0.84:
        score += 12.0
        reasons.append("above trend but not extreme")
    if np.isfinite(ret_63d) and ret_63d > 0:
        score += min(ret_63d * 120.0, 12.0)
        reasons.append("positive 63d trend")
    if np.isfinite(xmean_rev) and xmean_rev >= 0.55:
        score += 6.0
    if np.isfinite(xmean_rev) and xmean_rev > 0.92:
        score -= 14.0
        reasons.append("short-term stretch penalty")
    if np.isfinite(dist_ma50) and dist_ma50 > 14.0:
        score -= 12.0
        reasons.append("too far above MA50")
    candidates.append({
        "direction": "long",
        "setup_family": "bullish_continuation",
        "score": score,
        "rule_reasons": reasons,
    })

    # Bearish continuation is the other family currently showing edge. Rank
    # weak momentum and weak trend position, with volatility capped.
    score = 46.0
    reasons = []
    if np.isfinite(xret_21d) and xret_21d <= 0.35:
        score = _add_rank_score(score, xret_21d, high_is_good=False, threshold=0.35, weight=18.0)
        reasons.append("lower-quartile 21d momentum")
    if np.isfinite(xdist_50) and xdist_50 <= 0.35:
        score = _add_rank_score(score, xdist_50, high_is_good=False, threshold=0.35, weight=14.0)
        reasons.append("below trend rank")
    if np.isfinite(ret_63d) and ret_63d < 0:
        score += min(abs(ret_63d) * 120.0, 12.0)
        reasons.append("negative 63d trend")
    if np.isfinite(ret_5d) and ret_5d <= 0:
        score += min(abs(ret_5d) * 140.0, 8.0)
        reasons.append("recent pressure")
    if np.isfinite(xmean_rev) and xmean_rev <= 0.45:
        score += 7.0
    if np.isfinite(dist_ma50) and dist_ma50 > 4.0:
        score -= 12.0
        reasons.append("above MA50 penalty")
    if np.isfinite(xatr) and xatr > 0.92:
        score -= 10.0
        reasons.append("extreme vol penalty")
    candidates.append({
        "direction": "short",
        "setup_family": "bearish_continuation",
        "score": score,
        "rule_reasons": reasons,
    })

    best = max(candidates, key=lambda item: float(item["score"]))
    best["score"] = round(float(best["score"]), 2)
    best["setup_tier"] = _tier(float(best["score"]))
    if np.isfinite(xvol):
        best["volatility_rank"] = round(float(xvol), 3)
    if np.isfinite(xvolume):
        best["volume_rank"] = round(float(xvolume), 3)
    return best


def score_rule_candidates(
    features: pd.DataFrame,
    *,
    min_score: float = 68.0,
) -> pd.DataFrame:
    """Return ranked current-rule candidates from a feature matrix."""
    if features is None or features.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "direction",
                "setup_family",
                "setup_tier",
                "score",
                "rule_reasons",
            ]
        )

    rows: list[dict[str, Any]] = []
    for ticker, row in features.iterrows():
        candidate = _candidate_from_row(row.to_dict())
        if not candidate or float(candidate.get("score") or 0.0) < min_score:
            continue
        rows.append({"ticker": str(ticker), **candidate})

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "direction",
                "setup_family",
                "setup_tier",
                "score",
                "rule_reasons",
            ]
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _trading_dates(conn: sqlite3.Connection, start_date: str, end_date: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT date
        FROM price_daily
        WHERE ticker = 'SPY'
          AND date >= ?
          AND date <= ?
          AND close IS NOT NULL
        ORDER BY date
        """,
        (start_date, end_date),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _latest_spy_date(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT MAX(date) FROM price_daily WHERE ticker='SPY' AND close IS NOT NULL"
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _spy_return_pct(conn: sqlite3.Connection, start_date: str, end_date: str) -> float:
    prices = pd.read_sql_query(
        """
        SELECT date, CAST(close AS REAL) AS close
        FROM price_daily
        WHERE ticker='SPY'
          AND date >= ?
          AND date <= ?
          AND close IS NOT NULL
        ORDER BY date
        """,
        conn,
        params=[start_date, end_date],
    )
    if len(prices) < 2:
        return 0.0
    start = float(prices["close"].iloc[0])
    end = float(prices["close"].iloc[-1])
    return (end / start - 1.0) * 100.0 if start > 0 else 0.0


def _latest_close(conn: sqlite3.Connection, ticker: str, as_of_date: str) -> float | None:
    row = conn.execute(
        """
        SELECT CAST(close AS REAL)
        FROM price_daily
        WHERE ticker = ?
          AND date <= ?
          AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row or row[0] is None:
        return None
    close = float(row[0])
    return close if close > 0 else None


def _position_barriers(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    direction: str,
    entry_date: str,
    entry_price: float,
    profit_mult: float,
    stop_mult: float,
) -> tuple[float, float] | None:
    rows = conn.execute(
        """
        SELECT CAST(close AS REAL)
        FROM price_daily
        WHERE ticker=?
          AND date<=?
          AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT 22
        """,
        (ticker, entry_date),
    ).fetchall()
    if len(rows) < 10:
        return None
    closes = [float(row[0]) for row in rows if row[0] is not None and float(row[0]) > 0]
    if len(closes) < 10:
        return None
    log_rets = [
        np.log(closes[idx] / closes[idx + 1])
        for idx in range(len(closes) - 1)
        if closes[idx + 1] > 0
    ]
    daily_vol = float(np.std(log_rets))
    if daily_vol <= 0:
        return None
    if direction == "long":
        return (
            entry_price * (1 + profit_mult * daily_vol),
            entry_price * (1 - stop_mult * daily_vol),
        )
    return (
        entry_price * (1 - profit_mult * daily_vol),
        entry_price * (1 + stop_mult * daily_vol),
    )


def run_rule_baseline(
    conn: sqlite3.Connection,
    *,
    start_date: str = "2025-06-01",
    end_date: str | None = None,
    rebalance_frequency: int = 5,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    position_pct: float = DEFAULT_POSITION_PCT,
    max_holding_days: int = 10,
    profit_mult: float = 2.0,
    stop_mult: float = 2.0,
    min_score: float = 68.0,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Backtest the current technical-rule proxy over a validation window."""
    end_date = end_date or _latest_spy_date(conn)
    all_dates = _trading_dates(conn, start_date, end_date)
    if len(all_dates) < 20:
        return {
            "ok": False,
            "error": f"Only {len(all_dates)} trading dates -- need at least 20",
            "method": RULE_BASELINE_METHOD,
        }

    rebalance_dates = all_dates[::rebalance_frequency]
    spy_return = _spy_return_pct(conn, start_date, end_date)
    portfolio_value = initial_capital
    cash = initial_capital
    open_positions: list[dict[str, Any]] = []
    trade_log: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    scoring_errors = 0

    for rebalance_date in rebalance_dates:
        still_open: list[dict[str, Any]] = []
        for pos in open_positions:
            rows = conn.execute(
                """
                SELECT date, CAST(close AS REAL) AS close
                FROM price_daily
                WHERE ticker=?
                  AND date > ?
                  AND date <= ?
                  AND close IS NOT NULL
                ORDER BY date
                """,
                (pos["ticker"], pos["entry_date"], rebalance_date),
            ).fetchall()

            closed = False
            for bar_idx, row in enumerate(rows):
                price_date = str(row[0])
                price = float(row[1])
                days_held = bar_idx + 1
                exit_price = _apply_exit_slippage(price, pos["direction"], "market")
                exit_reason = "time_expired"
                if _hit_target(price, pos["target"], pos["direction"]):
                    exit_price = _apply_exit_slippage(price, pos["direction"], "target_hit")
                    exit_reason = "target_hit"
                elif _hit_stop(price, pos["stop"], pos["direction"]):
                    exit_price = _apply_exit_slippage(price, pos["direction"], "stopped_out")
                    exit_reason = "stopped_out"
                elif days_held < max_holding_days:
                    continue

                trade_entry = _compute_exit(
                    pos,
                    exit_price,
                    price_date,
                    exit_reason,
                    days_held,
                )
                cash += _return_cash_on_close(pos, exit_price, trade_entry["pnl"])
                trade_log.append(trade_entry)
                closed = True
                break

            if not closed:
                still_open.append(pos)
        open_positions = still_open

        available_slots = max_positions - len(open_positions)
        if available_slots > 0:
            try:
                features = extract_features_for_universe(
                    conn,
                    as_of_date=rebalance_date,
                    tickers=tickers,
                    strict=True,
                    feature_columns=RULE_BASELINE_FEATURE_COLUMNS,
                )
                candidates = score_rule_candidates(features, min_score=min_score)
                held = {str(pos["ticker"]) for pos in open_positions}
                candidates = candidates[~candidates["ticker"].isin(held)].head(available_slots)
            except Exception as exc:
                scoring_errors += 1
                LOG.warning("Rule baseline scoring failed for %s: %s", rebalance_date, exc)
                candidates = pd.DataFrame()

            for _, pick in candidates.iterrows():
                ticker = str(pick["ticker"])
                if ticker in ML_CONTEXT_TICKERS or ticker.startswith("^"):
                    continue
                direction = str(pick["direction"])
                entry_price = _latest_close(conn, ticker, rebalance_date)
                if entry_price is None:
                    continue

                if direction == "long":
                    entry_price *= 1 + DEFAULT_SLIPPAGE_PCT / 100
                else:
                    entry_price *= 1 - DEFAULT_SLIPPAGE_PCT / 100

                position_value = portfolio_value * (position_pct / 100)
                shares = int(position_value / entry_price)
                if shares <= 0:
                    continue

                barriers = _position_barriers(
                    conn,
                    ticker=ticker,
                    direction=direction,
                    entry_date=rebalance_date,
                    entry_price=entry_price,
                    profit_mult=profit_mult,
                    stop_mult=stop_mult,
                )
                if barriers is None:
                    continue
                target, stop = barriers

                cost = shares * entry_price + DEFAULT_COMMISSION_PER_TRADE / 2
                if cost > cash:
                    continue
                cash -= cost
                open_positions.append({
                    "ticker": ticker,
                    "direction": direction,
                    "entry_date": rebalance_date,
                    "entry_price": round(entry_price, 2),
                    "shares": shares,
                    "target": round(target, 2),
                    "stop": round(stop, 2),
                    "rule_score": round(float(pick["score"]), 2),
                    "setup_family": str(pick.get("setup_family") or ""),
                    "setup_tier": str(pick.get("setup_tier") or ""),
                    "rule_reasons": list(pick.get("rule_reasons") or []),
                })

        pos_value = 0.0
        for pos in open_positions:
            current = _latest_close(conn, str(pos["ticker"]), rebalance_date)
            if current is not None:
                pos_value += _mark_to_market(pos, current)
        portfolio_value = cash + pos_value
        equity_curve.append({
            "date": rebalance_date,
            "portfolio": round(portfolio_value, 2),
            "positions": len(open_positions),
        })

    if not equity_curve:
        return {
            "ok": False,
            "error": "No rule-baseline equity curve produced",
            "method": RULE_BASELINE_METHOD,
        }

    final_value = float(equity_curve[-1]["portfolio"])
    total_return = (final_value - initial_capital) / initial_capital * 100.0
    wins = [trade for trade in trade_log if float(trade.get("pnl") or 0.0) > 0]
    losses = [trade for trade in trade_log if float(trade.get("pnl") or 0.0) <= 0]
    equity_series = pd.Series([float(row["portfolio"]) for row in equity_curve])
    max_dd = float(((equity_series / equity_series.cummax()) - 1).min() * 100.0)
    gross_win = sum(float(trade.get("pnl") or 0.0) for trade in wins)
    gross_loss = abs(sum(float(trade.get("pnl") or 0.0) for trade in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0
    long_trades = [trade for trade in trade_log if trade.get("direction") == "long"]
    short_trades = [trade for trade in trade_log if trade.get("direction") == "short"]

    summary = {
        "method": RULE_BASELINE_METHOD,
        "return_pct": round(total_return, 2),
        "total_return_pct": round(total_return, 2),
        "spy_return_pct": round(spy_return, 2),
        "alpha_vs_spy_pct": round(total_return - spy_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": len(trade_log),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "win_rate_pct": round(len(wins) / len(trade_log) * 100.0, 1) if trade_log else 0.0,
        "profit_factor": round(profit_factor, 2),
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "open_positions": len(open_positions),
        "scoring_errors": scoring_errors,
    }
    return {
        "ok": True,
        "method": RULE_BASELINE_METHOD,
        "return_pct": summary["return_pct"],
        "summary": summary,
        "equity_curve": equity_curve,
        "trade_log": trade_log[:100],
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "rebalance_frequency": rebalance_frequency,
            "max_positions": max_positions,
            "position_pct": position_pct,
            "max_holding_days": max_holding_days,
            "profit_mult": profit_mult,
            "stop_mult": stop_mult,
            "min_score": min_score,
            "tickers": tickers,
        },
    }
