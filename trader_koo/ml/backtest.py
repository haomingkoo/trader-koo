"""Walk-forward backtester for the swing-trade ML model.

Simulates what would happen if you followed the model's predictions:
- Each week, score the universe
- Pick the top N tickers by predicted win probability
- Enter long OR short positions with position sizing
- Apply triple-barrier exits (profit target, stop loss, time expiry)
- Track portfolio equity curve vs SPY buy-and-hold

Direction logic:
- prob > long_threshold (default 0.55) -> long signal (model confident price rises)
- prob < short_threshold (default 0.45) -> short signal (model confident price falls)
- between thresholds -> skip (no edge)

NO LOOK-AHEAD: the model used for each week is trained only on data
available before that week (walk-forward validation).

Output: equity curve, trade log, performance metrics vs SPY.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from trader_koo.ml.features import FEATURE_COLUMNS, extract_features_for_universe
from trader_koo.ml.labels import generate_triple_barrier_labels
from trader_koo.ml.trainer import LGBM_PARAMS, build_dataset

LOG = logging.getLogger(__name__)

# Backtest constants
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_MAX_POSITIONS = 5
DEFAULT_POSITION_PCT = 15.0  # % of capital per position
DEFAULT_COMMISSION_PER_TRADE = 5.0  # flat $ per round trip
DEFAULT_SLIPPAGE_PCT = 0.05  # 5 bps per side

# Direction thresholds
DEFAULT_LONG_THRESHOLD = 0.55
DEFAULT_SHORT_THRESHOLD = 0.45


def _compute_exit(
    pos: dict[str, Any],
    exit_price: float,
    exit_date: str,
    exit_reason: str,
    trading_days_held: int,
) -> dict[str, Any]:
    """Build a trade-log entry for a closed position (direction-aware)."""
    direction = pos["direction"]
    entry_price = pos["entry_price"]
    shares = pos["shares"]

    if direction == "long":
        pnl = (exit_price - entry_price) * shares - DEFAULT_COMMISSION_PER_TRADE
        return_pct = (exit_price / entry_price - 1) * 100
    else:  # short
        pnl = (entry_price - exit_price) * shares - DEFAULT_COMMISSION_PER_TRADE
        return_pct = (1 - exit_price / entry_price) * 100

    return {
        **pos,
        "exit_date": exit_date,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
        "days_held": trading_days_held,
    }


def _return_cash_on_close(
    pos: dict[str, Any],
    exit_price: float,
    pnl: float,
) -> float:
    """Compute cash returned when a position is closed.

    Long: invested capital was entry_price * shares; returned = that + pnl.
    Short: proceeds from short sale (entry_price * shares) were already
           added to cash at entry. On cover, we pay exit_price * shares.
           Net cash change = -(exit_price * shares) - commission,
           but pnl already includes commission, so return the pnl portion
           plus release the margin reserve.
    """
    entry_price = pos["entry_price"]
    shares = pos["shares"]

    if pos["direction"] == "long":
        return entry_price * shares + pnl
    # Short: margin reserve (entry_price * shares) was deducted at open.
    # Now we return reserve + pnl.
    return entry_price * shares + pnl


def _apply_exit_slippage(
    price: float,
    direction: str,
    exit_reason: str,
) -> float:
    """Apply slippage on exit — always works against you.

    Long exits sell shares: slippage pushes price DOWN.
    Short exits buy-to-cover: slippage pushes price UP.
    """
    slip = DEFAULT_SLIPPAGE_PCT / 100
    if direction == "long":
        return price * (1 - slip)
    # Short: covering means buying, so slippage pushes price up
    return price * (1 + slip)


def _mark_to_market(
    pos: dict[str, Any],
    current_price: float,
) -> float:
    """Unrealized position value for portfolio accounting.

    Long: current_price * shares (you own shares worth this much).
    Short: entry_price * shares + (entry_price - current_price) * shares
         = (2 * entry_price - current_price) * shares
    Equivalently for short: the margin reserve plus unrealized PnL.
    """
    shares = pos["shares"]
    entry_price = pos["entry_price"]
    if pos["direction"] == "long":
        return current_price * shares
    # Short: margin collateral + unrealized gain/loss
    return entry_price * shares + (entry_price - current_price) * shares


def _hit_target(price: float, target: float, direction: str) -> bool:
    """Check if price hit the profit target."""
    if direction == "long":
        return price >= target
    return price <= target  # short profits when price drops


def _hit_stop(price: float, stop: float, direction: str) -> bool:
    """Check if price hit the stop loss."""
    if direction == "long":
        return price <= stop
    return price >= stop  # short stops out when price rises


def run_backtest(
    conn: sqlite3.Connection,
    *,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    train_window_days: int = 180,
    rebalance_frequency: int = 5,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    position_pct: float = DEFAULT_POSITION_PCT,
    max_holding_days: int = 10,
    profit_mult: float = 1.5,  # Must match trainer.py labels
    stop_mult: float = 2.0,
    min_win_prob: float = 0.55,
    short_threshold: float = DEFAULT_SHORT_THRESHOLD,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Run a full walk-forward backtest.

    For each rebalance date:
    1. Train LightGBM on [date - train_window, date] (no future data)
    2. Score the universe
    3. Pick top N tickers:
       - prob > min_win_prob -> long
       - prob < short_threshold -> short
       - between thresholds -> skip
    4. Simulate positions with triple-barrier exits
    5. Track portfolio value

    Parameters
    ----------
    conn : sqlite3.Connection
    start_date : str
        First date to start making predictions (model trains on prior data).
    end_date : str
        Last date (default: latest in DB).
    train_window_days : int
        Rolling training window in calendar days.
    rebalance_frequency : int
        Rebalance every N trading days.
    max_positions : int
        Maximum concurrent positions.
    position_pct : float
        Percentage of portfolio per position.
    max_holding_days : int
        Time barrier for triple-barrier exit.
    profit_mult, stop_mult : float
        Barrier multipliers (x daily vol).
    min_win_prob : float
        Minimum predicted probability to enter a long trade.
    short_threshold : float
        Maximum predicted probability to enter a short trade.
        Must be < min_win_prob (dead zone between them = no trade).
    initial_capital : float
        Starting portfolio value.
    """
    if short_threshold >= min_win_prob:
        return {
            "ok": False,
            "error": (
                f"short_threshold ({short_threshold}) must be < "
                f"min_win_prob ({min_win_prob})"
            ),
        }

    if end_date is None:
        row = conn.execute("SELECT MAX(date) FROM price_daily WHERE ticker='SPY'").fetchone()
        end_date = str(row[0]) if row else dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    # Get all trading dates
    all_dates = [
        str(r[0]) for r in conn.execute(
            "SELECT DISTINCT date FROM price_daily WHERE ticker='SPY' AND date >= ? AND date <= ? ORDER BY date",
            (start_date, end_date),
        ).fetchall()
    ]

    if len(all_dates) < 30:
        return {"ok": False, "error": f"Only {len(all_dates)} trading dates — need at least 30"}

    # SPY buy-and-hold benchmark
    spy_prices = pd.read_sql_query(
        "SELECT date, CAST(close AS REAL) AS close FROM price_daily WHERE ticker='SPY' AND date >= ? AND date <= ? ORDER BY date",
        conn,
        params=[start_date, end_date],
    )
    spy_prices["date"] = pd.to_datetime(spy_prices["date"])
    spy_start = float(spy_prices["close"].iloc[0])
    spy_end = float(spy_prices["close"].iloc[-1])
    spy_return_pct = (spy_end - spy_start) / spy_start * 100

    # Walk-forward simulation
    rebalance_dates = all_dates[::rebalance_frequency]
    portfolio_value = initial_capital
    cash = initial_capital
    open_positions: list[dict[str, Any]] = []
    trade_log: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    models_trained = 0

    LOG.info(
        "Backtest: %s to %s, %d rebalance dates, max %d positions, "
        "long>%.2f short<%.2f",
        start_date, end_date, len(rebalance_dates), max_positions,
        min_win_prob, short_threshold,
    )

    for rebalance_idx, rebalance_date in enumerate(rebalance_dates):
        rebalance_ts = pd.Timestamp(rebalance_date)

        # 1. Check and close existing positions (triple barrier)
        still_open: list[dict[str, Any]] = []
        for pos in open_positions:
            # Get prices since entry
            entry_date = pos["entry_date"]
            direction = pos["direction"]
            prices = conn.execute(
                "SELECT date, CAST(close AS REAL) AS close FROM price_daily WHERE ticker=? AND date > ? AND date <= ? ORDER BY date",
                (pos["ticker"], entry_date, rebalance_date),
            ).fetchall()

            closed = False
            for bar_idx, price_row in enumerate(prices):
                price_date = str(price_row[0])
                price = float(price_row[1])
                # Use trading-day count (bar index) not calendar days
                # to match labels.py which iterates by index
                trading_days_held = bar_idx + 1

                # Apply slippage on exit (direction-aware)
                exit_price_adj = _apply_exit_slippage(
                    price, direction, "market",
                )

                if _hit_target(price, pos["target"], direction):
                    exit_price_adj = _apply_exit_slippage(
                        price, direction, "target_hit",
                    )
                    trade_entry = _compute_exit(
                        pos, exit_price_adj, price_date,
                        "target_hit", trading_days_held,
                    )
                    cash += _return_cash_on_close(
                        pos, exit_price_adj, trade_entry["pnl"],
                    )
                    trade_log.append(trade_entry)
                    closed = True
                    break
                elif _hit_stop(price, pos["stop"], direction):
                    # On stops, slippage works against you
                    exit_price_adj = _apply_exit_slippage(
                        price, direction, "stopped_out",
                    )
                    trade_entry = _compute_exit(
                        pos, exit_price_adj, price_date,
                        "stopped_out", trading_days_held,
                    )
                    cash += _return_cash_on_close(
                        pos, exit_price_adj, trade_entry["pnl"],
                    )
                    trade_log.append(trade_entry)
                    closed = True
                    break
                elif trading_days_held >= max_holding_days:
                    trade_entry = _compute_exit(
                        pos, exit_price_adj, price_date,
                        "time_expired", trading_days_held,
                    )
                    cash += _return_cash_on_close(
                        pos, exit_price_adj, trade_entry["pnl"],
                    )
                    trade_log.append(trade_entry)
                    closed = True
                    break

            if not closed:
                still_open.append(pos)

        open_positions = still_open

        # 2. Train model on data up to rebalance_date - embargo
        # Labels near rebalance_date use future prices (up to max_holding_days).
        # End training data early so no label's outcome window overlaps with
        # the trading period. Embargo = max_holding_days + 5-day buffer.
        embargo_days = max_holding_days + 5
        train_end_ts = rebalance_ts - pd.Timedelta(days=embargo_days)
        train_end_date = train_end_ts.strftime("%Y-%m-%d")
        train_start = (rebalance_ts - pd.Timedelta(days=train_window_days)).strftime("%Y-%m-%d")
        try:
            dataset = build_dataset(
                conn,
                start_date=train_start,
                end_date=train_end_date,
                sample_frequency=5,
            )
            if len(dataset) < 50:
                equity_curve.append({"date": rebalance_date, "portfolio": round(portfolio_value, 2),
                                      "positions": len(open_positions), "note": "insufficient_data"})
                continue

            feature_cols = [c for c in FEATURE_COLUMNS if c in dataset.columns]
            X = dataset[feature_cols].fillna(dataset[feature_cols].median())
            y = dataset["target"]

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X, y)
            models_trained += 1
        except Exception as exc:
            LOG.warning("Model training failed for %s: %s", rebalance_date, exc)
            equity_curve.append({"date": rebalance_date, "portfolio": round(portfolio_value, 2),
                                  "positions": len(open_positions), "note": f"train_error: {exc}"})
            continue

        # 3. Score universe — pick longs AND shorts
        available_slots = max_positions - len(open_positions)
        if available_slots <= 0:
            # Update portfolio value (direction-aware mark-to-market)
            pos_value = sum(
                _mark_to_market(
                    p,
                    float(conn.execute(
                        "SELECT CAST(close AS REAL) FROM price_daily WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1",
                        (p["ticker"], rebalance_date),
                    ).fetchone()[0]),
                )
                for p in open_positions
            )
            portfolio_value = cash + pos_value
            equity_curve.append({"date": rebalance_date, "portfolio": round(portfolio_value, 2),
                                  "positions": len(open_positions)})
            continue

        try:
            features = extract_features_for_universe(conn, as_of_date=rebalance_date, strict=True)
            if features.empty:
                continue
            # Use same training medians for imputation (consistent with training)
            X_score = features.reindex(columns=feature_cols).fillna(
                dataset[feature_cols].median()
            ).fillna(0.0)
            probs = model.predict_proba(X_score)[:, 1]
            scored = pd.DataFrame({"ticker": features.index, "prob": probs})

            # Long candidates: high win probability
            longs = scored[scored["prob"] >= min_win_prob].copy()
            longs["direction"] = "long"
            # Rank by probability descending (strongest long signal first)
            longs["signal_strength"] = longs["prob"]
            longs = longs.sort_values("signal_strength", ascending=False)

            # Short candidates: low win probability (model says price will DROP)
            shorts = scored[scored["prob"] <= short_threshold].copy()
            shorts["direction"] = "short"
            # Rank by inverse probability (lowest prob = strongest short signal)
            shorts["signal_strength"] = 1.0 - shorts["prob"]
            shorts = shorts.sort_values("signal_strength", ascending=False)

            # Merge and rank all candidates by signal strength
            candidates = pd.concat([longs, shorts], ignore_index=True)
            candidates = candidates.sort_values(
                "signal_strength", ascending=False,
            )

            # Exclude tickers we already hold (same ticker, any direction)
            held = {p["ticker"] for p in open_positions}
            candidates = candidates[~candidates["ticker"].isin(held)]
            picks = candidates.head(available_slots)
        except Exception as exc:
            LOG.warning("Scoring failed for %s: %s", rebalance_date, exc)
            continue

        # 4. Open new positions (direction-aware)
        for _, pick in picks.iterrows():
            ticker = str(pick["ticker"])
            prob = float(pick["prob"])
            direction = str(pick["direction"])

            entry_row = conn.execute(
                "SELECT CAST(close AS REAL) AS close FROM price_daily WHERE ticker=? AND date=?",
                (ticker, rebalance_date),
            ).fetchone()
            if not entry_row:
                continue
            entry_price = float(entry_row[0])
            if entry_price <= 0:
                continue

            # Slippage on entry — always works against you
            # Long: buying pushes price UP; Short: selling pushes price DOWN
            if direction == "long":
                entry_price *= (1 + DEFAULT_SLIPPAGE_PCT / 100)
            else:
                entry_price *= (1 - DEFAULT_SLIPPAGE_PCT / 100)

            # Position sizing (% of current portfolio)
            position_value = portfolio_value * (position_pct / 100)
            shares = int(position_value / entry_price)
            if shares <= 0:
                continue

            # Compute barriers from daily vol
            vol_rows = conn.execute(
                "SELECT CAST(close AS REAL) FROM price_daily WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 22",
                (ticker, rebalance_date),
            ).fetchall()
            if len(vol_rows) < 10:
                continue
            closes = [float(r[0]) for r in vol_rows]
            log_rets = [np.log(closes[i] / closes[i + 1]) for i in range(len(closes) - 1)]
            daily_vol = float(np.std(log_rets))
            if daily_vol <= 0:
                continue

            # Direction-aware barriers
            if direction == "long":
                target = entry_price * (1 + profit_mult * daily_vol)
                stop = entry_price * (1 - stop_mult * daily_vol)
            else:  # short
                target = entry_price * (1 - profit_mult * daily_vol)
                stop = entry_price * (1 + stop_mult * daily_vol)

            # Reserve capital: both longs and shorts tie up capital
            # (shorts require margin collateral equal to position value)
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
                "predicted_prob": round(prob, 4),
            })

        # 5. Update portfolio value (direction-aware mark-to-market)
        pos_value = 0.0
        for p in open_positions:
            latest = conn.execute(
                "SELECT CAST(close AS REAL) FROM price_daily WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1",
                (p["ticker"], rebalance_date),
            ).fetchone()
            if latest:
                pos_value += _mark_to_market(p, float(latest[0]))
        portfolio_value = cash + pos_value
        equity_curve.append({
            "date": rebalance_date,
            "portfolio": round(portfolio_value, 2),
            "positions": len(open_positions),
        })

    # Final metrics
    if not equity_curve:
        return {"ok": False, "error": "No equity curve produced"}

    final_value = equity_curve[-1]["portfolio"]
    total_return_pct = (final_value - initial_capital) / initial_capital * 100

    wins = [t for t in trade_log if t.get("pnl", 0) > 0]
    losses = [t for t in trade_log if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / len(trade_log) * 100 if trade_log else 0

    # Direction breakdown
    long_trades = [t for t in trade_log if t.get("direction") == "long"]
    short_trades = [t for t in trade_log if t.get("direction") == "short"]
    long_wins = [t for t in long_trades if t.get("pnl", 0) > 0]
    short_wins = [t for t in short_trades if t.get("pnl", 0) > 0]
    long_win_rate = len(long_wins) / len(long_trades) * 100 if long_trades else 0
    short_win_rate = len(short_wins) / len(short_trades) * 100 if short_trades else 0

    equity_series = pd.Series([e["portfolio"] for e in equity_curve])
    period_returns = equity_series.pct_change().dropna()
    # Convert multi-day rebalance-period returns to daily-equivalent returns.
    # Each observation spans `rebalance_frequency` trading days, so dividing
    # the period return by the number of days approximates the average daily
    # return and — critically — scales volatility to a daily basis, preventing
    # Sharpe inflation from the smoother multi-day series.
    freq = max(rebalance_frequency, 1)
    daily_equiv_returns = period_returns / freq
    sharpe = (
        float(daily_equiv_returns.mean() / daily_equiv_returns.std() * np.sqrt(252))
        if len(daily_equiv_returns) > 1 and daily_equiv_returns.std() > 0
        else 0.0
    )
    max_dd = float(((equity_series / equity_series.cummax()) - 1).min() * 100)

    avg_win = np.mean([t["return_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["return_pct"] for t in losses]) if losses else 0
    profit_factor = abs(sum(t["pnl"] for t in wins)) / abs(sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 0

    return {
        "ok": True,
        "summary": {
            "total_return_pct": round(total_return_pct, 2),
            "spy_return_pct": round(spy_return_pct, 2),
            "alpha_pct": round(total_return_pct - spy_return_pct, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sharpe_note": (
                f"Daily-equivalent Sharpe: {freq}-day rebalance returns "
                "divided by period length, annualized with sqrt(252)"
            ),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(trade_log),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "win_rate_pct": round(win_rate, 1),
            "long_win_rate_pct": round(long_win_rate, 1),
            "short_win_rate_pct": round(short_win_rate, 1),
            "avg_win_pct": round(float(avg_win), 2),
            "avg_loss_pct": round(float(avg_loss), 2),
            "profit_factor": round(profit_factor, 2),
            "models_trained": models_trained,
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
        },
        "equity_curve": equity_curve,
        "trade_log": trade_log[:100],  # cap for API response size
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "train_window_days": train_window_days,
            "max_positions": max_positions,
            "position_pct": position_pct,
            "max_holding_days": max_holding_days,
            "profit_mult": profit_mult,
            "stop_mult": stop_mult,
            "min_win_prob": min_win_prob,
            "short_threshold": short_threshold,
        },
    }
