# Trader call scorecard rules, version 1

Registered 2026-10-26, before any call was scored. Applies to calls extracted
from a trader's dated posts (first use: "The Daily Market Roadmap" archive,
2026-03-06 to 2026-10-26). Post content stays private and is never committed;
only these rules and aggregate results are.

## Which calls are scored

- Only calls whose quote matches the post text verbatim (checked by code).
- Only calls with a direction in {up, down, volatility_up, volatility_down}
  and a horizon that maps to sessions below. "range" calls and calls without a
  horizon are counted and reported as unscored, never dropped silently.
- Conditional calls ("if SPY closes above X, then ...") are scored only when
  the condition was met within the horizon, from the session it was met;
  otherwise reported as "condition not met".

## Reference price

- Posts published after the US close (the evening roadmap) use that session's
  close as the reference. Posts published before the open use the prior
  session's close. Intraday posts use the prior session's close (conservative:
  we do not have the intraday price at posting time).

## Horizon mapping (XNYS sessions after the reference session)

| Stated horizon | Sessions |
|---|---|
| tomorrow, next session, Monday (next trading day) | 1 |
| 1-2 days, next couple of days | 2 |
| 1-3 days, few days | 3 |
| this week, next week, near-term, short-term | 5 |
| next 2 weeks, couple of weeks | 10 |
| weeks, coming weeks, this month | 20 |
| explicit date ("by Nov 7") | sessions to that date |
| anything else | unscored, listed |

## Outcome

- up: close at horizon > reference close. down: close at horizon < reference.
- Level target given: hit if the high (up) or low (down) reaches the target on
  any session within the horizon. Reported separately from direction.
- volatility_up / volatility_down: VIX close at horizon above / below the
  reference VIX close.
- Instruments: the stated ticker; SPX calls use ^GSPC; VIX uses ^VIX; yields
  use ^TNX; DXY uses DX-Y.NYB; oil uses BZ=F (Brent) or CL=F (WTI) as named.
  Prices from yfinance daily bars, fetched once and frozen with a hash.

## Baselines (the call must beat these to show skill)

1. Base rate: share of all windows of the same length in the same period where
   the instrument moved in the called direction (e.g. how often SPY was up over
   5 sessions from March to October 2026).
2. Always-same-direction: accuracy of always calling the trader's most common
   direction for that instrument.
- Report hit rate with a 95% Wilson interval, per instrument, per horizon, per
  call type (direction vs level target), and per stated confidence word.
- Calls on the same instrument made on the same day in overlapping posts count
  once (first post).

## Not claimed

A good scorecard measures published calls, not trading P&L; the trader states
no stops or sizes for most calls.
