# Pre-registered hypotheses, version 2 (Saty / rose.trading setups)

Registered 2026-10-26, before any run. Source material: Saty Mahajan's free
indicators (satyland.com) and the rose.trading Saty playbook, studied the same
day. The playbook targets SPX 0DTE options intraday; these are daily-bar
translations for S&P 500 stocks, so they test the logic, not the playbook's own
claims (which show hit rates only, with no sample, costs or out-of-sample data).
Rules are frozen; a change is a new version.

## Common rules

- Universe: point-in-time S&P 500 members with prices (`sp500_membership`).
- Signals at the close of day t from data available then; entry at the next
  open; long only; 10 bps per side, stress at 20 bps.
- Event study, not a 20-name portfolio: every qualifying entry is a trade, equal
  dollar size, overlapping trades allowed; returns measured per trade and as an
  equal-weight daily portfolio of open trades.
- EMA = exponential moving average of the as-of total-return-adjusted close
  (the bench's as-of view), span n, standard pandas `ewm(span=n, adjust=False)`.
- ATR14 = Wilder ATR over 14 sessions from as-of OHLC.
- Benchmarks per hypothesis:
  1. placebo: random entries on the same stocks and days-of-entry count, same
     exit rule (1,000 seeded draws);
  2. the hypothesis's own state baseline named below (the stronger test);
  3. SPY and the equal-weight universe over the same holding windows.
- Periods: walk-forward by calendar year from 2011; the sealed final window is
  the same one pinned for v1 and is opened once, together with v1.
- Multiple testing: this family has 3 hypotheses; p-values are Holm-corrected
  across them.
- Pass (all, net of 10 bps): mean excess per trade over the state baseline > 0
  with Holm-corrected block-bootstrap p < 0.05; beats the placebo above its 95th
  percentile; positive in at least 60% of walk-forward years; still positive at
  20 bps; holds in the sealed window.

## S1. Flag into ribbon (pullback in an uptrend)

- Claim: dips to the 21 EMA in a stacked uptrend earn more over the next weeks
  than simply owning stocks in a stacked uptrend.
- Why: under-reaction and trend continuation; sellers of short-term dips supply
  liquidity at a discount.
- Entry at t: EMA8 > EMA21 > EMA48 > EMA200 and close > EMA200; low_t <= EMA21_t
  and close_t >= EMA21_t. One open trade per stock at a time.
- Exit: first close below EMA48, or 20 sessions, whichever is first; at the next
  open.
- State baseline: enter on random days when the same stack condition holds
  (without the pullback), same exit rule.

## S2. iVomy (ribbon flip then retest)

- Claim: after EMA8 crosses above EMA13 with close > EMA21, buying the first
  retest of EMA13 earns more than holding from the cross itself.
- Why: catches the start of an intermediate trend at a better price.
- Cross on day c: EMA8_c > EMA13_c, EMA8_{c-1} <= EMA13_{c-1}, close_c > EMA21_c.
  Entry day t: first day within 10 sessions after c with low_t <= EMA13_t and
  close_t > EMA13_t. No retest within 10 sessions: no trade (count reported).
- Exit: first close below EMA21, or 20 sessions; at the next open.
- State baseline: enter at the open after the cross day c, same exit rule.

## S3. Compression then expansion

- Claim: when a volatility squeeze ends in an uptrend (EMA21 > EMA48), the next
  10 sessions beat comparable uptrend days without a preceding squeeze.
- Why: volatility clustering and breakout drift.
- Compression on day t (third-party port of the Saty Phase Oscillator; Saty's
  own code was not readable): pivot = 21-session simple average of close;
  Bollinger(21, 2 std) upper <= pivot + 2*ATR14 and lower >= pivot - 2*ATR14.
  Expansion day = first day after at least one compression day where that
  condition no longer holds, with EMA21 > EMA48.
- Entry at the next open; exit after 10 sessions.
- State baseline: random uptrend days (EMA21 > EMA48) with no compression in the
  previous 20 sessions, same 10-session exit.

## Not registered (and why)

Golden Gate, opening-range breakout, end-of-day divergence and call-to-put
reversal are intraday or 0DTE setups our daily data cannot test honestly. Gap
fill under 0.25% is close to mechanical on daily ranges. Tweezer bottom and
broadening formations need judgement calls; candlestick patterns also have a
poor record under data-snooping correction.
