# Pre-registered hypotheses, version 10 (Roy Mattox exposure ladder as a safety layer)

Registered 2026-10-31 (SGT), before any v10 rule was computed or run.
The git commit that adds this file is the registration record, and its sha256
is written into the config of every v10 ledger entry. Rules are frozen: a
change is a new version, and results are reported regardless.

v10 tests an exposure-control layer, not a strategy. It decides what share of
a book is invested (0 to 100%) from market conditions, on top of whatever is
traded. The rules are the ones Roy Mattox (@RoyLMattox) states in his own
posts, extracted with post ids in `~/dev/.research/traders/roy/stated_rules.md`.
A number with no source in his words is marked **registrant's choice** there
and below.

The question is whether the layer is a safety layer worth running: does it cut
the worst drawdown by a margin, by timing rather than by simply holding less,
without a worse risk-adjusted return than holding the same average amount all
the time? A return edge over buy-and-hold is not required and not claimed.

## What was looked at before registering

- Roy's posts (section sources in `stated_rules.md`). His replayed posted
  exposure (`scorecard_prelim.md` headline: on SPY 2021-2026, max drawdown
  -6.1% against -24.5%, return per unit of volatility 1.11 against 1.16 for a
  constant 43% mix). `work/exposure_daily.csv` and the exposure result files
  were not opened.
- `results-v7.md`: the regime filters that failed, including O'Neil
  distribution days on SPY volume (out of the market 67% of the time) and the
  10-month trend, and the finding that the price rules sold into the March
  2020 crash late and re-entered after the rebound.
- Data coverage only, with no v10 signal or return: SPY and QQQ have full
  daily bars with volume from 2010-01-04. `^GSPC` has rows only from
  2025-06-02. `^VIX` has no volume, and there is no Nasdaq Composite series.
  FRED DTB3 is loaded (v7 amendment 1).

## Data (all in `trader_koo_research.db`, a scratch copy)

- SPY and QQQ: total-return adjusted open and close from the bench loader,
  high and low on the same basis (`v2_run.load_prices`), quoted volume.
- The equal-weight universe: point-in-time S&P 500 members of each session,
  adjusted opens, as in v2 and v9.
- Cash: FRED DTB3 (percent), each value known from the next calendar day,
  used at the rate known at the previous close (as in v7).

## The ladder (signals at the close of session t, from data known by then)

All signals are computed on total-return adjusted SPY (registrant's choice:
an ETF drops on ex-dividend days, the index does not) with quoted SPY volume.

1. **Distribution day**: close down at least 0.20% on volume higher than the
   prior session (1714759891833741743).
2. **Rally attempt**: in a correction, the rally low is the lowest low since
   the correction began. Day 1 is the first session that closes up after it;
   a session whose low undercuts the rally low resets the count and is itself
   day 1 if it closes up (registrant's choice, O'Neil's definition).
3. **Follow-through day (FTD)**: day 4 or later of a rally attempt, close up at
   least 1.7% on volume higher than the prior session (2036229915293655114,
   1580652693437095937, 1637562686568382466). No latest day.
4. **Uptrend** starts on an FTD. It ends, back to a correction, on the first of:
   - a close below the low of that FTD session (1704953888057548954);
   - a distribution day on session 1, 2 or 3 after that FTD
     (1527458546186702855, 1541936852940324867, 1489064082430906371); the
     rally attempt then continues from its existing low and day count, so a
     later FTD can restart the uptrend;
   - the 21-session EMA of the close crossing below the 50-session SMA
     (1511866584264019969);
   - the start of a long-term downtrend (rule 6).
   When the uptrend ends on a close below the FTD low, an EMA/SMA cross or a
   long-term downtrend, a new correction starts: rally low = that session's
   low, no day 1 yet.
5. **Exposure in an uptrend**: 0.25 for the FTD, plus 0.25 while the close is
   above the 21-day EMA, plus 0.25 while the close is above the 50-day SMA,
   plus 0.25 while the 10-day EMA is above the 21-day EMA
   (1824220816713847269; the fourth step is registrant's choice from
   1544090137252052993 and 1551345717981085697). In a correction: 0.
6. **Long-term downtrend**: starts on the second consecutive weekly close
   below the 30-week SMA of weekly closes and ends on the second consecutive
   weekly close above it (1481815031192993795, 1552434627838623746). A weekly
   close is the last session of a calendar week. While it holds, exposure is
   capped at 0.25 (registrant's choice).
7. **Drawdown ladder** (1461123973727535111): the book's value at the close of
   t, against its peak since the FTD that started the current uptrend
   (registrant's choice: the peak resets at each such FTD). Drawdown of 5% or
   more: exposure x 0.5; 7% or more: x 0.25; 10% or more: x 0. The book's
   close value is its value at the open of t moved by the held exposure times
   the session's open-to-close return.
8. **Pace**: exposure rises by at most 0.25 per session (registrant's choice,
   "Progressive exposure", 1919164105450987866) and falls at once.
9. **Start**: indicators warm up from 2010-01-04. On the first session with
   all of them, the state is uptrend if the 21-day EMA is above the 50-day SMA
   (with no FTD low and the drawdown peak set there), else correction; the
   long-term flag starts from the weekly closes available.

## Hypotheses

| # | Book | Exposure | Buy-and-hold benchmark |
|---|---|---|---|
| H1 | SPY | the ladder, rules 1 to 9, drawdown measured on the SPY book | SPY, fully invested |
| H2 | the equal-weight universe | the same SPY signals (rules 1 to 6, 8, 9); rule 7 on the overlaid equal-weight book's own value | the equal-weight universe, fully invested |

Reported variants, outside the pass rule and the Holm family: H1 without the
drawdown ladder (rule 7 off); H1 with every signal computed on QQQ instead of
SPY (his "true barometer", 1527458546186702855), still trading SPY; and a
plain 50-day rule on SPY (exposure 1 when the close is above its 50-day SMA,
else 0) as a reference.

## Common rules

- **Execution**: exposure decided at the close of t is held from the open of
  t+1. Book returns are open-to-open. Over the interval from the open of t to
  the open of t+1, return = h x R + (1 - h) x cash - cost x |h - h_prev|.
  Here h is the exposure decided at the close of t-1 and R is the book's
  open-to-open return. Cash is the DTB3 rate known at the close of t-1
  divided by 252. The equal-weight R is the mean over the members of session
  t with prices at both opens.
- **Costs**: 10 bps on each change of exposure; stress run at 20 bps.
- **Window**: development, intervals from 2011-01-03 to the last session
  before the pinned seal (2023-06-12). The sealed window is not opened by
  this registration.
- **Constant mix**: the same book at a constant exposure equal to the rule's
  mean exposure over the window, rebalanced every session at no cost
  (registrant's choice; free rebalancing makes the bar harder for the rule).
- **Return per unit of volatility**: annualised mean daily return over
  annualised standard deviation, total return with no cash subtracted, as in
  the scorecard. **CAGR/maxDD**: CAGR over the absolute max drawdown.
- **Timing controls**: 1,000 circular shifts of the rule's own exposure path
  (as decided, after rule 7) by a random offset of at least 252 sessions, seed
  derived from the config. A shift keeps the average exposure and the number
  of changes and moves only the timing. Each shifted path pays the same costs.
  p = (1 + number of shifts whose max drawdown is as shallow as the rule's or
  shallower) / 1,001.

## Pass rule (all required, net of 10 bps)

| # | Criterion |
|---|---|
| a | Max drawdown at least 30% shallower than buy-and-hold: \|rule maxDD\| <= 0.7 x \|buy-and-hold maxDD\|. |
| b | Return per unit of volatility **and** CAGR/maxDD at least as high as the constant mix's. |
| c | Drawdown better than at least 95% of the timing controls: the Holm-adjusted p (over H1 and H2) below 0.05. |
| d | Criterion b holds within at least 60% of the full calendar years in the window (2011 to 2022, so 8 of 12). Within a year, volatility and drawdown are floored at 1% (registrant's choice, so an all-cash year does not divide by zero), and the mix is the full-window constant mix. |
| e | Criteria a and b still hold at 20 bps. |
| f | **The fastest crash.** From the SPY close of 2020-02-19 to the close of 2020-03-23 (intervals from the open of 2020-02-20 to the open of 2020-03-24), the rule's loss is at most 0.7 x buy-and-hold's loss over the same intervals. v7's lesson: a rule that misses the fastest crash fails, whatever its full-window numbers. |

No rubric score: `scoring-rubric-v1.md` grades a return edge, which a safety
layer does not claim. The output order is still reasoning, pros, cons, gates.

## What a pass means

A pass on a to f is a development result only. The sealed window stays closed
until the owner decides to open it. A layer that passes would at most become a
proposal for an observation-only shadow, which needs separate owner approval.
Nothing here touches production or the paper campaign.

## Outcomes reported regardless

For each of H1, H2 and the variants: CAGR, max drawdown, return per unit of
volatility, Sharpe over cash, CAGR/maxDD, mean exposure, share of sessions
invested, number of exposure changes, the same for buy-and-hold and the
constant mix, the year table, the March 2020 episode, every gate and the
Holm-adjusted p.
