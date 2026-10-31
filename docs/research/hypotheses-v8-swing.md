# Pre-registered hypotheses, version 8 (swing trades: earnings drift, earnings gaps, 52-week high)

Registered 2026-10-31 (SGT), before any of these rules was computed or run.
The git commit that adds this file is the registration record, and its sha256
is written into the config of every v8 ledger entry. Rules are frozen: a change
is a new version, and results are reported regardless.

The owner wants a swing edge: S&P 500 stocks, long only, paper, held 5 to 60
sessions. v1 to v7 found none (`RESEARCH_STATUS.md`). These three rules come
from peer-reviewed studies or from a practitioner's published rules. None of
them has been tested here. Every number below names its source. A number marked
"set here" has no source and was fixed before any result was seen.

## What was looked at before registering

Only data coverage and timing checks, no returns:

- `earnings_events` (yfinance, fetched 2026-10-26) has 27,134 rows for 578
  tickers. Coverage is thin before 2014 (fewer than 20 rows a year) and about
  2,000 to 2,200 rows a year from 2015. The yfinance request returns the last
  40 reports per ticker, which reaches back to about mid-2014.
- Session: 14,952 BMO, 11,822 AMC, 360 unknown. On 2015 to mid-2023 events,
  volume against its average peaks on the report day for 66% of BMO events and
  on the next session for 62% of AMC events. So most timings agree with the
  volume, and roughly a third may be wrong or ambiguous.
- EPS values are split-adjusted to today and rounded to cents (AAPL 2019 Q1
  shows 1.05, a quarter of the 4.18 printed then). The stored close in
  `price_daily` is on the same basis. On 14,737 events from 2015 to mid-2023,
  a cent of rounding is more than half the 90th-percentile surprise for 0.5%
  of events.
- 1,726 events from 2015 to mid-2023 are for tickers with no price history.
- No point-in-time industry classification exists in the database.

## Common rules

- **Universe**: point-in-time S&P 500 members with an open on the entry session
  (`sp500_membership`). Prices are the bench's as-of total-return-adjusted open,
  high, low and close (`v2_run.load_prices`). The bench's data-quality rules
  apply.
- **Costs**: 10 bps per side, stress run at 20 bps.
- **Window**: development window only, up to the pinned seal (2023-06-12). The
  sealed window is opened once, only for a rule that passes criteria 1 to 5 of
  its pass rule, together with any other survivors.
- **Benchmarks**: the equal-weight universe (primary) and SPY, both over the
  same holding window as each trade (open to open), as in v2.
- **Multiple testing**: one Holm family of three primary tests, one per
  hypothesis (H1, H2, H3), at a family-wise 5%. Variants are reported and do
  not enter the family or the decision.
- **Score**: `scoring-rubric-v1.md`. H3 is a portfolio run and is scored as
  written. H1 and H2 are scored on a calendar-time portfolio of their trades
  (below). Their "not luck" criterion uses the placebo percentile, because the
  random portfolios of the rubric do not exist for an event study. The pass
  rule below decides pass or fail; the score shows the distance.

### Event studies (H1 and H2)

- Every qualifying event is one trade of equal dollar size. Trades in
  different stocks may overlap. Two trades in one stock may overlap (as v2 S2
  and S3).
- First entry 2015-01-02 (first session of the first full year of earnings
  data). Entries whose maximum hold runs past 2023-06-09 are dropped and
  counted.
- Exits and substitutions follow `v2_trades` (missing exit open: next open,
  then last close; listed spin-offs exit at the close before the ex-date), each
  counted.
- **Placebo**: 1,000 seeded draws of random entry sessions on the same stocks,
  same count per stock, same exit rule (as v2).
- **State baseline**: the same entry and exit rule applied to the
  non-qualifying earnings events named per hypothesis.
- **Calendar-time portfolio** (for the score only): each session, equal weight
  across open trades, open-to-open returns, the round-trip cost charged on the
  entry and exit days. With no open trade it holds cash at 0%. The share of
  sessions invested is reported.
- **Bootstrap**: 10,000 resamples of 5-session moving blocks of entry sessions
  (as v2), seed derived from the run config.

### Pass rule for H1 and H2 (all required, net of 10 bps)

1. Mean per-trade excess over the equal-weight universe (same holding window)
   above 0, with Holm-adjusted bootstrap p below 0.05.
2. Mean net return above the 95th percentile of the placebo.
3. Mean excess over the equal-weight universe above 0 in at least 60% of entry
   years (2015 to 2023, 9 years, so at least 6).
4. Mean excess over the equal-weight universe still above 0 at 20 bps.
5. Mean net return above the state baseline (point estimate). The selection
   must add something over trading every earnings event.
6. Sealed window: mean excess over the equal-weight universe above 0 and at
   least half the development value.

## H1. Post-earnings-announcement drift, top surprise decile

- **Claim**: stocks with the largest positive earnings surprises keep beating
  the market for about 60 sessions after the report.
- **Sources**: Bernard and Thomas (1989), "Post-earnings-announcement drift:
  delayed price response or risk premium?", Journal of Accounting Research 27
  (supplement): the drift runs over the 60 trading days after the report and is
  largest in the top decile. Livnat and Mendenhall (2006), Journal of
  Accounting Research 44(1): the drift is stronger when the surprise is
  measured against the analyst consensus and scaled by price, with deciles cut
  on the previous quarter's surprises.
- **Event**: an `earnings_events` row with `eps_actual` and `eps_estimate`
  present.
- **Entry session** (first open after the report is public): BMO, the first
  session on or after `report_date`; AMC or unknown session, the first session
  after `report_date`. An unknown session is treated as AMC, which can only
  delay entry; the count is reported.
- **Surprise**: SUE = (`eps_actual` - `eps_estimate`) / stored close of the
  session before the entry session (the last close before the report was
  public; same split basis as the EPS).
- **Qualifies**: SUE at or above the 90th percentile of SUE over all eligible
  events (member, priced) whose entry session lies in the previous calendar
  quarter. A quarter without a previous quarter of events has no trades.
- **Exit**: the open 60 sessions after the entry (Bernard and Thomas's 60-day
  window; it is also the top of the owner's 5 to 60 band).
- **State baseline**: every other eligible event (below the cut-off), same
  entry and exit.
- **Variant** (reported only): exit at the open 20 sessions after entry.
- **Timing check** (reported only): H1 on AMC and unknown-session events only.
  A wrong AMC label can only make entry late, so this subset carries no timing
  look-ahead. Also reported: the count of BMO trades whose next session had
  higher volume than the entry session (a sign the report came after the
  close).

## H2. Earnings gap continuation ("episodic pivot")

- **Claim**: a stock that gaps up hard on earnings, on very heavy volume, and
  holds the gap into the close keeps rising for weeks.
- **Sources** (practitioners, no peer-reviewed study):
  - Kristjan Kullamagi (Qullamaggie), "My 3 timeless setups that have made me
    tens of millions" (qullamaggie.com): "Gap up 10%+"; "The stop is at the
    lows of the day"; "Trail your stop with the 10- or 20-day moving average
    once they surpass your initial stop."
  - Pradeep Bonde (Stockbee), "My process flow for Episodic Pivots (EP)"
    (stockbee.blogspot.com, July 2014): scan `v>3*avgv50` and `v>=300000`;
    in a reply, "volume is 3 times average 50 day volume calculated as of a
    day before today"; the stop "is normally low of the EP day".
- **Reaction session g**: BMO, the first session on or after `report_date`;
  AMC or unknown, the first session after it. EPS fields are not needed.
- **Qualifies at the close of g** (all):
  - gap: open(g) / close(g-1) - 1 >= 10% (Qullamaggie);
  - volume(g) >= 3 x the mean volume of the 50 sessions before g, and
    volume(g) >= 300,000 shares (Stockbee);
  - close in the upper half of the day's range: (close - low) / (high - low)
    >= 0.5 (set here; the owner's request, no source gives a number).
- **Entry**: the open of g+1. Daily bars cannot reproduce the sources'
  opening-range entry.
- **Stop**: the low of g. On every session s from g+1: if open(s) <= stop, exit
  at open(s); otherwise if low(s) <= stop, exit at the stop.
- **Trailing exit**: on a session where the 10-session simple average of
  closes is above the stop and the close is below that average, exit at the
  next open (the 10-day choice of the two the source gives).
- **Maximum hold**: exit at the open 60 sessions after entry (the owner's band).
- **State baseline**: every other reaction session of an earnings event of a
  member, same entry, stop (that day's low) and exits.
- **Variant** (reported only): the 20-session average as the trailing exit.

## H3. Nearness to the 52-week high

- **Claim**: stocks trading closest to their 52-week high beat the market over
  the next months.
- **Source**: George and Hwang (2004), "The 52-Week High and Momentum
  Investing", Journal of Finance 59(5): rank on price / highest price of the
  last 12 months, buy the top 30%.
- **Score** at review date t: stored close(t) / highest stored close of the
  252 sessions ending at t (stored closes are split-adjusted, so the ratio
  carries no dividend). Only names with all 252 closes present are scored.
- **Portfolio**: the bench `run` engine, 150 names (30% of about 500 members),
  equal weight, review every 21 sessions (monthly), no buffer
  (`buffer_n = 150`), no minimum hold. The paper holds 6 months with
  overlapping portfolios; the owner asked for a monthly rebalance, so this is a
  1-month holding. 200 random portfolios under the same rules.
- **Window**: 2011-01-01 to the seal (the 2010 sessions warm up the 252-session
  high), as v1.
- **Primary test p-value**: (1 + random portfolios with a higher CAGR) / 201.
- **Pass rule**: the rubric gates (beats SPY and equal weight in at least 60% of
  years; above the 95th percentile of the random portfolios; stress CAGR above
  both benchmarks) plus a Holm-adjusted p below 0.05. With 200 random
  portfolios that needs a random percentile of about 98.5 or more. Sealed
  window: CAGR above equal weight.

## Not registered (and why)

- Industry momentum (Moskowitz and Grinblatt 1999): there is no point-in-time
  industry or sector classification in the database. Current SIC codes would
  assign 2026 industries to 2011 companies, so it is not tested.

## Data caveats

- **Survivorship**: departed index members without price history are missing,
  which flatters long-only results. The earnings table was fetched in 2026 for
  tickers that could be queried then, so departed companies are missing from
  the earnings data even where prices exist. The run reports member-quarters
  with no earnings row.
- **Revisions**: yfinance's `eps_estimate` is the consensus as stored today,
  not a recorded snapshot from before the report, and `eps_actual` may be the
  later restated or "adjusted" figure. A surprise used at entry may differ from
  the one known on the day. This is a look-ahead risk for H1 that cannot be
  removed with this data. H2 uses only prices, volume, report date and session.
- **Report timing**: report dates and sessions come from yfinance timestamps;
  about a third disagree with the volume reaction (above). A wrong session can
  make H1 enter one session early, which is look-ahead. The run reports how
  many H1 trades fall on such events.
- **Volume**: stored volume is taken as split-adjusted, like the prices.
- Earnings data starts in mid-2014, so H1 and H2 cover 2015 to mid-2023 only
  (9 entry years). That is a short window for a yearly consistency test.
- A deterministic test or a historical replay proves software behaviour, not
  market performance.
