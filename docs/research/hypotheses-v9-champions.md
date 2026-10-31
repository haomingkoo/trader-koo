# Pre-registered hypotheses, version 9 (stated rules of US Investing Championship traders)

Registered 2026-10-31 (SGT), before any of these rules was computed or run.
The git commit that adds this file is the registration record, and its sha256
is written into the config of every v9 ledger entry. Rules are frozen: a change
is a new version, and results are reported regardless.

v9 tests the rules that verified US Investing Championship (USIC) traders state
in their own words: Christian Flanders, J Law, Sean Ryan and Martin Luk, plus
the two rule sets they say they took from Kristjan Kullamägi (Qullamaggie) and
Pradeep Bonde (Stockbee). The rule text comes from
`~/dev/.research/traders/swing_strategies.md` (each trader's sections 1 to 4
and 6, and section 7 "draft rule text") and
`~/dev/.research/traders/swing_candidates.md` (Qullamaggie's episodic pivot,
Stockbee's 4% burst, and the ranked shortlist). The named-trade lists in those
files and any per-trade analysis were **not read**, so no rule was tuned on a
trader's own winners.

Every number below names its source. A number marked **registrant's choice**
has no source in the traders' words and was fixed here, before any result. A
number marked **draft** comes from the draft rule text written by the source
reviewer, not from the trader; it is also a registrant's choice.

## What was looked at before registering

Only data coverage, no returns and no signal counts:

- `price_daily` starts 2010-01-04. An all-time high cannot be computed: the
  data does not reach back to any company's listing. H1a therefore uses the
  **52-week (252-session) closing high**, as the owner specified for this case.
  The longest high available (since 2010-01-04, at least five years at the
  first entry in 2015) is a registered variant.
- IWM has 352 rows, from 2025-06-02 only. QQQ and SPY start 2010-01-04. There
  is no Russell 2000 index series. H5 cannot be computed on the development
  window (below) and is skipped. Nothing is fetched outside the bench.
- `sec_fundamentals` has diluted EPS for 599 CIKs and net income for 609, filed
  from 2009-04-15. Revenue is spread over four tags (`Revenues`,
  `RevenueFromContractWithCustomerExcludingAssessedTax` from 2017,
  `SalesRevenueNet` to 2019, `RevenueFromContractWithCustomerIncludingAssessedTax`).
- `earnings_events` is the v8 table (thin before 2014, about 2,000 rows a year
  from 2015). The v8 timing and revision caveats apply unchanged.

## Universe limits (read before any result)

- **The universe is point-in-time S&P 500 members only.** These traders mostly
  trade liquid non-S&P growth names: small and mid caps, recent IPOs,
  biotech, crypto miners, themes with 5%+ daily ranges. Luk wants ADR above
  5%; Bonde says low-priced, low-float names move most. Most S&P 500 members do
  not move like that. A fail here does not refute the traders; a pass would not
  show that their skill is what is captured.
- **Survivorship.** Departed index members without price history are missing
  (190 of 745 window members in v8's H3 run had no prices), and the earnings
  table was fetched in 2026 for tickers that could be queried then. Both
  flatter long-only results. The equal-weight benchmark shares the bias; SPY
  does not.
- **Execution.** The traders enter intraday (opening-range highs, 1- to 5-minute
  triggers) with 1-3% stops, pyramid winners and size by discretion. Daily bars
  cannot copy any of that. Their USIC records come from self-selected,
  survivor-only standings in a strong small-cap momentum regime (2024-25).

## Common rules

- **Data**: the bench's as-of total-return-adjusted open, high, low and close
  (`v2_run.load_prices`) and its data-quality rules. Volume and the closes used
  for highs, quiet periods and dollar volume are the stored split-adjusted
  values (as v8 H2 and H3). Dollar volume = stored close x stored volume; its
  20-session median at the signal close sets entry priority.
- **Signals** use data through the close of the signal session t only, and only
  index members on t. Fills are on t+1 or later.
- **Window**: development only, up to the pinned seal (2023-06-12, last session
  2023-06-09). First entry 2015-01-02 for every primary test, the first full
  year of earnings data, so all sleeves share one window (H6 needs that).
  Indicators warm up on the 2010-2014 sessions. A trade still open on
  2023-06-09 is closed at that close and counted. The sealed window is opened
  once, only for a hypothesis that passes criteria 1 to 5, with every other
  survivor.
- **Costs**: 10 bps per side on every fill's notional; stress run at 20 bps.
  Cash earns 0%.
- **Fills on daily bars** (stated gap-fill rule):
  - A stop fills at the stop price. If the session opens at or below the stop,
    it fills at that open (gap through).
  - A buy-stop fills at max(open, trigger).
  - A profit target fills at max(open, target).
  - When one session could trigger both the stop and a target or a close rule,
    the stop is assumed to come first. On the entry session the stop is checked
    against the low (a buy-stop fill day included, so a fill and a stop on one
    bar count as a stop-out).
  - "First close below an average" exits at the next session's open (the close
    that triggers it is not known until the close). A missing open uses the
    next valid open, then the last close (as v2), each counted.
  - A listed spin-off exits at the close before the ex-date (`spinoffs.csv`).
- **Portfolio** (one per hypothesis, its own capital):
  - Size = risk per trade x equity at the prior close / stop distance (entry
    minus stop over entry), capped at the per-name cap, and at the room left
    under 100% gross exposure (no margin; positions valued at the prior close).
    A position smaller than 1% of equity is skipped (**registrant's choice**).
  - One position per stock. A signal on a held stock is skipped.
  - Same-day entries are taken in order of 20-session median dollar volume,
    highest first (Luk and J Law pick the highest dollar volume).
  - Exits at the open are booked before that session's entries; intraday and
    close exits after them.
  - Equity is marked at each close (adjusted close, carried over a missing bar).
- **Benchmarks**, same sessions: SPY (adjusted close to close) and the
  equal-weight universe (mean close-to-close return of the members of the
  previous session; primary).
- **Placebo** (the random baseline), 500 seeded draws per hypothesis: each real
  trade is replaced by a random index member on the same signal session (not
  the same stock) that enters at the open of the real fill session with the
  same stop distance, the same sizing rule, and the same exit rule applied to
  its own prices. Priority is the random stock's dollar volume. This keeps
  timing, count, risk and exposure and changes only the stock picked.
- **Primary test**: one-sided p = (1 + draws with a CAGR at or above the
  strategy's) / 501.
- **Multiple testing**: one Holm family of eight tests at a family-wise 5%:
  H1a, H1b, the H1a-vs-H1b comparison, H2, H3, H4, H6, H7. H5 is descriptive and
  is skipped. Variants are reported and do not enter the family or the decision.
- **Trade statistics** on the trades each portfolio took, and separately on
  every qualifying signal traded once with no portfolio limits ("all signals"):
  - R = net profit of the trade / its initial risk (shares x (entry - stop));
  - win rate (net R above 0), mean win and mean loss in R, mean R;
  - share of profit from the top 5% of trades: the net profit of the best
    ceil(5% of n) trades by net profit, over the total net profit, and over the
    gross profit of all winners;
  - holding sessions, exits by type.
- **Score**: `scoring-rubric-v1.md` on each portfolio, with the placebo
  percentile as "random portfolios". The pass rule decides; the score shows the
  distance.

### Pass rule (H1a, H1b, H2, H3, H4, H6, H7; all required, net of 10 bps)

1. Holm-adjusted placebo p below 0.05.
2. Portfolio CAGR above both the equal-weight universe and SPY.
3. Beats both in at least 60% of calendar years (2015 to 2023, 9 years, 2023
   partial; so at least 6).
4. Stress-run (20 bps) CAGR above both benchmark CAGRs.
5. Mean net R per taken trade above 0 (not applicable to H6).
6. Sealed window: CAGR above the equal-weight universe and mean net R above 0.

## H1a. Flanders: earnings gap into a new high, top relative strength

- **Source**: Flanders, "a gap ideally 10% or more, ideally into highs or
  all-time highs… on a gamechanging catalyst" (FL-B [30]); "the average RS of
  the names I traded last year was like 98" (FL-A [110]); codeable list "gap at
  least 10% on at least 3-5x volume" (swing_strategies.md, Flanders section 6).
- **Signal session t**: the earnings reaction session (v8 timing: BMO, the
  first session on or after `report_date`; AMC or unknown, the first after it).
- **Qualifies at the close of t** (all):
  - gap: open(t) / close(t-1) - 1 >= 10% (Flanders);
  - volume(t) >= 3 x the mean volume of the 50 sessions before t, all 50
    present (Flanders's lower bound; the 50-session window is Stockbee's, as v8);
  - stored close(t) >= the highest stored close of the 251 sessions before t,
    with all 252 closes present: a new 52-week closing high;
  - relative strength percentile >= 90 among members on t with a score.
    RS = 0.4 x r63 + 0.2 x r126 + 0.2 x r189 + 0.2 x r252, r_n the adjusted
    close return over n sessions to t (the public IBD-style weighting;
    weighting and the 90 cut are **registrant's choice**; Flanders's 98 is an
    average, not a cut-off).
- **Entry**: the open of t+1 (Flanders's alternative "buy the open"; his
  opening-range entry on t needs intraday bars). Skipped if that open is at or
  below the stop.
- **Stop**: the low of t (Flanders, Qullamaggie, Stockbee: "low of the day").
- **Exit** (Qullamaggie base rule, which Flanders uses: "sell half on days 3-5
  and trail the rest with the 10-day"):
  - sell half at the close of the 5th session of the trade (entry session is
    the 1st; day 5 of "3-5" is **registrant's choice**, as the draft), then move
    the stop on the rest to the entry price;
  - trail: exit at the next open after a close below the 10-session simple
    average of closes, while that average is above the current stop
    (Qullamaggie: "once they surpass your initial stop");
  - no holding through earnings (Flanders: "I never hold into earnings"): exit
    at the close of the session before the next reaction session;
  - maximum hold 120 sessions (**registrant's choice**, so the trail, not the
    clock, ends most trades); exit at the open of entry + 120.
- **Sizing** (Flanders's tiers): risk 0.5% of equity per trade ("starts at 0.5%
  or less"); 1.0% when the month-to-date portfolio return at the prior close is
  at least +8% ("grow it once up about 8% for the month"); 0.25% when it is
  below 0 ("cut size while losing"; 0.25% is **registrant's choice**). Monthly
  brake: no new entry for the rest of a calendar month once realised net profit
  of the trades entered that month reaches -5R, R = 1% of equity at the month's
  first prior close (Flanders: "-5 adjusted R", "1R = 1% of the account").
  Per-name cap 20% (**registrant's choice**; his average position was 17%).
- **Variants** (reported only): `H1a_since2010` (the highest stored close since
  2010-01-04 instead of 52 weeks, at least five years); `H1a_code33` (plus Code
  33, below); `H1a_flat` (flat 0.5% risk, no tiers or brake, to separate sizing
  from selection); `H1a_trail20` (20-session trail; Flanders's own exit study
  favoured longer trails); `H1a_spy50` (no entry unless SPY's close is above its
  50-session average, the shortlist's declared filter).

### Code 33 (variant only)

Minervini's term, cited by Flanders once (on NVDA): three straight quarters of
accelerating EPS growth, sales growth and profit margin. Point in time: only
facts filed before the signal session's date (`filed_date < t`, the bench's
filing cutoff), the latest value per period as known then.

- Discrete fiscal quarters: a duration fact of 80-100 days is a quarter; other
  quarters are the difference of two cumulative facts with the same start (six
  months minus three, nine minus six, the year minus nine: Q4 is never filed
  alone). EPS is `EarningsPerShareDiluted` (USD/shares; differenced EPS is
  approximate when the share count moved). Revenue is the first of
  `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`,
  `SalesRevenueNet`, `RevenueFromContractWithCustomerIncludingAssessedTax`
  with a value for that quarter. Net margin = `NetIncomeLoss` / revenue.
- With q0 the latest quarter and q0..q7 eight consecutive quarters (ends 80-100
  days apart): YoY growth g_k = (x_k - x_{k+4}) / |x_{k+4}|, undefined when
  x_{k+4} is 0. Code 33 holds when g0 > g1 > g2 > g3 for EPS and for revenue,
  and margin m0 > m1 > m2 > m3. Anything missing or undefined is "no".

## H1b. Qullamaggie: the same gap after a quiet period, with a positive surprise

- **Source**: Qullamaggie, "My 3 timeless setups" (qullamaggie.com): EP "gap of
  at least 10% on heavy volume", "a significant beat versus estimates",
  "preferably no major rally in the prior 3-6 months"; sizing "risk 0.25-1% per
  trade; positions 10-20%; no stock above 30% overnight".
- **Qualifies at the close of t**: the H1a gap and volume rules, plus
  - quiet: stored close(t-1) / stored close(t-63) - 1 <= 30% (the 63 sessions
    and 30% are the shortlist's, **registrant's choice**; his text gives no
    number);
  - positive surprise: `eps_actual` > `eps_estimate` (both present).
  - No new-high or relative-strength rule.
- **Entry, stop, exits**: as H1a (identical, so the head-to-head compares the
  entry filter only).
- **Sizing**: flat 0.5% risk (inside his 0.25-1%), per-name cap 20% (his
  "positions 10-20%").
- **Variant**: `H1b_spy50` (SPY above its 50-session average).

## H1a against H1b (head-to-head, in the Holm family)

The one real disagreement in the sources: Flanders wants the gap out of a base
at the highs; Qullamaggie wants it after neglect. Test: the difference in mean
net R per trade, H1a minus H1b, over **all signals** (each qualifying signal
traded once, no portfolio limits, entries from 2015-01-02). Two-sided p from
10,000 resamples of 5-session moving blocks of entry sessions (the bench's
`eventstats.bootstrap`, H1a as events, H1b as baseline). A signal in both sets
counts in both. A Holm-adjusted p below 0.05 says the entry filters differ; the
sign says which is better. It is not a pass for either rule.

## H2. J Law: first pullback to the 10/20-day band after a catalyst move

- **Source**: J Law, a momentum move "surging by 25%" (JL-CF [13]); catalysts
  are earnings "beating the market expectations" (JL-CF [12]); edges "the first
  pullback after a high-volume breakout, on low volume", "the 10/20 MA treated
  as a band", "RS holding on a market down day" (JL-CF [32], [82-83]; JL-METS);
  dollar volume "over $100 million" (JL-CF [27]); stops "3-5%", "just below the
  META"; maximum loss "0.3% of the portfolio" (JL-Z [16]); exit "below the 10/20
  MA with a buffer"; "zero-cost hold: at +70%, sell 58.8%" (JL-CF [48-53]).
- **Catalyst session e** (member, all):
  - close(e) / close(e-1) - 1 >= 25% (J Law), or the earnings reaction session
    with `eps_actual` > `eps_estimate` and open(e) / close(e-1) - 1 >= 10% (the
    10% gap is Flanders's and Qullamaggie's; the union is the draft's);
  - volume(e) >= 3 x the mean of the 50 sessions before e (draft);
  - 20-session median dollar volume at e >= $100M (J Law).
- **Pullback session t**, the first one with 5 <= t - e <= 30 (draft) such that:
  - a new 20-session closing high was made at some session in e+1..t-1
    (draft: "higher highs"); h is the session of the highest close in e..t-1,
    and the pullback is h+1..t (at least one session);
  - band: low(t) <= max(SMA10, SMA20) x 1.01 and close(t) >= min(SMA10, SMA20)
    x 0.98, simple averages of adjusted closes at t (1% and 2% are the draft's;
    J Law gives no number);
  - lighter volume: mean volume of t-2..t below the mean of e..e+3 (draft);
  - relative strength: at least one session in the pullback with SPY's close
    below its previous close, and on every such session the stock's
    close-to-close return is at least SPY's (requiring one down day is
    **registrant's choice**, so the check is never empty).
- **Entry**: the open of t+1 (J Law buys in the zone; his "no trades in the first
  15 minutes" makes the open approximate). **Stop**: the lowest low of the
  pullback x 0.99 (draft: "just below the META"). Skipped if the stop distance
  is above 5% (J Law: "3-5%") or the entry is at or below the stop.
- **Exits**: next open after a close below min(SMA10, SMA20) x 0.98 (J Law's
  buffer; 2% is the draft's); at +70% (high >= entry x 1.70) sell 58.8% at
  max(open, entry x 1.70) (J Law); the stop; maximum hold 120 sessions
  (**registrant's choice**).
- **Sizing**: risk 0.3% of equity (J Law), per-name cap 20% (**registrant's
  choice**).
- **Variant**: `H2_spy50`.

## H3. Sean Ryan: the "second move"

- **Source**: Sean Ryan (SR-A, SR-BI): a sideways stock, a sudden "+20% to
  +100% rally within a couple of weeks on big, rising volume", it "holds the new
  price and tightens into a small handle or rest", "buy the break of that rest's
  high"; "if the stock is down at the close on entry day, he sells it on the
  close"; risk "1% of the account per trade", size = 1% / stop distance;
  "moves the stop to breakeven after day one or two, or once up 10%"; sells on
  "a big-volume reversal day", "the third move up", "RSI divergence (price
  higher while RSI is lower)", "fading momentum after the first few weeks";
  holds "between two weeks and two months"; does not hold through earnings.
- **Setup at the close of t** (draft rule C, all):
  1. base: over t-60..t-21, max(high) / min(low) - 1 <= 25%;
  2. burst: max(close, t-20..t-6) / close(t-21) - 1 between 20% and 100%
     (Ryan), with mean volume of t-20..t-6 >= 1.5 x the mean of t-60..t-21
     (1.5 is the draft's);
  3. rest: over t-5..t, every close >= 0.85 x the burst high and max(high) /
     min(low) - 1 <= 12% (draft).
  Highs, lows and closes here are adjusted. A setup suppresses further setups
  on the stock for the next 5 sessions (its order is pending).
- **Entry**: buy-stop at max(high, t-5..t), live on t+1..t+5, filled at
  max(open, trigger) on the first session whose high reaches it; cancelled if
  none does. **Stop**: min(low, t-5..t).
- **Exits**, in the session order of the common rules:
  - day one: if the close of the fill session is below the fill price, exit at
    that close (Ryan). This assumes a market-on-close order placed on a
    near-close check, a small look-ahead that is stated;
  - breakeven: from the 3rd session, or from the session after a close at or
    above entry x 1.10, the stop is max(stop, entry) (Ryan; "day two" of "one
    or two" is **registrant's choice**);
  - next open after any of: a close below the 20-session average ("fading
    momentum", the draft's stand-in); RSI divergence; a big-volume reversal;
  - RSI divergence (exact): RSI is Wilder's 14-session RSI of adjusted closes.
    At session s, close(s) is above the highest close of s-19..s-1, and p is the
    latest session in s-60..s-5 whose close was also above its own prior 19
    closes; the exit fires when RSI(s) < RSI(p). No such p, no divergence. (The
    5-session separation and 60-session look-back are **registrant's choice**.)
  - big-volume reversal (exact): volume(s) >= 2 x the mean of s-50..s-1,
    high(s) is the highest high of s-19..s, close(s) < open(s) and close(s) <
    close(s-1) (thresholds **registrant's choice**);
  - no holding through earnings: exit at the close before the next reaction
    session;
  - maximum hold 60 sessions (his "two months", rounded up; **registrant's
    choice**).
  - "The third move" is **not modelled**: no exact definition exists in his
    words, and the divergence and trail rules sell into the same strength.
- **Sizing**: risk 1% (Ryan), per-name cap 30% (draft; his "3% stop gives a 30%
  position"; his own maximum is 50-75%), gross at most 100% (Ryan: "no margin").
- **Variants**: `H3_from2011` (first entry 2011-01-03; price-only rule),
  `H3_spy50`.

## H4. Stockbee: the 4% momentum burst

- **Source**: Bonde, stockbee.blogspot.com (2015-05): scan `c/c1>=1.04 and
  v>v1 and v>=100000`; "stocks should not be up 3 days in a row"; "narrow range
  day or negative day prior to breakout day"; better when "preceded by non
  momentum burst" (not up 8-40% in the prior 2-3 days); bursts last "3 to 5
  days". He publishes no stop or exit on that page.
- **Signal at the close of t** (all):
  1. stored close(t) / stored close(t-1) >= 1.04;
  2. volume(t) > volume(t-1) and volume(t) >= 100,000;
  3. not close(t-1) > close(t-2) > close(t-3) > close(t-4);
  4. range(t-1) below the median range of t-8..t-2 (range = high - low,
     adjusted; the 7-session median is the draft's), or close(t-1) < close(t-2);
  5. close(t-1) / close(t-4) - 1 < 8% (Bonde's lower bound).
  The draft's extra filters (close in the top quarter of the range, close above
  SMA50) are not Bonde's and are not used.
- **Entry**: the open of t+1; skipped if at or below the stop. **Stop**: the
  low of t (Stockbee's EP stop, applied here; **registrant's choice**, since he
  gives none for bursts). **Exit**: the close of the 5th session of the trade
  ("3 to 5 days"), or the stop.
- **Sizing**: risk 0.5% (the shortlist's common setting; **registrant's
  choice**), per-name cap 20%.
- The 20 bps stress run is central here: a 5-session hold turns over about 50
  times a year at full exposure.
- **Variants**: `H4_day3` (exit at the close of session 3), `H4_from2011`,
  `H4_spy50`.

## H5. Luk's IWM claim (descriptive, skipped)

- **Claim** (LUK-B [238-240]): since 2008, no QQQ correction of 15% or more
  began while IWM was leading QQQ.
- **Definition** (fixed before looking): a correction is a fall of at least 15%
  in QQQ's adjusted close from a running peak; it "begins" at that peak.
  "Leading" = IWM's 63-session return minus QQQ's above 0 at the peak session.
  Count corrections and how many began with IWM leading; one such correction
  refutes the claim.
- **Not run**: IWM has no rows before 2025-06-02 and the database starts in
  2010, not 2008. The bench does not fetch outside its own loaders, so this is
  skipped and stays outside the Holm family.

## H6. Sleeve blend (owner's request)

- **Claim**: one portfolio splitting capital equally over the H1a, H1b, H2, H3
  and H4 sleeves has smoother returns and smaller drawdowns than the sleeves,
  and still beats the market.
- **Rule**: each sleeve runs as its own primary portfolio. The blend holds 20%
  of capital in each; within a month the weights drift with sleeve returns;
  at the first session of each calendar month they reset to 20% each
  (monthly is **registrant's choice**). The daily blend return is the weighted
  sum of the sleeve returns. Moving capital between sleeves is a change of
  budget for new trades, not a trade, so no cost is charged for it (stated).
  No weight is fitted.
- **Placebo**: draw k of the blend is the blend of draw k of each sleeve.
- **Pass rule**: as above (criterion 5 not applicable). Also reported: the
  correlation matrix of the five sleeves' daily returns (Pearson, all sessions
  from 2015-01-02), the blend's Sharpe and maximum drawdown against each
  sleeve's, and the share of sessions each sleeve is invested. Diversification
  helps only if the sleeves are not the same bet.

## H7. Agreement of two rule families (owner's request)

- **Claim**: a stock flagged by at least two different traders' rules within a
  short time beats the market.
- **Families**: F1 = H1a or H1b signal (earnings gap), F2 = H2 pullback setup,
  F3 = H3 setup, F4 = H4 burst. "Fired" is the signal session t of each rule.
- **Qualifies**: on session t a family f fires on a stock, and a different
  family fired on the same stock in t-9..t (N = 10 sessions including t,
  **registrant's choice**, about two weeks). The triggering rule is f, the one
  that fired on t; if several fire on t, F1 before F2 before F3 before F4
  (**registrant's choice**). The trade is the triggering rule's own trade for
  that signal: its entry mechanics, stop and exits (an F1 trigger uses H1a's
  exits, which equal H1b's).
- **Sizing**: equal weight, 10% of equity per position, at most 100% gross (so
  at most 10 positions). No risk sizing.
- **Pass rule**: as above.

## Not registered (and why)

- Luk's pullback and flush-reclaim entries, Flanders's opening-range entries,
  pyramiding and same-day re-entries: they need intraday bars.
- Theme and group rules (Luk, J Law, Flanders): no point-in-time industry
  classification.
- Qullamaggie's breakout and Kell's EMA crossback (the shortlist's items 2 and
  4): not stated by the USIC traders themselves as their core rule; left for a
  later version.
- Flanders's NCFD breadth brake and Sean Ryan's index RSI market filter: index
  filters, not stock rules; the declared SPY > SMA50 variants stand in.

## Data caveats

- The v8 caveats on earnings data apply: yfinance estimates and actuals as
  stored in 2026 (possible revisions, a look-ahead risk for any rule using the
  surprise: H1b, H2's earnings leg); report timing labels that disagree with
  volume for about a third of events.
- Volume is taken as split-adjusted, like the stored prices.
- Adjusted prices include reinvested dividends, so a stop fixed at a past
  adjusted low sits slightly lower in raw terms after an ex-date (a fraction of
  a percent over these holding times).
- One development path, 2015 to mid-2023 (9 entry years). A deterministic test
  or a historical replay proves software behaviour, not market performance.
