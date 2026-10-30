# Pre-registered hypotheses, version 4 (published stock-selection rules)

Registered 2026-10-30 (SGT), before any of these were run on the research bench.
The git commit that adds this file is the registration record. Rules are frozen:
a change is a new version with a new file, and results are reported regardless.

These are published rules with long outside track records, chosen because the
owner's own rules (v1 to v3) produced no survivor. Where the research database
lacks an input the published rule needs, the adaptation is stated here, before
any run, and the result is reported as a test of the adapted rule, not of the
original.

## Data limits that force adaptations

- No current assets or current liabilities, so Piotroski's current-ratio signal
  and Greenblatt's net working capital cannot be computed.
- No net property, plant and equipment, so Greenblatt's tangible capital cannot
  be computed.
- Gross profit is filed by about half the companies (318 of 612 CIKs), so
  Piotroski's gross-margin signal is dropped rather than computed for half.
- No sector field, so financials and utilities (excluded by Greenblatt) cannot
  be removed by sector. Companies without the inputs below drop out, and the
  count is reported.

## Common rules (all hypotheses)

Identical to `hypotheses-v1.md` (universe, signal dates every 10th session,
next-open execution, long only, 20 names equal weight, top-40 buffer, 20-session
minimum hold, 10 bps per side with a 20 bps stress run, SPY / equal-weight /
200 random portfolio benchmarks, walk-forward by calendar year from 2011, the
pinned seal at 2023-06-12, and pass criteria 1 to 4). The sealed window is
opened once, only for hypotheses that already pass criteria 1 to 3.

- When fewer than 20 names qualify, the qualifying names are held equally; when
  none qualify the portfolio is in cash. Both counts are reported.
- Point in time: SEC facts filed before the signal date (usable from the next
  session), prices through the signal date, market cap as in v1.
- "Year ago" for a flow is the TTM ending 350 to 380 days before the current
  TTM's period end (as in v1 H2). For a balance-sheet item it is the instant
  fact whose period end is 350 to 380 days before the current one, using only
  filings before the signal date.

## H1. Magic Formula, adapted (Greenblatt 2005)

- **Claim**: good businesses at cheap prices beat the market.
- **Why**: published 1988-2004 results; investors underprice steady, high-return
  businesses that look dull.
- **Inputs**: EBIT = TTM operating income. EV = market cap + long-term debt
  (latest instant) - cash (latest instant); EV must be positive.
- **Earnings yield** = EBIT / EV.
- **Return on capital (adapted)** = EBIT / (total assets - cash). Greenblatt
  uses net working capital plus net fixed assets, which the data lacks.
- **Score**: minus the sum of the two ranks (rank 1 = highest), so the best
  combined rank scores highest. Ties keep rank order by earnings yield.
- **Excluded**: names missing any input, including names that file no
  long-term debt tag (counted and reported; not treated as zero debt).
- **Falsified if**: it fails the common pass rule.

## H2. Piotroski F-score, 7-signal adaptation, among value stocks (Piotroski 2000)

- **Claim**: among cheap stocks, those with improving fundamentals beat the
  ones with deteriorating fundamentals and the market.
- **Why**: published 1976-1996 result; cheap stocks are neglected, so filed
  improvements are priced slowly.
- **Value set**: top quintile of book-to-market (equity / market cap, equity
  above zero) within the universe on the signal date.
- **Signals** (1 point each, 7 maximum):
  1. ROA = TTM net income / total assets > 0;
  2. TTM operating cash flow > 0;
  3. ROA higher than its year-ago value;
  4. TTM operating cash flow > TTM net income;
  5. long-term debt / total assets lower than its year-ago value (a company
     that files no long-term debt tag in either period scores 1, the published
     rule's "no increase in leverage"; one that files it in only one period is
     excluded);
  6. latest filed share count not above the year-ago count after split
     adjustment;
  7. asset turnover (TTM revenue / total assets) higher than its year-ago value.
- **Dropped**: current ratio and gross margin (see Data limits).
- **Score**: F + 0.5 x (book-to-market percentile within the value set), so a
  higher F always ranks first and book-to-market breaks ties.
- **Excluded**: names outside the value set and names missing any input other
  than the long-term debt case above (counted).
- **Falsified if**: it fails the common pass rule.

## H3. Trend Template with earnings growth (Minervini 2013)

- **Claim**: stocks in a confirmed stage-2 uptrend with growing earnings keep
  outperforming over the following months.
- **Why**: published as Minervini's selection rule (US Investing Championship
  results); trends persist when backed by earnings.
- **Price rules** (all required, total-return-adjusted closes, sessions):
  1. close > 150-session and 200-session moving averages;
  2. 150-session average > 200-session average;
  3. 200-session average higher than it was 21 sessions earlier;
  4. 50-session average > 150-session and 200-session averages;
  5. close > 50-session average;
  6. close at least 30% above the 252-session low;
  7. close within 25% of the 252-session high;
  8. 252-session total return at or above the 70th percentile of the universe.
- **Earnings rule**: TTM diluted EPS growth year on year at least 25%
  (Minervini's stated minimum), computed as in v1 H2.
- **Score**: 252-session total return among names passing every rule.
- **Excluded**: names with fewer than 253 sessions of closes or no EPS growth
  value (counted).
- **Falsified if**: it fails the common pass rule.

## Not in this version

- Faber's 10-month market filter needs a cash overlay in the portfolio engine;
  it is deferred to a later version rather than approximated.
- Roy Mattox's exposure ladder: his posts are not fully read, so the rules
  cannot yet be stated precisely enough to register.
