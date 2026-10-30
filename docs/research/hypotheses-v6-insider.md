# Pre-registered hypotheses, version 6 (insider buying)

Registered 2026-10-30 (SGT), before any insider signal was computed or run.
The git commit that adds this file is the registration record. Rules are
frozen: a change is a new version, and results are reported regardless.

v1 to v5 used prices and filed financials, which every investor sees. Insider
purchases are a different source: officers and directors buying their own
stock with their own money, disclosed on SEC Form 4 within two business days.

## Data

`sec_insider_trades`, loaded by `trader_koo/research/insider.py` from the SEC
insider transaction data sets (2006 Q1 onward): Form 4 and 4/A, non-derivative,
transaction code P (open-market purchase) or S (open-market sale). A trade is
usable from the session after its filing date. Issuers are matched to tickers
through `sec_cik_map`. An amended filing (4/A) is a separate row; both are
counted, and the share of 4/A rows is reported.

## Common rules

Identical to `hypotheses-v1.md` (universe, signal dates every 10th session,
next-open execution, long only, 20 names equal weight, top-40 buffer, 20-session
minimum hold, costs, benchmarks, walk-forward from 2011, pinned seal at
2023-06-12, pass criteria 1 to 4), scored with `scoring-rubric-v1.md`.

- An insider is an officer or a director on the filing. Filings by holders
  who are only 10% owners (mostly funds) are left out.
- When fewer than 20 names qualify, the qualifying names are held equally;
  when none qualify the portfolio is in cash. Both counts are reported.
- The look-ahead guard is extended to the insider table: rows filed on or
  after the signal date are removed in the perturbed view, so any use of them
  changes the scores and fails the run.

## H1. Insider purchase clusters

- **Claim**: stocks where several insiders bought on the open market in the
  last 90 days beat the market over the next 1 to 3 months.
- **Why**: published (Lakonishok and Lee 2001; Jeng, Metrick and Zeckhauser
  2003). Insiders sell for many reasons but buy for one, and several buying at
  once is a stronger signal than one.
- **Window**: purchases filed in the 90 calendar days before the signal date.
- **Qualifies**: at least 2 distinct insiders with a purchase, and net insider
  dollar flow (purchase value minus sale value, shares times price, same
  window and insiders) above zero.
- **Score**: number of distinct buying insiders, plus net purchase value divided
  by market cap, scaled to below 1 by its percentile among qualifying names
  (so the buyer count ranks first).
- **Falsified if**: it fails the common pass rule.

## H2. Opportunistic purchases only

- **Claim**: the H1 effect comes from insiders who do not buy on a schedule.
- **Why**: published (Cohen, Malloy and Pomorski 2012): routine traders carry
  no information, opportunistic ones do.
- **Routine insider**: one who filed a purchase in the same calendar month in
  each of the three calendar years before the purchase year. Their purchases
  are dropped. Insiders with fewer than three years of filing history are
  treated as opportunistic, as in the paper.
- **Rule**: H1 on the remaining purchases.
- **Falsified if**: it fails the common pass rule.
