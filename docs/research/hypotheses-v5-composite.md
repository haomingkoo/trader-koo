# Pre-registered hypotheses, version 5 (multi-signal composite)

Registered 2026-10-30 (SGT), before any v4 result was seen and before any v5
run. The git commit that adds this file is the registration record. Rules are
frozen: a change is a new version, and results are reported regardless.

The single-signal tests of v1 failed one at a time. Published work finds that
signals which fail at different times (quality, value, momentum, low risk,
shrinking share count) hold up better combined. The components are fixed here
from the v1 and v4 definitions, so the choice cannot follow the v4 results.

## Common rules

Identical to `hypotheses-v1.md` (universe, signal dates every 10th session,
next-open execution, long only, 20 names equal weight, top-40 buffer, 20-session
minimum hold, costs, benchmarks, walk-forward from 2011, pinned seal at
2023-06-12, pass criteria 1 to 4), with the v4 rule for "year ago" facts.

## H1. Five-signal composite

- **Claim**: a stock that ranks well on quality, value, momentum, low risk and
  shrinking share count together beats the market over the next 1 to 3 months.
- **Why**: each is a published premium; they are weakly correlated, so the
  average is steadier than any one of them.
- **Components** (cross-sectional z-scores, winsorised at plus or minus 3):
  1. quality: the v1 H1 score (mean of its three z-scores);
  2. value: earnings yield, EBIT / EV, as in v4 H1;
  3. momentum: total return from 252 to 21 sessions before the signal date;
  4. low risk: minus the standard deviation of the last 252 daily returns;
  5. share count change: minus the year-on-year change of the latest filed
     share count, split adjusted, as in v4 H2 signal 6.
- **Score**: mean of the available component z-scores; a name needs at least 4
  of the 5 (banks lack gross profit for quality). The count scored on 4 is
  reported.
- **Falsified if**: it fails the common pass rule.

## H2. Same composite, held longer

- **Claim**: the H1 edge, if any, survives slower trading, and lower turnover
  keeps more of it after costs.
- **Rule**: H1 score with review every 21 sessions and a 63-session minimum
  hold. Everything else as H1.
- **Falsified if**: it fails the common pass rule.
