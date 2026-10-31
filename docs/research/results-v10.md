# Results, hypotheses version 10 (Roy Mattox exposure ladder as a safety layer)

Recorded 2026-10-31 for the rules frozen in `hypotheses-v10-exposure.md`
(registered in commit `a38698ad`, sha256
`f66e9aa4069a0678345a73319c56c0e19a9c5f71fd2cbe065867b4256af923a6`, the same
hash the run wrote into its ledger config). The rule extraction with post ids
is `~/dev/.research/traders/roy/stated_rules.md`. Every number below comes
from the bench trial ledger (`/Users/koohaoming/dev/.research/bench_trial_ledger.jsonl`),
from the report the same run printed, or from the audit named in "Checks".
A frozen copy of the ledger line is `ledger-v10.jsonl` (sha256
`3503e058f2c4d6cc90f9fc8053610119fff02a0b1236730585478cda3842f0b5`). This is a
research result on historical data. It is not a claim about Roy's account.

**Outcome: H1 and H2 both fail. Each passes the drawdown gate (a) and the
March 2020 gate (f). Each fails b, c, d and e. The sealed window was not
opened.**

## Run inputs

| field | value |
|---|---|
| database | a copy of `trader_koo_research.db` in the session scratchpad, opened read-only |
| seal | `2023-06-12`, read from the ledger; last interval from the open of 2023-06-08 to the open of 2023-06-09 |
| code | `87c04b9a` (clean tree) |
| trial | `f40ac446dae5`, seed 609074412, params hash `244dbcecdace` |
| window | interval starts 2011-01-03 to 2023-06-08; indicators warm from 2010-07-30 |
| controls | 1,000 circular shifts per hypothesis, offset at least 252 sessions |
| command | `python -m trader_koo.research.bench.v10_run --db <copy> --ledger ~/dev/.research/bench_trial_ledger.jsonl --json-out <file>` |

The command ran once against the real ledger. A smoke run before it wrote to
a throwaway ledger copy with its output discarded, to catch a crash without
seeing a result. No rule or parameter was changed after a result was seen.

## Pass/fail

| gate | H1 (SPY) | H2 (equal weight) |
|---|---|---|
| a. max DD at least 30% shallower than buy-and-hold | pass (-15.6% vs -32.0%) | pass (-15.9% vs -38.0%) |
| b. return/vol and CAGR/maxDD at least the constant mix's | FAIL (0.18 vs 0.93; 0.06 vs 0.44) | FAIL (0.20 vs 0.91; 0.07 vs 0.37) |
| c. drawdown beats 95% of timing shifts (Holm) | FAIL (beats 70.5%, Holm p 0.591) | FAIL (beats 70.1%, Holm p 0.591) |
| d. b holds in at least 60% of years | FAIL (1 of 12) | FAIL (1 of 12) |
| e. a and b at 20 bps | FAIL | FAIL |
| f. March 2020 loss at most 70% of buy-and-hold's | pass (+0.1% vs -30.2%) | pass (+0.1% vs -36.1%) |
| verdict | **fail** | **fail** |

## Numbers (development window, net of 10 bps per exposure change)

**H1, SPY book**

| series | CAGR | max DD | vol | return/vol | Sharpe over cash | CAGR/maxDD | mean exposure |
|---|---|---|---|---|---|---|---|
| ladder, 10 bps | +0.93% | -15.6% | 6.4% | 0.18 | 0.05 | 0.06 | 25% |
| ladder, 20 bps | +0.19% | -17.7% | 6.3% | 0.06 | -0.07 | 0.01 | 25% |
| buy-and-hold SPY | +12.43% | -32.0% | 16.7% | 0.78 | 0.74 | 0.39 | 100% |
| constant 25% mix | +3.87% | -8.8% | 4.2% | 0.93 | 0.74 | 0.44 | 25% |

Invested on 33% of sessions with 266 exposure changes. Timing shifts: max DD
5th/50th/95th percentile -32.1% / -19.2% / -9.4%.

**H2, equal-weight universe book**

| series | CAGR | max DD | vol | return/vol | Sharpe over cash | CAGR/maxDD | mean exposure |
|---|---|---|---|---|---|---|---|
| ladder, 10 bps | +1.10% | -15.9% | 6.6% | 0.20 | 0.08 | 0.07 | 24% |
| ladder, 20 bps | +0.48% | -18.1% | 6.5% | 0.11 | -0.02 | 0.03 | 24% |
| buy-and-hold equal weight | +12.93% | -38.0% | 17.9% | 0.77 | 0.72 | 0.34 | 100% |
| constant 24% mix | +3.91% | -10.5% | 4.3% | 0.91 | 0.72 | 0.37 | 24% |

Invested on 33% of sessions with 273 exposure changes. Timing shifts: max DD
5th/50th/95th percentile -38.3% / -19.4% / -10.0%.

**Variants (reported, outside the pass rule)**

| variant | CAGR | max DD | return/vol | CAGR/maxDD | mean exposure | mix return/vol | mix CAGR/maxDD | DD beats shifts | years b holds |
|---|---|---|---|---|---|---|---|---|---|
| H1 without the drawdown ladder | +0.91% | -15.7% | 0.17 | 0.06 | 25% | 0.93 | 0.44 | 74.2% | 1 of 12 |
| H1 with QQQ signals | +1.81% | -15.4% | 0.30 | 0.12 | 32% | 0.89 | 0.42 | 78.2% | 2 of 12 |
| SPY 50-day rule | +2.67% | -24.9% | 0.30 | 0.11 | 72% | 0.80 | 0.39 | 80.3% | 1 of 12 |

## Reasoning

The ladder avoided the March 2020 crash, as designed. It lost on everything
else. Its worst drawdown (-15.6%) came from
repeated small losses rather than from a crash. A constant 25% holding of SPY
fell only 8.8% at worst over the same window. Its drawdown also does not beat
its own exposure path moved in time (70th percentile), so the timing added no
drawdown protection beyond holding less.

The audit of the state machine shows why. SPY closed up 1.7% or more on higher
volume on 2.0% of sessions. After an uptrend ended in a calm market, no
follow-through day could come. From 2017-04-19 until 2018-02-06, SPY had no
session up 1.7%. All of 2019 had three such sessions, none on higher volume
(2018-12-26 was day 1 of the rally attempt). So the ladder held cash through
both rallies: 2017 +4.5% and 2019 +2.1%, against +5.7% and +9.4% for the
constant mix. In choppy markets the other half of the rule fired. Of the 32
uptrends started by an FTD, 15 lasted 3 sessions or less. 11 of those ended on
a distribution day right after the FTD alone. A distribution day (down 0.2% on higher SPY volume) occurs on
22% of sessions. Each false start paid costs and a loss, which shows in 2011,
2014, 2015 and 2022.

The drawdown ladder almost never acted. Removing it changes CAGR by 0.02 points
and max DD by 0.1 point. A likely reason (inferred from the variant, not traced
session by session) is that the market exits cut exposure before the book
falls 5% from its post-FTD peak.

Roy's posted exposure (scorecard_prelim.md, 2021-2026) drew down 6.1% on SPY
and beat 99.6% of its shifts. His written rules, run mechanically over
2011-2023, do not reproduce that. His posts show re-entries this rule cannot
make, such as pilot positions in single stocks, Nasdaq signals, breakouts and
judgement on volume. The windows also differ, and the 2011-2019 bull market is
the period that hurts this rule most.

**Pros**

- March 2020: +0.1% against -30.2% for SPY and -36.1% for equal weight.
- Max drawdown about half of buy-and-hold's on both books (-15.6% vs -32.0%,
  -15.9% vs -38.0%).

**Cons**

- Return per unit of volatility is 0.18 against 0.93 for the constant mix (H1)
  and 0.20 against 0.91 (H2).
- The constant mix at the same average exposure has a shallower drawdown
  (-8.8% vs -15.6% on SPY).
- The timing does not beat its own shifts (Holm p 0.591 for both).
- b holds in 1 of 12 years (2020 only).
- At 20 bps, CAGR falls to +0.19% (H1) with 266 exposure changes in 12.4 years.
- With QQQ signals (his "true barometer") the result is no better (0.30 vs 0.89).

## Checks

- Smoke run first, output discarded, against a ledger copy. The official run
  then ran once.
- Buy-and-hold SPY CAGR over the window is 12.43%. v7 gave 12.41% on nearly the
  same window with a different execution split.
- State-machine audit (a scratchpad script, not in the repo): uptrend segments
  with the trigger that ended each one, and every SPY session from 2017-04-19
  to 2020-03-31 up 1.7% or more, with its volume direction. It showed no stuck
  state. The long flat spells are the 1.7%-on-higher-volume rule doing what it
  was registered to do.
- Unit tests (`tests/test_research_v10.py`, 17) cover the distribution and FTD
  definitions, the day-4 rule, undercut resets, each uptrend exit, the
  long-term cap, the weekly flag, the pace, the drawdown ladder with its reset,
  execution timing, costs, shifts, the year floor and the gates.

## Limits

- Signals use SPY's ETF volume, as registered. O'Neil and Roy use index or
  exchange volume, which the database does not hold before 2025.
- Several rule posts are truncated, the drawdown ladder among them. A re-entry
  rule he may have written is not on disk.
- The equal-weight universe is point-in-time S&P 500 members without departed
  names, which flatters both the book and its benchmark.
- A development-window fail on written rules does not measure Roy's own
  trading. It shows that these rules, run without his judgement, do not make a
  safety layer.

## What this means for a shadow

Not worth running as a live shadow. The layer passed only the crash gate. On
every risk-adjusted measure, holding a quarter of the book all the time did
better, and the timing showed no edge over chance. A shadow would watch a rule
that this window already rejects.
