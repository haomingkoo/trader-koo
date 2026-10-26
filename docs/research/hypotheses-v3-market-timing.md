# Pre-registered hypotheses, version 3 (market-timing rules on daily bars)

Registered 2026-10-27 (SGT), before any run. Sources: H1-H12 were read out of a
private newsletter studied by the owner ("Trader J", paid content, not quoted
here); H13 replicates claims published on algoquantsuccession.com/proof.
The newsletter period (2026) lies inside the sealed window, so 2026 data is not
used to choose or tune anything; development runs use 2010-01-04 to the pinned
seal (2023-06-12), and the sealed window is opened once with v1/v2 survivors.
Rules are frozen; a change is a new version.

## Data available

- `price_daily` in `trader_koo_research.db`: SPY 4,228 rows, QQQ 4,228, ^VIX 4,241 (all 2010-01-04 to 2026-10-23). DIA, IWM, ^GSPC, ^TNX, TLT, UUP, USO, ^VIX3M have only 352 rows from 2025-06-02, so they are not usable here.
- "SPX" in the author's wording is tested on SPY. ^VIX has 13 more rows than SPY; align on SPY sessions and report dropped rows.
- `adjustment_basis` is "unverified" on many rows: report which basis was used, and for gap rules note ex-dividend gaps (up to about 0.3% on SPY).
- FOMC dates for 2010 to 2025 are not in the DB (only a 2026 list in `catalyst_data.py`). H9 needs that list supplied and frozen before running.

## Common rules (all hypotheses)

- Signal at the close of session t, using data through t only. No execution, costs or position sizing: these are tests of whether a stated condition changes the distribution of later closes, not of strategies.
- Outcome is measured on closes only, t+k sessions later, strict inequality (an unchanged close is a miss), same convention as the scorecard.
- Events: one event per signal date; signals within 5 sessions of an earlier signal are dropped from the primary count (independent events) and kept for a sensitivity count. Minimum 30 independent events in the development window or the result is "insufficient", never a pass.
- Baselines (both reported, the matched one decides): (1) unconditional: the same outcome on every session in the window; (2) matched: the same outcome on sessions that share the confound (stated per hypothesis, for example "VIX in the same level tercile").
- Intervals: 95% moving-block bootstrap, 5-session blocks, 10,000 resamples, seed recorded. Multiple tests: Holm correction across all hypotheses registered together.
- Pass (all required): (1) development window: effect over the matched baseline of at least +5 percentage points with a Holm-adjusted interval excluding zero; (2) n_independent at least 30; (3) sealed window: same sign and at least half the development effect. A stated Trader J claim (for example "85%") is falsified if the upper interval bound lies below it, whatever the baseline result.
- Fail is reported with the same detail as a pass. Thresholds below are fixed; no grid search. Listed variants are pre-specified secondary checks, never selected after the fact.

## H1. VIX one-day drop is followed by an SPX down move

- Claim (stated by the author in March, April and July): after the VIX falls 15% or more in a day, SPX has a down day within the next 2 sessions.
- Why it might hold: a sharp vol drop reflects relief that is already priced and can precede a retrace as hedges are re-bought. Equally likely to be nothing.
- Signal: ^VIX close(t) / close(t-1) - 1 <= -0.15.
- Primary outcome: SPY close(t+2) < close(t).
- Secondary: at least one close-to-close down day in t+1, t+2 (closest to the author's wording; high base rate, so the matched baseline matters).
- Baselines: unconditional frequency of the same outcome; matched: sessions with a VIX one-day fall between -5% and -15%.
- Falsified if: primary effect over the matched baseline is below +5 points, or the interval includes zero after Holm, or the sealed window has the opposite sign. Note: the author's own 2026 record was 0 of 3, so a pass here would be a surprise and should be treated with care.
- Variants (pre-specified): threshold -12%, -20%.

## H2. VIX one-day jump is followed by an SPY bounce and a VIX pullback (the author's "85%")

- Claim: after a 20%+ one-session VIX rise, the author states an 85% chance of a bounce in SPY on the next one or two sessions, and a very high probability that the VIX pulls back in 1 to 2 sessions.
- Why it might hold: volatility spikes mean-revert; hedging demand peaks on the spike day.
- Signal: ^VIX close(t) / close(t-1) - 1 >= +0.20.
- Primary outcomes (two, both reported): (a) SPY close(t+2) > close(t); (b) ^VIX close(t+2) < close(t).
- Secondary: SPY close(t+1) > close(t); VIX close(t+1) < close(t).
- Baselines: unconditional frequency; matched: sessions with a VIX rise between +8% and +20%.
- Falsified if: (a) or (b) fails the common pass rule, or the 85% claim has an interval upper bound below 0.85 for outcome (a) (which is the more likely failure).
- Variants: +25%, +40% (he calls +40% a "superior" signal; expect very few events, report n).

## H3. Bollinger compression on VIX precedes a VIX rise within 2 to 3 sessions

- Claim: with VIX Bollinger Bands tightly compressed (often with a descending wedge, VIX near 15 to 16), VIX rises 5 to 15% within 2 to 3 sessions.
- Why it might hold: low realised range in vol tends to precede expansion, but compression also coincides with low VIX where upward mean reversion is expected anyway, which is the confound this test must control.
- Signal: 20-session SMA of ^VIX close with +/- 2 sample standard deviations; width = (upper - lower) / SMA; compressed when width(t) <= the 10th percentile of width over the trailing 252 sessions.
- Primary outcome: ^VIX close(t+3) / close(t) - 1 >= +0.05.
- Secondary: max ^VIX close over t+1..t+3 >= 1.10 x close(t).
- Baselines: unconditional frequency; matched: sessions in the same ^VIX level tercile (computed over the trailing 252 sessions) that are not compressed. The matched baseline decides.
- Falsified if: effect over matched baseline below +5 points or interval includes zero. Reporting requirement: in 2026 the author's repeated calls for a VIX breakout above 20 matched an always-say-up baseline, so show the always-up comparison too.
- Variants: width percentile 5th, 20th; horizon 2 and 5 sessions.

## H4. Compression precedes a large expansion within 10 sessions (direction unspecified)

- Claim (Sept to Oct): tightly compressed SPY and VIX bands mean a large move comes within 7 to 10 trading days.
- Signal: H3 compression on ^VIX AND SPY 20-session band width at or below its trailing-252 10th percentile.
- Primary outcome: within t+1..t+10, SPY has a close at least 3% away from close(t) in either direction.
- Baselines: unconditional frequency of a 3% excursion in 10 sessions; matched: sessions with the same trailing 20-session realised volatility tercile and no SPY compression.
- Falsified if: effect below +5 points or interval includes zero.
- Note: this tests whether compression predicts range expansion, not direction. The author gave no direction.

## H5. A close beyond a range confirms; a wick does not

- Claim: price must close beyond a level for a break to count, and intraday wicks or trend-line pokes are not confirmation; after a close-only break the move continues.
- Definitions: range = highest and lowest SPY close over t-20..t-1; qualifying only if (max close - min close) / mean close <= 5% (set here, not tuned) and the band has held at least 10 sessions.
- Signal A (close break): close(t) > range high (long) or < range low (short).
- Signal B (wick break): intraday high(t) > range high but close(t) <= range high (long), or the mirror for short.
- Primary outcome: sign of SPY close(t+5) - close(t) matches the break direction.
- Baselines: unconditional frequency of an up (or down) 5-session return; matched: sessions inside a qualifying band with no break.
- Falsified if: A does not beat its matched baseline by +5 points, or B is not at least 5 points weaker than A (the stated claim is the difference, not just A).
- Variants: SPY and QQQ separately, horizon 3 and 10.

## H6. Candle tails at extremes reverse over the next 1 to 3 sessions

- Claim: a long topping tail near highs is followed by a pullback in 1 to 2 sessions; a long bottoming tail near lows by a bounce.
- Definitions (set here, the author gives none): range = high - low, body = |close - open|. Topping tail: upper wick = high - max(open, close) >= 2 x body and >= 0.5 x range, close in the lower half of the day's range, and close(t) >= 0.98 x highest close of the last 60 sessions. Bottoming tail: lower wick >= 2 x body and >= 0.5 x range, close in the upper half, and close(t) <= 1.02 x lowest close of the last 60 sessions.
- Primary outcome: topping tail: SPY close(t+2) < close(t); bottoming tail: SPY close(t+2) > close(t). SPY and QQQ reported separately.
- Baselines: unconditional frequency; matched: sessions within the same 2% of the 60-session high (or low) with no qualifying tail.
- Falsified if: effect over matched baseline below +5 points after Holm, or fewer than 30 independent events per instrument per side. The author's 2026 scored record on tails was 54.5% against 49.5% (n=22), so expect a weak or null result.
- Variants: horizon 1 and 3; wick multiple 1.5 and 3.

## H7. SPY lower low with a VIX lower high precedes a bounce

- Claim: when SPY makes a new closing low but VIX does not make a new closing high, a relief rally follows in 1 to 3 sessions (a bullish divergence; the author called the 3/30 low this way).
- Signal: SPY close(t) is the lowest close of the last 10 sessions AND ^VIX close(t) < highest ^VIX close of the last 10 sessions.
- Primary outcome: SPY close(t+3) > close(t).
- Baselines: unconditional; matched: other sessions where SPY closes at a 10-session low and VIX also makes a 10-session closing high (no divergence). The claim is that the divergence adds information beyond "new low".
- Falsified if: the divergent group does not beat the non-divergent group by +5 points, or the interval includes zero.
- Variants: window 5 and 20.

## H8. Index above all short-term MAs: shorts do not pay

- Claim: when SPY closes above every one of its 5 DMA, 8 EMA, 10 DMA, 20 DMA, 21 EMA and 50 DMA, the author does not short or buy puts, even with divergences; there is no edge in puts.
- Signal: all six conditions true at close(t).
- Primary outcome: SPY close(t+5) < close(t) (a short would have worked).
- Baselines: unconditional frequency of a down 5-session close; matched: sessions below at least four of the six averages.
- Falsified if: the down frequency in the gate state exceeds the unconditional baseline by 3 points or more (the gate would then be removing a real edge for shorts); a result in the other direction supports the author's stand-aside rule but is not a signal to trade anything. Report SPY and QQQ.
- Note: this tests the author's stand-aside gate, which is the rule that dominates the author's behaviour.

## H9. Close below the 8 EMA and 5 DMA starts a decline

- Claim: after a strong run, an SPY close below the 8 EMA (with the 5 DMA) on higher volume is the trigger for a pullback toward the 21 EMA, and later closes are lower (staged puts).
- Signal: close(t) < 8 EMA(t) AND < 5 DMA(t) AND volume(t) > 20-session average volume, AND on t-1 all six conditions of H8 were true.
- Primary outcome: SPY close(t+5) < close(t); secondary t+10.
- Baselines: unconditional down-frequency; matched: sessions that close below the 8 EMA and 5 DMA without the preceding all-above state.
- Falsified if: effect below +5 points or interval includes zero. In 2026 the author's own calls with MA triggers hit 48.9% (n=47), so a null is the expected outcome.

## H10. The "FOMC flip"

- Claim: after a down FOMC day the next day often reverses higher ("FOMC Flip", stated 83% in 2026).
- Signal: t is an FOMC decision day (list must be supplied and frozen, 2010 to 2023) AND SPY close(t) / close(t-1) - 1 <= -0.005.
- Primary outcome: SPY close(t+1) > close(t).
- Baselines: unconditional next-day up frequency; matched: next-day up frequency after any non-FOMC day with a return of -0.5% or lower.
- Falsified if: effect below +5 points, or the 83% claim has an interval upper bound below 0.83. Expect fewer than 30 events in 2010 to 2023 (about 110 meetings): if so, record "insufficient" and do not register a verdict.
- Blocked until the FOMC date list is available.

## H11. Friday VIX decline

- Claim: since 2024 the VIX has fallen on most Fridays (the author states over 85% of them).
- Signal: Friday sessions.
- Primary outcome: ^VIX close(t) < close(t-1).
- Baselines: other weekdays' frequency of the same outcome.
- Window note: the claim is about 2024 onward, which is entirely inside the sealed segment. Development (2010 to 2023) tests the generic Friday effect; the 2024+ claim is checked once, in the sealed window.
- Falsified if: Friday frequency is not at least 5 points above the other-weekday frequency in development, or the 2024+ frequency has an upper bound below 0.85.

## H12. Unfilled gaps are filled

- Claim: small upside gaps (under 1.5%) are very likely to be filled (unquantified).
- Signal: open(t) > high(t-1) (up gap) or open(t) < low(t-1) (down gap), gap size under 1.5% of close(t-1).
- Primary outcome: within t..t+10, SPY trades back to close(t-1) (low(t+k) <= close(t-1) for an up gap; high >= close(t-1) for a down gap).
- Baselines: matched touch rate: on non-gap sessions, the frequency that price comes back within 10 sessions to a level the same distance below (above) the open. This isolates "gaps fill" from "price wanders".
- Falsified if: the fill frequency exceeds the matched touch rate by less than 5 points.
- Note: ex-dividend gaps on SPY are mechanical; exclude the 4 ex-dividend sessions per year and report the count.


## H13. Quiet-days trigger precedes a shallow pullback (daggercapital replication)

- Claim (published on algoquantsuccession.com/proof, own 15-year test): a trigger
  of at least two consecutive sessions with daily range under 0.60% is followed by
  (a) a close below the trigger-day close on day 8 or 9 (their figure: 44.2% vs
  30.1% matched) and (b) price reaching 2.36% below the day 4-5 pivot high within
  days 8-19 (their figure: 43.2% vs 35.7% at 2.124%, 38.4% vs 31.6% at 2.36%).
- Signal: SPY sessions where (high - low) / previous close < 0.0060 on t and t-1;
  the trigger date is the last session of the streak (their own convention); the
  alert is usable from the next session (their stated one-session lag).
- Pivot high: the highest high of sessions 4-5 after the trigger date.
- Outcomes: (a) close on session 8 or 9 < trigger-day close; (b) min low over
  sessions 8-19 <= pivot high x (1 - 0.0236).
- Baselines: matched by instrument and calendar month on non-trigger sessions,
  same outcome definitions; SPY and QQQ reported separately (SPX = SPY here).
- Falsified if: (a) or (b) fails the common pass rule, or our measured lift is
  below half of their published lift.

## Not registrable


- MOC imbalance to next open, dealer gamma, call walls, overnight/premarket retest, 60/90-minute opening range, scalps: no data in the DB.
- VIX/VVIX divergence, VVIX higher lows: no ^VVIX. VIX term structure: ^VIX3M only from 2025-06.
- Midterm-year top (5 to 10% correction before Nov 3 after September OPEX): 4 election cycles in 2010 to 2023, impossible to test with power. (The 2026 instance had not occurred by 10/23: SPY high 781.62 on 10/6 to close 759.97 on 10/23, -2.8%.)
- 9/11 anniversary, Memorial Day gap fade, July effect, Boxing Day: calendar frames with 13 or fewer observations each in the window, and the author's quoted win rates (85%, 88%) are unsourced.
- Oil, yields, DXY, USDJPY headwind stack: not available as long histories, and not a stated numeric trigger.
- Economic-data surprise tags: no consensus data.
- "Correction coming" from extension and divergence: no horizon, so not falsifiable as stated.

## Family size and expected multiplicity

12 hypotheses, 13 primary outcomes (H2 has two). Holm correction is applied across all primary outcomes. Variants are reported but excluded from the pass decision.
