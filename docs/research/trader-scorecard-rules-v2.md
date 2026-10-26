# Trader call scorecard rules, version 2

Registered 2026-10-27 (SGT), before any call is scored under it. Version 1
(`trader-scorecard-rules-v1.md`) stays in force for everything it covers; v2
only adds the two rule sets below. Results under v1 are not rescored or
replaced; v2 results are reported alongside them, labelled v2.

## A. Named-day direction calls

Applies to calls whose horizon names a single session ("tomorrow", "Friday",
"Monday", an explicit date) and whose direction is up or down for that session.
Outcome: the named session's close versus the previous session's close (a
green or red day), in addition to the v1 outcome measured from the reference
close. Both are reported for every such call; neither is chosen after the fact.

## B. Level-ladder calls

Applies to calls that state a trigger level and at least one target, with or
without an invalidation level (e.g. "hold 7650, targets 7655 / 7663, ❌ 7637").

- Prices: Yahoo Finance hourly bars (available about 730 days back), fetched
  once and frozen with a hash. Where 5-minute bars exist (last 60 days), they
  are used instead, and the bar size is recorded per call.
- Start: the first bar that begins after the post timestamp.
- Window: same session for intraday or 0-1 DTE calls; 5 sessions for weekly
  calls; 20 sessions for calls on daily or weekly charts. A call with no stated
  or implied timeframe uses 5 sessions.
- Trigger: if the call is conditional on reaching or holding the trigger level,
  the call is live only once a bar trades through the trigger in the stated
  direction; otherwise "not triggered".
- Outcome, first touch wins:
  - hit: first target reached before invalidation;
  - stopped: a bar closes beyond the invalidation level before the first
    target (a wick through it does not count, matching the trader's own
    close-beyond convention);
  - unresolved: neither within the window.
  If one bar reaches both, the outcome is "ambiguous" and counted separately.
- Further targets: report the share of hits that also reached target 2 and 3.
- Underlying only: option premiums, screenshots and percentages are not used.
- Baseline: for each scored call, the same trigger, target and invalidation
  distances applied in the opposite direction at the same start bar ("mirror
  call"); and a random-start baseline: the same distances from 200 random bars
  of the same instrument and session type in the same month (seeded). The
  trader shows skill only if the hit rate beats both.
- Reported: hit, stopped, unresolved, ambiguous and not-triggered counts; hit
  rate over resolved calls with a 95% Wilson interval; by instrument, timeframe
  and direction; and the self-reported outcomes compared with the measured
  ones, to show selective reporting.
