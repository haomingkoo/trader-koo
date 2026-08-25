# Research status

This file states what the repository's current evidence supports. It is not a
performance claim.

## Backtest status

- The next-open execution and accounting framework is implemented and covered
  by deterministic tests.
- One copied-local-database descriptive study ran with 50 selected setup calls,
  four closed trades, one traded signal date, and 11 daily observations. It
  returned -0.2066% net, but explicitly failed causal-validity and
  decision-eligibility gates. It is not a qualified backtest.
- The three-challenger tournament ran only its data audit. It stopped
  `blocked_before_validation`; all challenger metrics and the split are null,
  no winner was selected, and the sealed held-out data was not opened.
- Release replay evidence uses a synthetic contract fixture. It verifies code
  and accounting behavior, not trading performance.
- Campaign v2 remains inactive until separately approved after qualified
  research evidence exists.

The checked-in challenger artifact is immutable historical evidence. Its
negative result is useful: the fixed current-universe dataset lacks verified
total-return basis, point-in-time membership, and complete price-revision
lineage, so the runner correctly refuses to produce performance metrics.

## Supported language

It is accurate to say that Trader Koo has a fail-closed backtest framework and
that its current evidence gates prevented an invalid tournament from being
reported as a result.

Do not claim alpha, profitability, a challenger winner, held-out performance,
generalization, risk-adjusted improvement, or an LLM contribution to returns.
