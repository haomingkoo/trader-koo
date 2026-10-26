# Research decision log

One line per decision: date, decision, reason. The owner made these decisions on
2026-10-25 and 2026-10-26. The exact day of each is not recorded in the
repository, so the date column gives the pair of days.

| date | decision | reason |
|---|---|---|
| 2026-10-25/26 | Research data lives off the production volume, in `~/dev/.research/trader_koo_research.db`. | The production volume is 4.4 GB with 1.4 GB free, and 2010+ history does not fit. Production is never written by the research tools. |
| 2026-10-25/26 | The bench is minimal: no market-impact model, and a deflated Sharpe ratio comes later. | A small bench that runs now beats a large one that is not validated. Costs are a flat 10 bps per side with a 20 bps stress run. |
| 2026-10-25/26 | The code restructure is paused. Code moves only when a change already touches it. | Moving files while research results are being produced would mix refactoring with evidence. |
| 2026-10-25/26 | The sealed window is the last 20% of sessions and is opened once, for all hypotheses together. | One look limits selection on the final data. The boundary is pinned in the trial ledger (2023-06-12). |
| 2026-10-25/26 | The equal-weight return of the same universe is the primary benchmark. SPY is second. | Departed index members have no price history, so the universe is survivor-biased. Equal weight shares that bias and SPY does not. |
| 2026-10-25/26 | Spin-offs use an exit rule in the bench (`bench/spinoffs.csv`). | A spun-off share is not a position the strategy chose. The rule is fixed before results so it cannot be tuned afterwards. |
| 2026-10-25/26 | Options data is stored as derived daily metrics, not raw chains. | Raw chains do not fit the storage budget and the hypotheses need only daily summary values. |
| 2026-10-25/26 | The rule-based debate is kept until a new policy version is registered. | Changing admission or debate rules mid-campaign is what made Campaign v2 hard to analyse. A change needs a new version. |
