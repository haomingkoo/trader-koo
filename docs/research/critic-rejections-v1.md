# Pre-registered measurement: critic rejections, version 1

Registered 2026-11-01 (SGT), before the evaluator existed and before any
outcome was computed. The git commit that adds this file is the registration
record. The rules are frozen: a change is a new version, and results are
reported whatever they show.

This is a measurement, not a hypothesis test. It asks what the trades the
paper critic rejected would have done, check by check, next to the trades it
approved, using each decision's own stop and target. It has no pass rule and
cannot change policy. Any change to critic thresholds needs a new registered
policy version, as `decisions.md` already requires.

Amendment 1, 2026-11-01, before the evaluator first ran on any data: rule 4
compared the report-date close with the reference price. For a filled
decision the reference is the next open plus slippage, so an overnight gap
above 2% would have been excluded as a basis mismatch. Rule 4 now compares it
with the report's own close.

## What was looked at before registering

- The critic code (`trader_koo/paper_trade/critic.py`), the decision ledger
  schema (`paper_candidate_decisions`, `paper_decision_sets`), and how
  `critic_outcome_json` is written (`campaign.persist_decision_set`).
- No production rows, no candidate outcomes, and no price windows after any
  decision were opened.

## Data

- One campaign per run, named by `--campaign-id`. Frozen v1 history is never
  mixed with v2 decisions.
- Decisions: `paper_candidate_decisions` rows of that campaign that belong to
  a sealed `paper_decision_sets` row with `is_canonical = 1`. The same
  report date, ticker and direction can appear in more than one canonical run;
  only the first by (`generated_ts`, `report_run_id`) is kept.
- In scope: rows whose `critic_outcome_json` is a non-empty object with a
  `critic_reasons` list and an `approved` flag. Outcomes without
  `critic_reasons` (critic infrastructure errors, fail-open) are counted and
  excluded. Rows with no `stop_loss` or `target_price` are counted and
  excluded.
- Prices: `price_daily` bars for the ticker dated strictly after the report
  date, read through a read-only connection (`file:...?mode=ro`). These are
  the same split-adjusted prices the paper path uses. Nothing is written.

## Rules

1. **Group.** Approved means `critic_outcome.approved` is true, whatever the
   later gates did (pending, duplicate, campaign inactive). Rejected means it
   is false. A check's group is every rejected decision whose
   `critic_reasons` contains `FAIL [<check>]`, so a decision with three failed
   checks counts once in each of the three. A check's sole-failure count is the
   subset where that check is the only `FAIL`.
2. **Direction.** `inputs_json.candidate.call_direction` when it is `long` or
   `short`. Otherwise the production function
   `trader_koo.paper_trade.decision.direction_from_row` applied to
   `inputs_json.candidate`, the same rule the policy used. The source of each
   direction is counted in the output. Anything other than long or short is
   excluded.
3. **Risk unit.** The reference price is `inputs_json.context.entry_price`
   when present, otherwise `inputs_json.candidate.close`, the price
   `decide_candidate` passed to the stop and target calculation. One R is
   |reference - stop_loss|. A long needs stop < reference < target and a short
   target < reference < stop; otherwise the row is excluded as a geometry
   mismatch.
4. **Basis check.** The `price_daily` close on the report date must be within
   2% of `inputs_json.candidate.close`, the close the report saw, or the row
   is excluded as a basis mismatch. This catches a split that rescaled history
   after the stop was stored. The report-date bar is used only for this check,
   never for the outcome.
5. **Entry.** The open of the scheduled NYSE session after the report date
   (`chronology.next_scheduled_session_after`). If the ticker has no bar on
   that date the row is excluded as having no entry bar. No later date is
   substituted.
6. **Window.** The entry bar and the ticker's next bars, 10 bars in all (the
   paper trade expiry). A window is closed only when all 10 bars exist; until
   then the row is reported as open, whatever the price did, so fast stops or
   targets do not enter the sample early. No bar is ever filled or
   interpolated. A window bar with `basis_status = 'unresolved'` excludes the
   row.
7. **Exit, first touch.** Bar by bar from the entry bar: if the open is at or
   beyond the stop, exit at the open; else if the open is at or beyond the
   target, exit at the open; else if the bar's range reaches the stop, exit at
   the stop (the stop wins when one bar reaches both); else if it reaches the
   target, exit at the target. With no touch, exit at the close of bar 10.
8. **Return.** Gross R = direction x (exit - entry) / one R. Net R subtracts
   10 bps of the entry price and 10 bps of the exit price, in R. Net R is the
   primary figure. A win is net R > 0.
9. **Uncertainty.** 95% percentile intervals from 10,000 bootstrap resamples
   with seed 20261101. Resampling is by report date (all of a date's decisions
   move together), because same-day candidates share one market. The
   difference between a check's group and the approved group uses independent
   resamples of each.
10. **Minimum n.** A group is read only when it and the approved group each
    have at least 30 closed windows. Below that the output labels it
    `insufficient` and gives no reading. At or above it the figures are still
    descriptive. Windows overlap and the sample is one campaign. There are no
    costs beyond rule 8 and no control for regime. Nothing here is a
    conclusion, an edge or a reason to change a threshold.

## Output

`python -m trader_koo.research.critic_rejections --db PATH --campaign-id ID
--out PATH.md` writes a Markdown table per group (approved, rejected, each
check) with n closed, n open, mean net R and its interval, win rate and its
interval, mean gross R, the difference against approved, and the excluded
counts by reason. It is to be run only on a copy of the production database,
never on the production volume.
