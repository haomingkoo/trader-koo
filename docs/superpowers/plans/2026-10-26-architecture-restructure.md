# Architecture Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move trader-koo into one-job modules with one-way dependencies and one implementation per job, without changing report output or paper decisions.

**Architecture:** Milestone 1 builds the proof tools (a golden-report harness and ratcheting guardrails). Every later milestone is a behaviour-identical move, proven by those tools. Milestones 2 to 11 are planned here at milestone level; each gets its own detailed task plan when it starts, because its exact file list depends on what the cleanup and earlier moves leave behind.

**Tech Stack:** Python 3.11, FastAPI, SQLite, pytest, `time-machine` (frozen clock), `import-linter` (layer contracts), `pylint` duplicate-code checker, React/Vite frontend.

**Spec:** `docs/superpowers/specs/2026-10-26-architecture-restructure-design.md`

## Global Constraints

- Step 0 cleanup PRs (ML, fake signals, research machinery, subscribers and misc, Gemini rewrite) merge before any file moves (milestone 2 onward). Milestone 1 may start now.
- Policy fingerprint `canonical_hash(config_snapshot(PaperTradeConfig()))` is identical on main and on every branch.
- No change to any `paper_*` or `report_*` table, `_PAPER_DECISION_VERSION`, or `ops/PAPER_SCHEMA_V5_CONTRACT.md`.
- `paper/` constants stay where they are; `paper/` moves as code only and is exempt from the bare-number check.
- File limit 400 lines, excluding tests, generated frontend types, `settings.toml` and baselined legacy files.
- Bare number: numeric literal other than 0, 1, -1, 2, 100 in non-test backend code outside `config/` and `paper/`.
- Module names: `config/`, `market_data/`, `signals/`, `report/`, `paper/`, `research/`, `notify/`, `web/`, `ops/`, `addons/`, `agents/` (reserved).
- Layers, top imports bottom only: `web, ops` > `paper, research, addons` > `report` > `signals` > `market_data, notify` > `config`.
- Each move leaves a re-export shim at the old path for one milestone; shims are deleted in milestone 11.
- One restructure PR deploys per trading day; no deploys 21:30 to 23:30 UTC on trading days; the next PR waits for the nightly run after the previous deploy to publish.
- Never stage with `git add -A` (worktrees carry a `.venv` symlink). Conventional Commits. Squash merges pass `--author-email haomingkoo@gmail.com`.
- Golden database copies are never committed.

## Review Focus

- Nondeterminism in report output (set iteration order, float formatting, `random`, dict order from SQL without ORDER BY): two harness runs on the same inputs must be identical.
- The harness silently reaching the network or Gemini (yfinance, Finnhub, RSS): any socket connect during a snapshot must raise.
- The generator failing closed on the pinned database (missing data, stale prices): the harness must record the failure as output, so a branch that stops failing closed shows up as a diff.
- Baseline drift when files move or are renamed: moving a file must not reset its guardrail allowance or add a new violation.
- Policy hash computed from defaults while production runs a registered config: the harness also hashes the campaign registration found in the pinned DB and compares both.

---

### Task 1: Golden-report snapshot

**Files:**
- Create: `ops/golden_report.py`
- Test: `tests/test_golden_report.py`
- Modify: `requirements-dev.txt` (add `time-machine` pinned to the current release)

**Interfaces:**
- Produces:
  - `snapshot(db_path: Path, out_dir: Path, now: datetime) -> Path`: writes `report.json`, `email.txt`, `email.html`, `paper.json`, `policy.json`, `network.json` into `out_dir` and returns it.
  - `normalize(payload: dict) -> dict`: removes volatile fields (generated timestamps, durations, absolute paths, run ids) and sorts lists that have no defined order.
  - CLI `python -m ops.golden_report snapshot --db PATH --out DIR --now ISO8601`.

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshot_is_deterministic(tmp_path, golden_db):
    now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
    a = snapshot(golden_db, tmp_path / "a", now)
    b = snapshot(golden_db, tmp_path / "b", now)
    for name in ("report.json", "email.txt", "email.html", "paper.json", "policy.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()

def test_snapshot_blocks_network(tmp_path, golden_db, monkeypatch):
    # any socket connect inside snapshot raises; snapshot still completes using stored data
    out = snapshot(golden_db, tmp_path / "s", datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc))
    assert json.loads((out / "network.json").read_text()) == []

def test_policy_json_has_default_and_registered_hash(tmp_path, golden_db):
    policy = json.loads((snapshot(golden_db, tmp_path / "p", NOW) / "policy.json").read_text())
    assert policy["default_hash"] == canonical_hash(config_snapshot(PaperTradeConfig()))
    assert "registered_hash" in policy

def test_generator_failure_is_recorded(tmp_path, empty_db):
    report = json.loads((snapshot(empty_db, tmp_path / "f", NOW) / "report.json").read_text())
    assert report["ok"] is False and report["error"]
```

`golden_db` is a fixture that copies the newest local price database (`trader_koo/data/trader_koo.db` or `data/prod_paper_price.db`) into `tmp_path`, and skips the test if neither exists.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_golden_report.py -v`
Expected: FAIL with `ModuleNotFoundError: ops.golden_report`

- [ ] **Step 3: Implement `snapshot`, `normalize` and the CLI in `ops/golden_report.py`**

Freeze the clock with `time_machine.travel(now, tick=False)`. Set `TRADER_KOO_LLM_ENABLED=0`. Replace `socket.socket.connect` with a function that records the address in `network.json` and raises `OSError`. Call `trader_koo.report.generator.fetch_report_payload(db_path, run_log, tail_lines=0)` on a temporary copy of the DB. Build the email with `trader_koo.report_email.build_report_email_bodies`. For `paper.json`, run `trader_koo.paper_trades.evaluate_setup_for_paper_trade` on every setup row in the report, sorted by ticker. Read `registered_hash` from the active campaign registration row in the DB, or `null` if there is none. Catch generator exceptions and write `{"ok": false, "error": "<type>: <message>"}`. Write JSON with `sort_keys=True, indent=1`.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_golden_report.py -v`
Expected: 4 passed (or skipped where no local DB exists).

- [ ] **Step 5: Commit**

```bash
git add ops/golden_report.py tests/test_golden_report.py requirements-dev.txt
git commit -m "test(ops): add golden-report snapshot harness"
```

### Task 2: Golden comparison and make target

**Files:**
- Modify: `ops/golden_report.py`, `Makefile`
- Test: `tests/test_golden_report.py`

**Interfaces:**
- Consumes: `snapshot`, `normalize` from Task 1.
- Produces: `compare(base_dir: Path, head_dir: Path) -> list[str]` (one line per differing file and JSON path); CLI `python -m ops.golden_report compare BASE HEAD` exiting 1 when the list is non-empty; `make golden BASE=origin/main` that snapshots `BASE` in a temporary worktree and the current tree on the same DB copy and `--now`, then compares.

- [ ] **Step 1: Write failing tests**

```python
def test_compare_identical_is_empty(tmp_path, golden_db):
    a = snapshot(golden_db, tmp_path / "a", NOW); b = snapshot(golden_db, tmp_path / "b", NOW)
    assert compare(a, b) == []

def test_compare_reports_json_path(tmp_path):
    write_snapshot(tmp_path / "a", report={"setups": [{"ticker": "AAPL", "score": 70}]})
    write_snapshot(tmp_path / "b", report={"setups": [{"ticker": "AAPL", "score": 71}]})
    assert compare(tmp_path / "a", tmp_path / "b") == ["report.json: setups[0].score 70 -> 71"]
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_golden_report.py -k compare -v`, expected FAIL (`compare` undefined).
- [ ] **Step 3: Implement `compare` and the `golden` make target.**
- [ ] **Step 4: Run tests and `make golden BASE=origin/main` on the unchanged tree.** Expected: tests pass; make prints `golden: identical` and exits 0.
- [ ] **Step 5: Commit** — `git add ops/golden_report.py Makefile tests/test_golden_report.py && git commit -m "test(ops): compare golden snapshots against a base ref"`

### Task 3: Ratcheting guardrails

**Files:**
- Create: `ops/guardrails.py`, `ops/guardrails_baseline.json`
- Modify: `Makefile` (`guardrails` target, added to `ci`), `requirements-dev.txt` (`pylint`, `import-linter`), `.importlinter`
- Test: `tests/test_guardrails.py`

**Interfaces:**
- Produces:
  - `scan(root: Path) -> dict[str, dict[str, int]]`: per file, `{"lines": n, "bare_numbers": n}` for Python files under `trader_koo/` (tests excluded, `config/` and `paper/` excluded from bare numbers).
  - `check(current: dict, baseline: dict, limit: int = 400) -> list[str]`: violations are a new file over 400 lines, a baselined file growing past its baseline lines, or any file exceeding its baselined bare-number count. A file whose content moved keeps its allowance when the baseline entry is renamed in the same PR.
  - `update_baseline(current: dict, baseline: dict) -> tuple[dict, list[str]]`: returns the baseline lowered to current values where current is lower, plus a list of entries where current is higher (left unchanged).
  - CLI `python -m ops.guardrails check` and `python -m ops.guardrails baseline` (writes `update_baseline` output; exits 1 if anything would rise).
  - Duplicate check: `pylint --disable=all --enable=duplicate-code --min-similarity-lines=12 trader_koo` count recorded in the baseline; CI fails if it rises.
  - `.importlinter` with a `layers` contract over the modules that exist; each milestone adds its module.

- [ ] **Step 1: Write failing tests**

```python
def test_new_file_over_limit_fails():
    assert check({"a.py": {"lines": 401, "bare_numbers": 0}}, {}) == ["a.py: 401 lines > 400"]

def test_baselined_file_may_shrink_not_grow():
    base = {"g.py": {"lines": 1700, "bare_numbers": 90}}
    assert check({"g.py": {"lines": 1650, "bare_numbers": 90}}, base) == []
    assert check({"g.py": {"lines": 1701, "bare_numbers": 90}}, base) == ["g.py: 1701 lines > baseline 1700"]

def test_bare_number_allowlist(tmp_path):
    (tmp_path / "trader_koo").mkdir(); f = tmp_path / "trader_koo" / "m.py"
    f.write_text("x = 0\ny = 1\nz = -1\nw = 2\nv = 100\nthreshold = 0.35\n")
    assert scan(tmp_path)["trader_koo/m.py"]["bare_numbers"] == 1

def test_baseline_update_never_raises_numbers():
    base = {"g.py": {"lines": 10, "bare_numbers": 3}}
    new, raised = update_baseline({"g.py": {"lines": 12, "bare_numbers": 1}}, base)
    assert new == {"g.py": {"lines": 10, "bare_numbers": 1}}
    assert raised == ["g.py: lines 12 > baseline 10"]
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_guardrails.py -v`, expected FAIL (module missing).
- [ ] **Step 3: Implement `scan`, `check`, CLI; record the baseline from origin/main after the step 0 cleanup merges** (record from current main if cleanup has not merged, and re-record once it does, which only lowers numbers).
- [ ] **Step 4: Run** `pytest tests/test_guardrails.py -v && make guardrails` — expected: pass, `guardrails: 0 new violations`.
- [ ] **Step 5: Commit** — `git add ops/guardrails.py ops/guardrails_baseline.json .importlinter Makefile requirements-dev.txt tests/test_guardrails.py && git commit -m "ci: add ratcheting size, bare-number and duplicate guardrails"`

### Task 4: PR proof template

**Files:**
- Create: `.github/pull_request_template.md`
- Modify: `CONTRIBUTING.md` (one paragraph pointing to `make golden` and `make guardrails`)

- [ ] **Step 1:** Template sections: Policy fingerprint (main / branch), `make golden` result with every listed diff explained, paper decisions identical (`paper.json`), schema files untouched (`git diff --stat origin/main -- trader_koo/paper_trade/schema.py ops/PAPER_SCHEMA_V5_CONTRACT.md`), `make guardrails`, `make ci`, frontend lint and build.
- [ ] **Step 2: Commit** — `git add .github/pull_request_template.md CONTRIBUTING.md && git commit -m "docs: add restructure proof checklist to PR template"`

---

## Milestones 2 to 11

Each milestone starts with its own detailed task plan (`docs/superpowers/plans/2026-..-restructure-mN-<module>.md`), written from the tree as it is then. Every milestone PR passes the Task 4 template and adds its module to `.importlinter`.

| # | Milestone | Done when |
|---|---|---|
| 2 | `config/`: typed frozen settings from `config/settings.toml`; existing `trader_koo/config.py` merged in; non-paper tunables move with current values | bare-number baseline drops; golden identical |
| 3 | `market_data/`: `store.load_prices()`/`upsert_prices()`, one client per vendor, basis checks, ingest; inline SQL on market tables replaced | no SQL against market tables outside `market_data/` (import-linter + grep check); golden identical |
| 4 | `signals/`: one indicators module, levels, patterns, YOLO wrapper, regime/VIX; old `structure/`, `analysis/`, `features/`, `signals/` merged | one copy each of RSI/EMA/ATR, line fit, weekly resample, chart render, VIX read; golden identical |
| 5 | `notify/`: one `send(channel, dedup_key, body)` with a persisted dedup table (new table, additive); Telegram, email transport, Finnhub client unified | one sender per channel; alert behaviour verified against recorded alert fixtures |
| 6 | `ops/`: one scheduler module, DB-row overlap guard only, one backup path, cleanup as a script the scheduler calls | lock file and writer lease removed; nightly run publishes after deploy |
| 7 | `report/`: scoring, calibration, lineage, build, renderers (web, email, telegram); generator shrinks below 400 lines per file | golden identical including `email.*` |
| 8 | `paper/` and `research/`: move behind `paper_trades.py`; split `trading.py`; research keeps `next_open_baseline` and the evidence catalogue; SPY-matched return, Sharpe, drawdown helpers unified where outputs are identical | policy hash and `paper.json` identical; exceptions listed in ARCHITECTURE.md |
| 9 | `web/`: thin routers on shared loaders, one CORS, one throttle; `response_model` on every router; entrypoint moves to `trader_koo.web.app:app` with `start.sh`/`railway.toml` updated in a dark-deploy PR | contract tests pass; production health after deploy |
| 10 | `addons/`: crypto, Hyperliquid, Polymarket moved; no core import of `addons/` | import-linter forbids core -> addons |
| 11 | Frontend shared components, generated API types, oversized pages split; delete all shims; ARCHITECTURE.md rewritten for the new layout | Playwright screenshots at 1440/768/390 match main; no shim imports remain; guardrails blocking with no baselined legacy files left except listed exceptions |
