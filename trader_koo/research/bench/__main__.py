"""CLI: python -m trader_koo.research.bench run|replay ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ledger
from .data import load_market
from .portfolio import DEFAULT_BUFFER_N, DEFAULT_MIN_HOLD, DEFAULT_REVIEW_EVERY, DEFAULT_TOP_N
from .run import DEFAULT_COST_BPS, DEFAULT_N_RANDOM, DEFAULT_STRESS_BPS, BenchConfig, run_bench

EXIT_REFUSED = 3


def _ledger_path(args) -> Path:
    return Path(args.ledger) if args.ledger else Path(args.db).with_name("bench_trial_ledger.jsonl")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trader_koo.research.bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one candidate and append it to the trial ledger")
    r.add_argument("--db", required=True, help="research database (opened read-only)")
    r.add_argument("--candidate", required=True)
    r.add_argument("--params", default="{}", help="JSON object of candidate parameters")
    r.add_argument("--start")
    r.add_argument("--end")
    r.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    r.add_argument("--stress-bps", type=float, default=DEFAULT_STRESS_BPS)
    r.add_argument("--n-random", type=int, default=DEFAULT_N_RANDOM)
    r.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    r.add_argument("--buffer-n", type=int, default=DEFAULT_BUFFER_N)
    r.add_argument("--review-every", type=int, default=DEFAULT_REVIEW_EVERY)
    r.add_argument("--min-hold", type=int, default=DEFAULT_MIN_HOLD)
    r.add_argument("--open-sealed", action="store_true", help="include the sealed final 20%% of sessions")
    r.add_argument("--ledger", help="trial ledger JSONL (default: next to the database)")
    r.add_argument("--json", action="store_true", help="print the full report as JSON")
    q = sub.add_parser("replay", help="re-run a ledger entry and compare its summary")
    q.add_argument("--db", required=True)
    q.add_argument("--trial", required=True, help="trial_id of a ledger run entry")
    q.add_argument("--open-sealed", action="store_true", help="required to replay a sealed run")
    q.add_argument("--ledger")
    args = p.parse_args(argv)
    path = _ledger_path(args)

    if args.cmd == "run":
        cfg = BenchConfig(
            args.candidate, json.loads(args.params), args.start, args.end, args.cost_bps,
            args.stress_bps, args.n_random, args.open_sealed, args.top_n, args.buffer_n,
            args.review_every, args.min_hold)
    else:
        entry = next((e for e in ledger.read(path)
                      if e.get("event") == "run" and e.get("trial_id") == args.trial), None)
        if entry is None:
            print(f"no run entry {args.trial} in {path}", file=sys.stderr)
            return 2
        if entry["config"]["open_sealed"] and not args.open_sealed:
            print("this entry opened the sealed window; replay needs --open-sealed", file=sys.stderr)
            return 2
        cfg = BenchConfig(**entry["config"])
    try:
        out = run_bench(load_market(args.db), cfg, path)
    except ledger.SealedWindowError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.cmd == "replay":
        same = out["summary"] == entry["summary"]
        print(f"replay {'MATCHES' if same else 'DIFFERS FROM'} ledger entry {args.trial}")
        return 0 if same else 1
    print(json.dumps({k: v for k, v in out.items() if k != "markdown"}, indent=2) if args.json
          else out["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
