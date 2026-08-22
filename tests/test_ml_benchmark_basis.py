from __future__ import annotations

import json
import sqlite3

import pandas as pd

from trader_koo.ml import benchmark
from trader_koo.scripts.update_market_db import ensure_schema
from trader_koo.db.price_contract import record_price_series_revision


def test_successful_benchmark_persists_price_estimand(monkeypatch, tmp_path):
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    conn.execute(
        """INSERT INTO price_daily (
        ticker, date, close, adjustment_basis, adjustment_version, basis_status
        ) VALUES ('SPY', '2026-01-02', 100, ?, ?, 'verified')""",
        ("split_adjusted_price_only", "test-v1"),
    )
    record_price_series_revision(
        conn,
        "SPY",
        evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
        fetch_timestamp="2026-01-02T00:00:00Z",
    )
    dataset = pd.DataFrame(
        {
            "entry_date": ["2026-01-02", "2026-02-02"],
            "label": [-1, 1],
            "return_pct": [-1.0, 1.0],
            "target": [0, 1],
        }
    )
    monkeypatch.setattr(benchmark, "build_dataset", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(benchmark, "_get_models", lambda: {})
    monkeypatch.chdir(tmp_path)

    result = benchmark.run_benchmark(conn, start_date="2026-01-01")
    artifact = json.loads(
        (tmp_path / "data/models/benchmark_results.json").read_text(encoding="utf-8")
    )

    assert result["return_basis"] == "split_adjusted_price_only"
    assert result["adjustment_version"] == "test-v1"
    assert result["distributions_included"] is False
    assert artifact["return_basis"] == result["return_basis"]
    assert artifact["adjustment_version"] == result["adjustment_version"]
    assert artifact["distributions_included"] is False
