from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from trader_koo.ml import trainer as trainer_mod
from trader_koo.ml.trainer import _apply_target_mode, _save_model, build_dataset


def test_apply_target_mode_return_sign_keeps_time_expired_samples():
    dataset = pd.DataFrame({
        "label": [1, -1, 0, 0],
        "return_pct": [2.0, -2.0, 1.0, -1.0],
        "entry_date": ["2025-01-01"] * 4,
    })

    out = _apply_target_mode(dataset.copy(), "return_sign")

    assert out["target"].tolist() == [1, 0, 1, 0]
    assert len(out) == 4


def test_apply_target_mode_barrier_treats_time_expiry_as_no_target_hit():
    dataset = pd.DataFrame({
        "label": [1, -1, 0],
        "return_pct": [2.0, -2.0, 0.5],
        "entry_date": ["2025-01-01"] * 3,
    })

    out = _apply_target_mode(dataset.copy(), "barrier")

    assert out["target"].tolist() == [1, 0, 0]


def test_build_dataset_keeps_time_expired_labels_by_default(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE price_daily (
        ticker TEXT, date TEXT, close REAL,
        adjustment_basis TEXT DEFAULT 'split_adjusted_price_only',
        adjustment_version TEXT DEFAULT 'test-v1',
        basis_status TEXT DEFAULT 'verified'
    )""")
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
        [("SPY", f"2025-01-{day:02d}", 100.0 + day) for day in range(1, 6)],
    )

    def fake_features(*_args, **_kwargs):
        return pd.DataFrame({"ret_1d": [0.01]}, index=pd.Index(["AAPL"], name="ticker"))

    def fake_labels(*_args, **_kwargs):
        return pd.DataFrame({
            "ticker": ["AAPL"],
            "entry_date": ["2025-01-01"],
            "label": [0],
            "exit_reason": ["time_expired"],
            "return_pct": [0.8],
            "days_held": [10],
        })

    monkeypatch.setattr(trainer_mod, "extract_features_for_universe", fake_features)
    monkeypatch.setattr(trainer_mod, "generate_triple_barrier_labels", fake_labels)

    out = build_dataset(
        conn,
        start_date="2025-01-01",
        end_date="2025-01-05",
        sample_frequency=10,
        feature_columns=["ret_1d"],
    )

    assert len(out) == 1
    assert out.loc[0, "exit_reason"] == "time_expired"
    assert int(out.loc[0, "target"]) == 1
    assert out.attrs["return_basis"] == "split_adjusted_price_only"
    assert out.attrs["adjustment_version"] == "test-v1"
    assert out.attrs["distributions_included"] is False


def test_build_dataset_can_drop_time_expired_labels(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE price_daily (
        ticker TEXT, date TEXT, close REAL,
        adjustment_basis TEXT DEFAULT 'split_adjusted_price_only',
        adjustment_version TEXT DEFAULT 'test-v1',
        basis_status TEXT DEFAULT 'verified'
    )""")
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
        [("SPY", f"2025-01-{day:02d}", 100.0 + day) for day in range(1, 6)],
    )

    def fake_features(*_args, **_kwargs):
        return pd.DataFrame({"ret_1d": [0.01]}, index=pd.Index(["AAPL"], name="ticker"))

    def fake_labels(*_args, **_kwargs):
        return pd.DataFrame({
            "ticker": ["AAPL"],
            "entry_date": ["2025-01-01"],
            "label": [0],
            "exit_reason": ["time_expired"],
            "return_pct": [0.8],
            "days_held": [10],
        })

    monkeypatch.setattr(trainer_mod, "extract_features_for_universe", fake_features)
    monkeypatch.setattr(trainer_mod, "generate_triple_barrier_labels", fake_labels)

    out = build_dataset(
        conn,
        start_date="2025-01-01",
        end_date="2025-01-05",
        sample_frequency=10,
        feature_columns=["ret_1d"],
        drop_time_expired=True,
    )

    assert out.empty
    assert out.attrs["return_basis"] == "split_adjusted_price_only"


def test_saved_model_metadata_carries_price_estimand(monkeypatch, tmp_path: Path):
    class Booster:
        def save_model(self, path: str) -> None:
            Path(path).write_text("model", encoding="utf-8")

    class Model:
        booster_ = Booster()

    monkeypatch.setattr(trainer_mod, "_model_dir", lambda: tmp_path)
    _save_model(
        Model(),
        ["ret_1d"],
        [],
        price_contract={
            "basis": "split_adjusted_price_only",
            "version": "test-v1",
            "distributions_included": False,
        },
    )

    metadata = json.loads(
        (tmp_path / "swing_lgbm_latest_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["return_basis"] == "split_adjusted_price_only"
    assert metadata["adjustment_version"] == "test-v1"
    assert metadata["distributions_included"] is False
