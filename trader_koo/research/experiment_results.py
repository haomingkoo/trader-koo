"""Hash-verified, read-only experiment results catalogue."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from trader_koo.research.next_open_baseline import (
    PACKAGED_ARTIFACT_PATH as BASELINE_PATH,
    artifact_state as baseline_state,
    canonical_json_bytes,
)

TOURNAMENT_PATH = Path(__file__).with_name("challenger_tournament_artifact_20260823.json")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_tournament(path: Path = TOURNAMENT_PATH) -> dict[str, Any]:
    unavailable = {
        "available": False, "status": "evidence_unavailable",
        "warnings": ["tournament_artifact_unavailable"],
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return unavailable
    if not isinstance(payload, dict):
        return unavailable
    expected = str(payload.get("artifact_sha256") or "")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if len(expected) != 64 or _sha256(body) != expected:
        return {
            **unavailable,
            "warnings": ["tournament_artifact_hash_mismatch"],
        }
    return {"available": True, "artifact_path": str(path), **payload}


def _baseline_result() -> dict[str, Any]:
    state = baseline_state(BASELINE_PATH)
    payload: dict[str, Any] = {}
    if state.get("available"):
        payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    warnings = list(payload.get("readiness_reasons") or state.get("readiness_reasons") or [])
    ledger = payload.get("execution_ledger")
    if not isinstance(ledger, dict):
        warnings.append("complete_execution_ledger_unavailable")
    return {
        "experiment_id": "next-open-baseline",
        "title": "Next-open setup baseline",
        "available": bool(state.get("available")),
        "evidence_label": "invalid",
        "status": payload.get("readiness_status") or state.get("readiness_status") or "evidence_unavailable",
        "selected": False,
        "automatic_promotion": False,
        "warnings": sorted(set(str(item) for item in warnings)),
        "manifest": {
            "strategy_version": payload.get("method"),
            "code_sha": provenance.get("implementation_sha256"),
            "data_snapshot_hash": provenance.get("input_sha256"),
            "universe_basis": "fixed_current_universe_survivor_study",
            "return_basis": payload.get("return_basis"),
            "config_hash": provenance.get("config_sha256"),
            "seed": None,
            "evaluation_windows": payload.get("splits"),
            "benchmark": payload.get("benchmark_basis"),
            "costs": {
                key: (payload.get("config") or {}).get(key)
                for key in (
                    "commission_bps_per_side", "minimum_commission_per_side",
                    "entry_slippage_bps", "exit_slippage_bps",
                    "short_borrow_bps_annual", "cash_rate_bps_annual",
                )
            },
            "artifact_hash": provenance.get("artifact_sha256"),
            "ledger_hash": (
                (ledger.get("provenance") or {}).get("ledger_sha256")
                if isinstance(ledger, dict) else None
            ),
        },
        "metrics": {
            "net_total_return_pct": summary.get("net_return_pct"),
            "cagr_pct": None,
            "volatility_pct": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown_pct": None,
            "calmar": None,
            "profit_factor": None,
            "win_rate_pct": None,
            "average_r": None,
            "exposure_pct": summary.get("max_gross_exposure_pct"),
            "turnover_pct": None,
            "capacity": summary.get("matched_spy_filled_notional"),
            "trade_count": summary.get("closed_trades"),
            "confidence_intervals": payload.get("confidence_intervals"),
        },
        "curves": {
            "strategy": payload.get("equity_curve") or [],
            "spy_total_return": [],
            "cash": [],
        },
        "folds": payload.get("splits"),
        "regimes": None,
        "attribution": None,
        "cost_stress": None,
        "downloads": {
            "manifest": "/api/research/experiments/next-open-baseline/download/manifest",
            "ledger": (
                "/api/research/experiments/next-open-baseline/download/ledger"
                if isinstance(ledger, dict) else None
            ),
        },
        "_download": {"manifest": payload, "ledger": ledger},
    }


def _tournament_result() -> dict[str, Any]:
    payload = _load_tournament()
    audit = payload.get("dataset_audit") if isinstance(payload.get("dataset_audit"), dict) else {}
    prereg = payload.get("preregistration") if isinstance(payload.get("preregistration"), dict) else {}
    warnings = list(audit.get("reasons") or payload.get("warnings") or [])
    return {
        "experiment_id": "challenger-tournament",
        "title": "Non-TA challenger tournament",
        "available": bool(payload.get("available")),
        "evidence_label": "invalid",
        "status": payload.get("status") or "evidence_unavailable",
        "selected": payload.get("selected_challenger") is not None,
        "automatic_promotion": False,
        "warnings": sorted(set(str(item) for item in warnings)),
        "manifest": {
            "strategy_version": prereg.get("schema_version"),
            "code_sha": None,
            "data_snapshot_hash": audit.get("dataset_sha256"),
            "universe_basis": audit.get("universe_treatment"),
            "return_basis": (audit.get("price_contract") or {}).get("basis"),
            "config_hash": prereg.get("preregistration_sha256"),
            "seed": None,
            "evaluation_windows": prereg.get("selection"),
            "benchmark": "SPY total return and cash",
            "costs": {
                name: spec.get("one_way_cost_bps", spec.get("one_way_cost_scenarios_bps"))
                for name, spec in (prereg.get("challengers") or {}).items()
            },
            "artifact_hash": payload.get("artifact_sha256"),
            "ledger_hash": None,
        },
        "metrics": None,
        "curves": {"strategy": [], "spy_total_return": [], "cash": []},
        "folds": payload.get("split"),
        "regimes": {"count": audit.get("volatility_regime_count")},
        "attribution": None,
        "cost_stress": None,
        "challengers": payload.get("challenger_results") or {},
        "heldout": payload.get("sealed_heldout"),
        "downloads": {
            "manifest": "/api/research/experiments/challenger-tournament/download/manifest",
            "ledger": None,
        },
        "_download": {"manifest": {key: value for key, value in payload.items() if key != "available"}},
    }


def experiment_catalogue() -> list[dict[str, Any]]:
    """Return hash-verified experiments without private download payloads."""
    results = [_baseline_result(), _tournament_result()]
    return [{key: value for key, value in row.items() if key != "_download"} for row in results]


def experiment_result(experiment_id: str) -> dict[str, Any] | None:
    for result in (_baseline_result(), _tournament_result()):
        if result["experiment_id"] == experiment_id:
            return {key: value for key, value in result.items() if key != "_download"}
    return None


def experiment_download(experiment_id: str, component: str) -> dict[str, Any] | None:
    for result in (_baseline_result(), _tournament_result()):
        if result["experiment_id"] != experiment_id:
            continue
        value = result["_download"].get(component)
        return copy.deepcopy(value) if isinstance(value, dict) else None
    return None
