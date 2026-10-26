"""Run hypotheses v3 on the development window and append the run to the trial ledger.

The window is every SPY session before the seal pinned in the ledger. The seal is read, never written or
moved; a ledger without one stops the run. Primary tests (one per hypothesis, outcome, instrument and side)
form the Holm family; secondary checks and variants are reported with plain 95% intervals and no verdict.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trader_koo.research.settings import get as _settings

from . import eventstats as es
from . import ledger, v3_price, v3_report, v3_vol
from .eventstudy import Spec, evaluate
from .v3_data import Frame, load_frame
from .v3_gap import h12
from .v3_price import H13_PUBLISHED_LIFT, H8_FALSIFY_MARGIN

_E = _settings().event_study
FAMILY = "v3"
HALF = 0.5  # H13: "our measured lift is below half of their published lift"


def sealed_boundary(path: str | Path) -> str:
    """The date pinned by the ledger's seal entry. Read only: a missing seal is an error, not a reason to create one."""
    for e in ledger.read(path):
        if e.get("event") == "seal":
            return e["boundary_date"]
    raise LookupError(f"{path} has no seal entry; the sealed window is not pinned, so nothing is run")


def build_specs(f: Frame) -> list[Spec]:
    specs = [*v3_vol.h1(f), *v3_vol.h2(f), *v3_vol.h3(f), *v3_vol.h4(f), *v3_price.h5(f)]
    specs += v3_price.h5(f, "SPY", v3_price.H5_HORIZONS) + v3_price.h5(f, "QQQ", (v3_price.H5_HORIZON, *v3_price.H5_HORIZONS))
    for t in ("SPY", "QQQ"):
        specs += v3_price.h6(f, t)
        specs += [s for h in v3_price.H6_HORIZONS for s in v3_price.h6(f, t, horizon=h, variant=True)]
        specs += [s for m in v3_price.H6_MULTS for s in v3_price.h6(f, t, mult=m, variant=True)]
    specs += v3_vol.h7(f) + v3_price.h9(f)
    specs += v3_price.h8(f, "SPY") + v3_price.h8(f, "QQQ") + v3_vol.h11(f)
    specs += v3_price.h13(f, "SPY") + v3_price.h13(f, "QQQ")
    h10 = v3_price.h10(f)
    return specs + (h10 or [])


# Primary family, in the order of the document. Blocked tests (H10 without its date list) stay in the family.
PRIMARY = ("H1", "H2a", "H2b", "H3", "H4", "H5-A", "H6-SPY-top", "H6-SPY-bottom", "H6-QQQ-top", "H6-QQQ-bottom", "H7",
           "H8-SPY", "H8-QQQ", "H9", "H10", "H11", "H12", "H13-SPY-a", "H13-SPY-b", "H13-QQQ-a", "H13-QQQ-b")


def _status(key: str, evals: dict, adj_p: float, level: float, specs: dict) -> dict:
    """Pass / fail / insufficient per the document's rules for one primary test (development window only)."""
    ev = evals.get(key)
    if ev is None:
        return {"status": "blocked", "why": "the FOMC decision-day list for 2010-2023 is incomplete"}
    if not ev.sufficient:
        return {"status": "insufficient", "why": f"{ev.n_independent} independent events, fewer than {_E.min_independent}"}
    if ev.effect is None or ev.boot is None:
        return {"status": "insufficient", "why": "the matched baseline has no session for a stratum with events"}
    ci = es.interval(ev.boot.effect, level)
    out = {"interval": ci, "level_alpha": level, "adj_p": adj_p}
    if key.startswith("H8"):  # the registered rule is a stand-aside gate, not a +5 point lift
        gap = ev.effect_vs_unconditional
        out["status"] = "falsified" if gap >= H8_FALSIFY_MARGIN else "not falsified"
        out["why"] = f"down frequency {gap * 100:+.2f} points against the unconditional (falsified at +3 or more)"
        return out
    reasons = []
    if ev.effect < _E.pass_margin:
        reasons.append(f"effect {ev.effect * 100:+.2f} points is below +{_E.pass_margin * 100:.0f}")
    if ci is None or ci[0] <= 0:
        reasons.append("Holm-adjusted interval includes zero")
    claim = specs[key].claim
    if claim is not None and ev.boot is not None:
        cci = es.interval(ev.boot.rate_event, level)
        out["claim"] = {"value": claim, "interval": cci, "falsified": bool(cci and cci[1] < claim)}
        if out["claim"]["falsified"]:
            reasons.append(f"stated {claim:.0%}: interval upper bound {cci[1]:.3f} is below it")
    if key.startswith("H13"):
        need = HALF * H13_PUBLISHED_LIFT[key[-1]]
        out["half_published_lift"] = need
        if ev.effect < need:
            reasons.append(f"lift {ev.effect * 100:.2f} points is below half the published {need * 100:.2f}")
    if key == "H5-A":
        diff = evals["H5-AvsB"]
        out["wick_weaker_points"] = diff.effect
        if diff.effect is None or diff.effect < v3_price.H5_WEAKER:
            reasons.append("a wick break is not at least 5 points weaker than a close break")
    falsified = bool(out.get("claim", {}).get("falsified"))
    out["status"] = ("fail" + (" (stated claim falsified)" if falsified else "")) if reasons else \
        "development criteria met (sealed window not opened)"
    out["why"] = "; ".join(reasons) if reasons else "criteria 1 and 2 met; criterion 3 needs the sealed window"
    return out


def run_v3(db_path: str | Path, ledger_path: str | Path, frame: Frame | None = None) -> dict:
    boundary = sealed_boundary(ledger_path)
    f = frame or load_frame(db_path, boundary)
    config = {"family": FAMILY, "boundary": boundary, "first_session": f.dates[0], "last_session": f.dates[-1],
              "sessions": f.n, "spacing": _E.spacing_sessions, "block": _E.bootstrap_block,
              "resamples": _E.bootstrap_resamples, "alpha": _E.alpha}
    seed = ledger.seed_from_config(config)
    specs = build_specs(f)
    evals = {s.key: evaluate(s, ledger.seed_from_config({**config, "spec": s.key})) for s in specs}
    evals["H12"] = h12(f, ledger.seed_from_config({**config, "spec": "H12"}))
    by_key = {s.key: s for s in specs} | {"H12": evals["H12"].spec}
    p = [(evals[k].p if k in evals and evals[k].p is not None and evals[k].sufficient else 1.0) for k in PRIMARY]
    adj, levels, rejected = es.holm(p)
    verdicts = {k: _status(k, evals, adj[i], levels[i], by_key) for i, k in enumerate(PRIMARY)}
    out = {"family": FAMILY, "seed": seed, "config": config, "boundary": boundary, "evaluations": evals,
           "verdicts": verdicts, "holm": {"m": len(PRIMARY), "adjusted_p": dict(zip(PRIMARY, adj)),
                                          "rejected": dict(zip(PRIMARY, rejected))},
           "data": {"basis": f.basis, "substitutions": f.diag.as_dict(), "fomc_years_missing": f.fomc_missing}}
    out["markdown"] = v3_report.markdown(out)
    summary = {k: {"n_independent": evals[k].n_independent if k in evals else None,
                   "effect": None if k not in evals or evals[k].effect is None else round(evals[k].effect, _E.report_digits),
                   "adjusted_p": round(adj[i], _E.report_digits), "status": verdicts[k]["status"]}
               for i, k in enumerate(PRIMARY)}
    out["summary"] = summary
    base = {"candidate": "v3_market_timing", "config": config, "seed": seed, "code_version": ledger.code_version(),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    ledger.append(ledger_path, {**base, "event": "event_study_run", "trial_id": ledger.short_id(base), "family": FAMILY,
                                "sealed_opened": False, "sealed_boundary": boundary,
                                "date_range": [f.dates[0], f.dates[-1]], "n_sessions": f.n,
                                "params_hash": ledger.sha(config), "summary": summary})
    out["trial_id"] = ledger.short_id(base)
    return out
