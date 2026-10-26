"""Event-study mode: one ``Spec`` per tested outcome, evaluated against its matched and unconditional baselines.

A ``Spec`` holds per-session arrays over the window's sessions: raw signal days, the stratum of each
signal day, the outcome of every stratum on every session (NaN where the outcome window leaves the
window or an input is missing) and the matched baseline sessions per stratum. The evaluation keeps
independent events (``eventstats.independent``) whose outcome is defined, and reports every count it
removes. Outcomes are close-to-close frequencies; there are no costs and no positions here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trader_koo.research.settings import get as _settings

from . import eventstats as es

_E = _settings().event_study


@dataclass
class Spec:
    key: str                  # "H1", "H2a", "H6-SPY-top"
    title: str
    role: str                 # "primary" (in the Holm family), "secondary" or "variant" (reported only)
    signal: np.ndarray        # (N,) bool: signal at the close of each session, from data through it
    stratum: np.ndarray       # (N,) int: stratum of a signal day (ignored where there is no signal)
    outcome: np.ndarray       # (K, N) float: outcome per stratum, 1.0 / 0.0 / NaN
    baseline: np.ndarray      # (K, N) bool: matched baseline sessions per stratum
    claim: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        n = len(self.signal)
        if self.outcome.shape[1] != n or self.baseline.shape != self.outcome.shape or len(self.stratum) != n:
            raise ValueError(f"{self.key}: array shapes disagree")

    @property
    def strata(self) -> int:
        return self.outcome.shape[0]


def _event_outcome(spec: Spec) -> np.ndarray:
    """Outcome of the stratum each signal day belongs to (NaN where there is no signal)."""
    cols = np.arange(len(spec.signal))
    return np.where(spec.signal, spec.outcome[np.clip(spec.stratum, 0, spec.strata - 1), cols], np.nan)


def parts(spec: Spec, events: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """(4, K, N) sums: event counts, event hits, baseline counts, baseline hits."""
    k, n = spec.outcome.shape
    out = np.nan_to_num(_event_outcome(spec))
    onehot = np.zeros((k, n))
    onehot[spec.stratum[events], np.flatnonzero(events)] = 1.0
    base = baseline & np.isfinite(spec.outcome)
    return np.stack([onehot, onehot * out, base.astype(float), np.where(base, spec.outcome, 0.0)])


@dataclass
class Evaluation:
    spec: Spec
    n_signals: int                 # raw signal days
    n_no_outcome: int              # signal days dropped because their outcome window leaves the window
    n_all: int                     # signal days with an outcome
    n_independent: int
    n_independent_strict: int      # the "5 or fewer sessions apart" reading
    rate_event: float | None
    rate_event_all: float | None   # every signal day with an outcome, not only the independent ones
    rate_matched: float | None
    rate_unconditional: float | None
    effect: float | None
    effect_vs_unconditional: float | None
    boot: es.Bootstrap | None
    p: float | None
    claim_interval: tuple | None = None
    extra: dict = field(default_factory=dict)

    @property
    def sufficient(self) -> bool:
        return self.n_independent >= _E.min_independent


def evaluate(spec: Spec, seed: int) -> Evaluation:
    defined = np.isfinite(_event_outcome(spec))
    signal = spec.signal & (spec.stratum >= 0)
    indep = es.independent(signal) & defined
    strict = es.independent(signal, strict=True) & defined
    n_signals, n_all = int(signal.sum()), int((signal & defined).sum())
    base = dict(spec=spec, n_signals=n_signals, n_no_outcome=n_signals - n_all, n_all=n_all,
                n_independent=int(indep.sum()), n_independent_strict=int(strict.sum()))
    if not indep.any():
        return Evaluation(**base, rate_event=None, rate_event_all=None, rate_matched=None, rate_unconditional=None,
                          effect=None, effect_vs_unconditional=None, boot=None, p=None)
    main = es.point(parts(spec, indep, spec.baseline))
    uncond = es.point(parts(spec, indep, np.isfinite(spec.outcome)))
    allev = es.point(parts(spec, signal & defined, spec.baseline))
    boot = es.bootstrap(parts(spec, indep, spec.baseline), seed)
    # A matched baseline with no session for a stratum that has events is not estimable: no effect, no p.
    estimable = np.isfinite(main["effect"])
    return Evaluation(
        **base, rate_event=main["rate_event"], rate_event_all=allev["rate_event"],
        rate_matched=main["rate_base"] if estimable else None, rate_unconditional=uncond["rate_base"],
        effect=main["effect"] if estimable else None, effect_vs_unconditional=uncond["effect"],
        boot=boot, p=es.p_two_sided(boot.effect) if estimable else None)


def claim_interval(ev: Evaluation, alpha: float, seed: int) -> tuple | None:
    """Interval of the event rate itself (for a stated claim such as 85%), same blocks and seed."""
    if ev.boot is None:
        return None
    return es.interval(ev.boot.rate_event, alpha)
