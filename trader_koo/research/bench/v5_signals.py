"""Signals for docs/research/hypotheses-v5-composite.md (pre-registered, implemented as written)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import hypotheses as hyp
from . import v4_signals as v4
from .data import AsOfView

# Pre-registered definitions. Changing one is a new version.
MIN_COMPONENTS = 4  # "a name needs at least 4 of the 5"


def _quality_score(data: AsOfView, universe: list[str], mcap: pd.Series) -> pd.Series:
    """The v1 H1 score: mean of its three z-scores (clipped), over the names with all three inputs."""
    x = hyp.complete_rows(pd.DataFrame({
        "gp_assets": [hyp.gross_profit_to_assets(data, t) for t in universe],
        "neg_leverage": [-hyp.liabilities_to_assets(data, t) for t in universe],
        "fcf_yield": [hyp.free_cash_flow(data, t) for t in universe] / mcap.where(mcap > 0).to_numpy(),
    }, index=universe), data.diag, "v5_quality")
    return hyp.zscores(x, hyp.WINSOR).mean(axis=1).reindex(universe)


def composite_components(data: AsOfView, universe: list[str]) -> pd.DataFrame:
    """Raw component values per name (higher is better), before z-scoring."""
    mcap = hyp.market_caps(data, universe)
    px = data.close.reindex(columns=universe)
    if len(px) > hyp.MOM_LOOKBACK:
        momentum = px.iloc[-1 - hyp.MOM_SKIP] / px.iloc[-1 - hyp.MOM_LOOKBACK] - 1.0
        ret = px.iloc[-hyp.VOL_WINDOW - 1:].pct_change(fill_method=None).iloc[1:]
        low_risk = -ret.std(ddof=1).where(ret.notna().all())
    else:
        momentum = low_risk = pd.Series(np.nan, index=universe)
    return pd.DataFrame({
        "quality": _quality_score(data, universe, mcap),
        "value": [v4.magic_formula_inputs(data, t, mcap[t])[0] for t in universe],
        "momentum": momentum,
        "low_risk": low_risk,
        "share_shrink": [-v4.share_count_change(data, t) for t in universe],
    }, index=universe).replace([np.inf, -np.inf], np.nan)


def composite(params: dict):
    """H1/H2: mean of the available component z-scores (clipped at +-3), needing at least 4 of 5."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        x = composite_components(data, universe)
        for col in x.columns:
            data.diag.count(f"v5_input_missing:{col}", int(x[col].isna().sum()))
        z = hyp.zscores(x, hyp.WINSOR)
        n = z.notna().sum(axis=1)
        data.diag.count("v5_scored_on_4_of_5", int((n == MIN_COMPONENTS).sum()))
        data.diag.count("v5_unscored_under_4", int((n < MIN_COMPONENTS).sum()))
        return z.mean(axis=1).where(n >= MIN_COMPONENTS).reindex(universe)

    return signal
