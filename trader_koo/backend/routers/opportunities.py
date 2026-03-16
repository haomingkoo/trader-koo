"""Opportunities endpoint: PEG screening data."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from trader_koo.backend.services.database import get_conn, select_fund_snapshot

router = APIRouter()


@router.get("/api/opportunities")
def opportunities(
    limit: int = Query(default=500, ge=1, le=1000),
    min_discount: float = Query(default=10.0),
    max_peg: float = Query(default=2.0),
    view: str = Query(default="all", pattern="^(undervalued|overvalued|all)$"),
    overvalued_threshold: float = Query(default=-10.0),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        snapshot_ts, universe_count = select_fund_snapshot(conn, min_complete_tickers=400)
        if snapshot_ts is None:
            return {"snapshot_ts": None, "count": 0, "rows": []}

        rows = conn.execute(
            """
            SELECT
                ticker,
                price,
                pe,
                peg,
                eps_ttm,
                eps_growth_5y,
                target_price,
                discount_pct,
                target_reason
            FROM finviz_fundamentals
            WHERE snapshot_ts = ?
            """,
            (snapshot_ts,),
        ).fetchall()

        all_rows = [dict(r) for r in rows]
        source_counts: dict[str, int] = {}
        enriched: list[dict[str, Any]] = []
        for r in all_rows:
            discount = r.get("discount_pct")
            peg_val = r.get("peg")
            reason = str(r.get("target_reason") or "")
            if reason.startswith("FINVIZ_"):
                target_source = "analyst_target"
            elif reason.startswith("MODEL_"):
                target_source = "model_eps_pe"
            else:
                target_source = "other"
            source_counts[target_source] = source_counts.get(target_source, 0) + 1

            valuation_label = "fair"
            if isinstance(discount, (int, float)):
                if discount >= 20:
                    valuation_label = "deep_undervalued"
                elif discount >= 10:
                    valuation_label = "undervalued"
                elif discount <= -20:
                    valuation_label = "deep_overvalued"
                elif discount <= -10:
                    valuation_label = "overvalued"

            if isinstance(peg_val, (int, float)) and peg_val > 3.0 and valuation_label in {"fair", "undervalued"}:
                valuation_label = "high_peg"

            r["target_source"] = target_source
            r["valuation_label"] = valuation_label
            enriched.append(r)

        def include_row(r: dict[str, Any]) -> bool:
            disc = r.get("discount_pct")
            peg_v = r.get("peg")
            if view == "all":
                return True
            if not isinstance(disc, (int, float)):
                return False
            if view == "undervalued":
                if not isinstance(peg_v, (int, float)) or peg_v <= 0 or peg_v > max_peg:
                    return False
                return disc >= min_discount
            if view == "overvalued":
                return disc <= overvalued_threshold
            return False

        filtered = [r for r in enriched if include_row(r)]
        if view == "overvalued":
            filtered.sort(key=lambda r: (r.get("discount_pct", 0.0), -(r.get("peg") or 0.0)))
        elif view == "all":
            filtered.sort(key=lambda r: str(r.get("ticker") or ""))
        else:
            filtered.sort(key=lambda r: (-(r.get("discount_pct") or 0.0), (r.get("peg") or 9999.0)))

        eligible_count = len(filtered)
        out_rows = filtered[:limit]

        return {
            "snapshot_ts": snapshot_ts,
            "count": len(out_rows),
            "eligible_count": eligible_count,
            "universe_count": universe_count,
            "rows": out_rows,
            "source_counts": source_counts,
            "filters": {
                "view": view,
                "min_discount": min_discount,
                "max_peg": max_peg,
                "overvalued_threshold": overvalued_threshold,
                "limit": limit,
            },
            "filter_help": {
                "view": "all: full universe, undervalued: upside candidates, overvalued: downside candidates",
                "min_discount": "Undervalued threshold: minimum upside discount to target (%)",
                "max_peg": "Valuation cap used in undervalued mode",
                "overvalued_threshold": "Overvalued threshold: maximum discount (negative means above target)",
                "limit": "Maximum rows returned",
            },
        }
    finally:
        conn.close()
