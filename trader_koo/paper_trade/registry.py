"""Campaigns declared in code, registered as drafts the first time the runtime policy reaches them.

On a contracted (v5) database ``ensure_paper_trade_schema`` runs no migration, so a new
campaign row cannot come from the schema initializer. A campaign listed here is inserted
as ``draft`` when report admission first runs with a matching runtime policy. It records
decisions in shadow mode until an authenticated human activates it; registration never
activates, freezes or resets any campaign. A campaign id that is not declared here, or a
declared one whose policy version differs from the runtime's, still fails closed.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from trader_koo.paper_trade.config import PaperTradeConfig

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeclaredCampaign:
    label: str
    policy_version: str


DECLARED_CAMPAIGNS: dict[str, DeclaredCampaign] = {
    "paper-v3": DeclaredCampaign(label="Paper Campaign v3", policy_version="paper-campaign-v3.0"),
}


def register_declared_campaign(conn: sqlite3.Connection, config: PaperTradeConfig) -> bool:
    """Insert the runtime campaign as a draft if it is declared and not yet registered.

    Returns True when a row was inserted. Raises ValueError when the runtime policy
    version does not match the declaration, so a mismatched build never registers.
    """
    declared = DECLARED_CAMPAIGNS.get(config.campaign_id)
    if declared is None:
        return False
    if declared.policy_version != config.decision_version:
        raise ValueError(
            f"campaign {config.campaign_id} is declared for {declared.policy_version}, "
            f"runtime policy is {config.decision_version}"
        )
    inserted = conn.execute(
        """INSERT OR IGNORE INTO paper_campaigns (
               campaign_id, label, policy_version, status, starting_capital, zero_admission_streak_limit
           ) VALUES (?, ?, ?, 'draft', ?, ?)""",
        (config.campaign_id, declared.label, declared.policy_version,
         config.starting_capital, config.zero_admission_streak_limit),
    ).rowcount
    if inserted:
        LOG.warning("Registered declared paper campaign %s (%s) as draft; activation is a separate human action",
                    config.campaign_id, declared.policy_version)
    return bool(inserted)


def registered_campaign_row(conn: sqlite3.Connection, config: PaperTradeConfig) -> tuple | None:
    """(status, policy_version, starting_capital, policy_hash) of the runtime campaign, registering it if declared."""
    register_declared_campaign(conn, config)
    return conn.execute(
        "SELECT status, policy_version, starting_capital, policy_hash FROM paper_campaigns WHERE campaign_id=?",
        (config.campaign_id,),
    ).fetchone()
