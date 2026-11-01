"""trader_koo.report — Split report generation package.

This package replaces the monolithic generate_daily_report.py script.
Submodules:
  - utils: shared helpers, calendar, constants
  - pattern_analysis: YOLO pattern delta / persistence / lifecycle
  - market_context: VIX regime, volatility inputs, technical context
  - setup_scoring: confluence scoring, debate guardrails, evaluation
  - setup_ranking: setup step sequence and order (score, enrich, guardrail, calibrate)
  - generator: fetch_signals / fetch_report_payload orchestrator
  - serializer: Markdown / JSON output, snapshot pruning
  - email_dispatch: nightly report email (transport in trader_koo.email_delivery)

Import from submodules directly for best performance:
    from trader_koo.report.generator import fetch_report_payload
"""
from __future__ import annotations
