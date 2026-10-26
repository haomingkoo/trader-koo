"""Tunables for the price-history backfill, read from config/research.toml.

SAME_COMPANY_READDS and SYMBOL_RENAMES are reference data (corrections checked
by hand against company names and history), not tunables.
"""
from __future__ import annotations

from trader_koo.research.settings import get as _settings

_P = _settings().price_backfill

FLOOR = _P.floor
WARMUP_DAYS = _P.warmup_days
DEPARTURE_BUFFER_DAYS = _P.departure_buffer_days
FLOOR_TOLERANCE_DAYS = _P.floor_tolerance_days
RETRIES = _P.retries
BACKOFF_SEC = _P.backoff_sec
PACE_SEC = _P.pace_sec
SEED_BATCH = _P.seed_batch
DETAIL_MAX_CHARS = _P.detail_max_chars
PROBE_DAYS = _P.probe_days
HTTP_TIMEOUT_SEC = _P.http_timeout_sec
BUSY_TIMEOUT_MS = _P.busy_timeout_ms
SHA_PREFIX_CHARS = _P.sha_prefix_chars
GAP_DAYS = _P.gap_days
LATE_START_DAYS = _P.late_start_days
# Re-added members whose pre-gap bars on Yahoo are the same company (checked by name and history).
SAME_COMPANY_READDS = ("AMD", "EQT", "FISV", "FSLR", "JBL", "LDOS", "PCG", "TER")
# Facebook became Meta: the old symbol's spells are merged under the new one.
SYMBOL_RENAMES = {"FB": "META"}
