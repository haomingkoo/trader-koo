"""Tunables for the price-history backfill."""
from __future__ import annotations

WARMUP_DAYS = 400
DEPARTURE_BUFFER_DAYS = 10
FLOOR_TOLERANCE_DAYS = 7
RETRIES = 3
BACKOFF_SEC = 10.0
PACE_SEC = 1.0
SEED_BATCH = 20000
DETAIL_MAX_CHARS = 300
PROBE_DAYS = 14
HTTP_TIMEOUT_SEC = 30
BUSY_TIMEOUT_MS = 30000
SHA_PREFIX_CHARS = 10
GAP_DAYS = 365
LATE_START_DAYS = 45
# Re-added members whose pre-gap bars on Yahoo are the same company (checked by name and history).
SAME_COMPANY_READDS = ("AMD", "EQT", "FISV", "FSLR", "JBL", "LDOS", "PCG", "TER")
# Facebook became Meta: the old symbol's spells are merged under the new one.
SYMBOL_RENAMES = {"FB": "META"}
