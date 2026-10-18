"""VACUUM is a cliff, not a chore: it stages a full rewrite and must refuse to start.

On a WAL database the whole rewrite lands in the -wal file inside one transaction, so
wal_autocheckpoint cannot interrupt it and journal_size_limit cannot cap it. It therefore
needs roughly the database size in free space, and on a volume at 68% that margin was
168 MiB. Failure is silent: the caller logs a warning and nothing alerts.
"""
from __future__ import annotations

from trader_koo.scripts.cleanup_storage import (
    VACUUM_FREE_SPACE_MULTIPLIER,
    VACUUM_MIN_FREELIST_BYTES,
    _vacuum_decision,
)

_MB = 1024 * 1024
_PAGE = 4096


def _pages(num_bytes: int) -> int:
    return num_bytes // _PAGE


def test_refuses_when_free_space_cannot_stage_the_rewrite():
    """The production case on 2026-10-18: 1297 MiB db against 1466 MiB free."""
    db_bytes = 1297 * _MB

    safe, reason = _vacuum_decision(
        free_bytes=1466 * _MB,
        db_bytes=db_bytes,
        freelist_pages=_pages(500 * _MB),
        page_size=_PAGE,
    )

    assert safe is False
    assert "free" in reason.lower()


def test_allows_when_free_space_clears_the_multiplier():
    """After reclaiming the three oldest backups: same db, 2.0 GB free."""
    safe, reason = _vacuum_decision(
        free_bytes=2048 * _MB,
        db_bytes=1297 * _MB,
        freelist_pages=_pages(500 * _MB),
        page_size=_PAGE,
    )

    assert safe is True, reason


def test_skips_when_the_freelist_is_too_small_to_be_worth_it():
    """Over 17 nights VACUUM reclaimed 108.6 MiB total, and nothing on 10 of them."""
    safe, reason = _vacuum_decision(
        free_bytes=4000 * _MB,
        db_bytes=1000 * _MB,
        freelist_pages=_pages(VACUUM_MIN_FREELIST_BYTES - _MB),
        page_size=_PAGE,
    )

    assert safe is False
    assert "freelist" in reason.lower()


def test_free_space_boundary_is_inclusive():
    """Exactly the multiplier is enough; below it is not."""
    db_bytes = 1000 * _MB
    ample_freelist = _pages(500 * _MB)

    at_boundary, _ = _vacuum_decision(
        free_bytes=int(db_bytes * VACUUM_FREE_SPACE_MULTIPLIER),
        db_bytes=db_bytes,
        freelist_pages=ample_freelist,
        page_size=_PAGE,
    )
    below_boundary, _ = _vacuum_decision(
        free_bytes=int(db_bytes * VACUUM_FREE_SPACE_MULTIPLIER) - _MB,
        db_bytes=db_bytes,
        freelist_pages=ample_freelist,
        page_size=_PAGE,
    )

    assert at_boundary is True
    assert below_boundary is False


def test_space_is_checked_before_worth_so_the_danger_is_reported_first():
    """When both fail, the operator needs to hear about disk, not about tidiness."""
    safe, reason = _vacuum_decision(
        free_bytes=100 * _MB,
        db_bytes=1000 * _MB,
        freelist_pages=0,
        page_size=_PAGE,
    )

    assert safe is False
    assert "free" in reason.lower()
