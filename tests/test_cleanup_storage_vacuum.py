"""VACUUM is a cliff, not a chore: it stages a full rewrite and must refuse to start.

On a WAL database the whole rewrite lands in the -wal file inside one transaction, so
wal_autocheckpoint cannot interrupt it and journal_size_limit cannot cap it. It therefore
needs roughly the database size in free space, and on a volume at 68% that margin was
168 MiB. Failure is silent: the caller logs a warning and nothing alerts.
"""
from __future__ import annotations

from trader_koo.scripts.cleanup_storage import (
    DISK_ALERT_THRESHOLD_PCT,
    VACUUM_FREE_SPACE_MULTIPLIER,
    VACUUM_MIN_FREELIST_BYTES,
    _vacuum_decision,
    disk_pressure_alert,
    vacuum_headroom_alert,
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


# ---------------------------------------------------------------------------
# disk_pressure_alert -- the volume filled to 68% with no alarm of any kind
# ---------------------------------------------------------------------------

def test_stays_quiet_below_the_threshold():
    """Production after reclaiming three backups: 2.5 GB of 4.4 GB, about 57%."""
    assert disk_pressure_alert(
        free_bytes=2048 * _MB,
        total_bytes=4505 * _MB,
        db_bytes=1297 * _MB,
    ) is None


def test_alerts_at_the_threshold():
    total = 1000 * _MB
    used = int(total * DISK_ALERT_THRESHOLD_PCT / 100.0)

    alert = disk_pressure_alert(
        free_bytes=total - used,
        total_bytes=total,
        db_bytes=100 * _MB,
    )

    assert alert is not None
    assert f"{DISK_ALERT_THRESHOLD_PCT:.0f}%" in alert


def test_says_so_when_the_vacuum_is_already_blocked():
    """The dangerous state: full enough to alert AND too full to vacuum out of."""
    alert = disk_pressure_alert(
        free_bytes=200 * _MB,
        total_bytes=4505 * _MB,
        db_bytes=1297 * _MB,
    )

    assert alert is not None
    assert "blocked" in alert.lower()


def test_says_the_vacuum_still_fits_when_it_does():
    """Full, but still able to dig out: the operator needs to know which it is.

    400 MB free of 4505 MB is 91% used, so it alerts, while a 100 MB database
    needs only 130 MB to vacuum.
    """
    alert = disk_pressure_alert(
        free_bytes=400 * _MB,
        total_bytes=4505 * _MB,
        db_bytes=100 * _MB,
    )

    assert alert is not None
    assert "blocked" not in alert.lower()


def test_names_the_ruler_because_two_disagree():
    """df reports 4.4 GB usable; the Railway dashboard reports 5 GB of raw device."""
    alert = disk_pressure_alert(
        free_bytes=100 * _MB,
        total_bytes=4505 * _MB,
        db_bytes=1297 * _MB,
    )

    assert alert is not None
    assert "df" in alert.lower()


def test_a_zero_sized_volume_does_not_divide_by_zero():
    assert disk_pressure_alert(free_bytes=0, total_bytes=0, db_bytes=0) is None


# ---------------------------------------------------------------------------
# vacuum_headroom_alert -- the alarm a percentage threshold cannot raise
# ---------------------------------------------------------------------------

def test_headroom_alert_fires_while_the_disk_threshold_stays_silent():
    """The actual production state on 2026-10-18, and the whole point of this alarm.

    The volume was two thirds full, quiet by any sane disk threshold, yet VACUUM
    was already unable to start: it needs 1.3x the DATABASE and only 1537 MB of
    the required 1768 MB was free. The requirement scales with the database, not
    with the volume, so a percentage alert alone stays silent right up to the
    cliff. These two assertions together are the reason this function exists.
    """
    state = {
        "free_bytes": 1537 * _MB,
        "total_bytes": 4505 * _MB,
        "db_bytes": 1297 * _MB,
    }

    assert disk_pressure_alert(**state) is None
    alert = vacuum_headroom_alert(**state)

    assert alert is not None
    assert "headroom" in alert.lower()


def test_headroom_alert_clears_after_reclaiming_the_backups():
    """Same database, 2.0 GB free: the state after deleting three weekly backups."""
    assert vacuum_headroom_alert(
        free_bytes=2048 * _MB,
        total_bytes=4505 * _MB,
        db_bytes=1297 * _MB,
    ) is None


def test_headroom_alert_tracks_the_database_not_the_volume():
    """Identical free space, different database: only the larger one is starved.

    800 MB needs 1040 MB to vacuum and so is blocked by 1000 MB free; 100 MB
    needs 130 MB and is fine. The volume is 90% empty in both cases.
    """
    free, total = 1000 * _MB, 10_000 * _MB

    assert vacuum_headroom_alert(free_bytes=free, total_bytes=total, db_bytes=800 * _MB) is not None
    assert vacuum_headroom_alert(free_bytes=free, total_bytes=total, db_bytes=100 * _MB) is None
