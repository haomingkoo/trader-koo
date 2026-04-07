"""
VIX Analysis Module with Term Structure Fallback

This module provides VIX analysis capabilities including:
- Multi-source term structure calculation with fallback
- VIX3M -> VIX6M -> synthetic (VXX/UVXY) fallback chain
- Source labeling and timestamp tracking
- Logging for monitoring and debugging
"""

import logging
import sqlite3
from dataclasses import dataclass
import datetime as dt
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TermStructure:
    """Term structure data with source tracking."""

    vix_spot: float
    vix_3m: Optional[float]
    vix_6m: Optional[float]
    source: str  # "VIX3M" | "VIX6M" | "synthetic" | "unavailable"
    contango: bool
    slope: Optional[float]
    timestamp: dt.datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "vix_spot": round(self.vix_spot, 2),
            "vix_3m": round(self.vix_3m, 2) if self.vix_3m else None,
            "vix_6m": round(self.vix_6m, 2) if self.vix_6m else None,
            "source": self.source,
            "contango": self.contango,
            "slope": round(self.slope, 4) if self.slope else None,
            "timestamp": self.timestamp.isoformat(),
        }


def _fetch_latest_close(conn: sqlite3.Connection, ticker: str) -> Optional[float]:
    """Fetch the latest close price for a ticker."""
    try:
        result = conn.execute(
            """
            SELECT CAST(close AS REAL)
            FROM price_daily
            WHERE ticker = ? AND close IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()

        if result and result[0]:
            return float(result[0])
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {e}")
        return None


def _calculate_synthetic_term_structure(
    conn: sqlite3.Connection, vix_spot: float
) -> Optional[float]:
    """
    Calculate synthetic term structure from VXX/UVXY when VIX3M and VIX6M unavailable.

    VXX tracks 1-month VIX futures, UVXY tracks short-term VIX futures.
    We can estimate a 3-month forward VIX by analyzing the ratio.
    """
    try:
        vxx = _fetch_latest_close(conn, "VXX")
        uvxy = _fetch_latest_close(conn, "UVXY")

        if not vxx or not uvxy or vxx <= 0 or uvxy <= 0:
            logger.info("VXX or UVXY data unavailable for synthetic calculation")
            return None

        # Synthetic calculation: estimate 3M forward based on VXX premium
        # VXX typically trades at a premium to spot VIX due to contango
        # This is a simplified heuristic - in production, use futures curve
        vxx_premium_ratio = vxx / 20.0  # VXX baseline around 20
        synthetic_3m = vix_spot * (1.0 + (vxx_premium_ratio - 1.0) * 0.5)

        logger.info(
            f"Calculated synthetic 3M: {synthetic_3m:.2f} "
            f"(VIX: {vix_spot:.2f}, VXX: {vxx:.2f}, UVXY: {uvxy:.2f})"
        )
        return synthetic_3m

    except Exception as e:
        logger.error(f"Failed to calculate synthetic term structure: {e}")
        return None


def calculate_term_structure(conn: sqlite3.Connection) -> TermStructure:
    """
    Calculate VIX term structure with multi-source fallback.

    Fallback order:
    1. Try VIX3M (primary)
    2. Try VIX6M (secondary)
    3. Calculate synthetic from VXX/UVXY (tertiary)
    4. Return unavailable status

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
    """
    timestamp = dt.datetime.now(dt.timezone.utc)

    # Fetch VIX spot (required)
    vix_spot = _fetch_latest_close(conn, "^VIX")
    if not vix_spot or vix_spot <= 0:
        logger.error("VIX spot data unavailable")
        return TermStructure(
            vix_spot=0.0,
            vix_3m=None,
            vix_6m=None,
            source="unavailable",
            contango=False,
            slope=None,
            timestamp=timestamp,
        )

    # Try VIX3M first (Requirement 9.1)
    vix3m = _fetch_latest_close(conn, "^VIX3M") or _fetch_latest_close(conn, "VIX3M")

    if vix3m and vix3m > 0:
        slope = (vix3m - vix_spot) / vix_spot
        contango = slope > 0.03  # 3% threshold

        logger.info(
            f"Term structure from VIX3M: spot={vix_spot:.2f}, "
            f"3M={vix3m:.2f}, slope={slope:.4f}"
        )

        return TermStructure(
            vix_spot=vix_spot,
            vix_3m=vix3m,
            vix_6m=None,
            source="VIX3M",
            contango=contango,
            slope=slope,
            timestamp=timestamp,
        )

    logger.warning("VIX3M unavailable, attempting VIX6M fallback")

    # Try VIX6M fallback (Requirement 9.1)
    vix6m = _fetch_latest_close(conn, "^VIX6M") or _fetch_latest_close(conn, "VIX6M")

    if vix6m and vix6m > 0:
        slope = (vix6m - vix_spot) / vix_spot
        contango = slope > 0.03

        logger.info(
            f"Term structure from VIX6M: spot={vix_spot:.2f}, "
            f"6M={vix6m:.2f}, slope={slope:.4f}"
        )

        return TermStructure(
            vix_spot=vix_spot,
            vix_3m=None,
            vix_6m=vix6m,
            source="VIX6M",
            contango=contango,
            slope=slope,
            timestamp=timestamp,
        )

    logger.warning("VIX6M unavailable, attempting synthetic calculation")

    # Calculate synthetic from VXX/UVXY (Requirement 9.2)
    synthetic_3m = _calculate_synthetic_term_structure(conn, vix_spot)

    if synthetic_3m and synthetic_3m > 0:
        slope = (synthetic_3m - vix_spot) / vix_spot
        contango = slope > 0.03

        logger.info(
            f"Term structure from synthetic: spot={vix_spot:.2f}, "
            f"synthetic_3M={synthetic_3m:.2f}, slope={slope:.4f}"
        )

        return TermStructure(
            vix_spot=vix_spot,
            vix_3m=synthetic_3m,
            vix_6m=None,
            source="synthetic",
            contango=contango,
            slope=slope,
            timestamp=timestamp,
        )

    # All sources failed (Requirement 9.5)
    logger.error("All term structure sources unavailable (VIX3M, VIX6M, synthetic)")

    return TermStructure(
        vix_spot=vix_spot,
        vix_3m=None,
        vix_6m=None,
        source="unavailable",
        contango=False,
        slope=None,
        timestamp=timestamp,
    )


def format_term_structure_display(term_structure: TermStructure) -> str:
    """
    Format term structure for display with source labeling.

    Requirements: 9.3, 9.6
    """
    if term_structure.source == "unavailable":
        return "Term structure unavailable"

    lines = [
        f"VIX Spot: {term_structure.vix_spot:.2f}",
    ]

    if term_structure.vix_3m:
        lines.append(f"VIX 3M: {term_structure.vix_3m:.2f}")

    if term_structure.vix_6m:
        lines.append(f"VIX 6M: {term_structure.vix_6m:.2f}")

    if term_structure.slope is not None:
        lines.append(f"Slope: {term_structure.slope:.2%}")

    lines.append(f"State: {'Contango' if term_structure.contango else 'Backwardation/Flat'}")
    lines.append(f"Source: {term_structure.source}")
    lines.append(f"Timestamp: {term_structure.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    return "\n".join(lines)


def calculate_vix_percentile(conn: sqlite3.Connection, window_days: int = 252) -> Optional[float]:
    """
    Calculate VIX percentile rank over a rolling window.

    Args:
        conn: Database connection
        window_days: Rolling window size in days (default: 252 trading days = 1 year)

    Returns:
        Percentile rank (0-100) or None if insufficient data

    Requirements: 11.1
    """
    try:
        # Fetch VIX close prices for the window
        result = conn.execute(
            """
            SELECT CAST(close AS REAL) as close
            FROM price_daily
            WHERE ticker = '^VIX' AND close IS NOT NULL
            ORDER BY date DESC
            LIMIT ?
            """,
            (window_days + 1,),  # +1 to include current day
        ).fetchall()

        if not result or len(result) < 2:
            logger.warning(f"Insufficient VIX data for percentile calculation: {len(result) if result else 0} rows")
            return None

        # Extract close prices
        closes = [float(row[0]) for row in result if row[0] is not None]

        if len(closes) < 2:
            logger.warning("Insufficient valid VIX close prices")
            return None

        # Current VIX is the most recent (first in DESC order)
        current_vix = closes[0]

        # Calculate percentile: percentage of values less than current
        values_below = sum(1 for price in closes if price < current_vix)
        percentile = (values_below / len(closes)) * 100.0

        logger.info(
            f"VIX percentile: {percentile:.1f}% "
            f"(current: {current_vix:.2f}, window: {len(closes)} days)"
        )

        return percentile

    except Exception as e:
        logger.error(f"Failed to calculate VIX percentile: {e}")
        return None


def get_percentile_color(percentile: Optional[float]) -> str:
    """
    Get color code for VIX percentile display.

    Args:
        percentile: VIX percentile value (0-100)

    Returns:
        Color code: "green", "yellow", or "red"

    Requirements: 11.3
    """
    if percentile is None:
        return "gray"

    if percentile < 30:
        return "green"
    elif percentile < 70:
        return "yellow"
    else:
        return "red"


def should_show_volatility_warning(percentile: Optional[float]) -> bool:
    """
    Determine if elevated volatility warning should be displayed.

    Args:
        percentile: VIX percentile value (0-100)

    Returns:
        True if warning should be shown (percentile > 80)

    Requirements: 11.5
    """
    return percentile is not None and percentile > 80


def format_percentile_display(percentile: Optional[float]) -> str:
    """
    Format VIX percentile for display with color coding and warning.

    Args:
        percentile: VIX percentile value (0-100)

    Returns:
        Formatted display string

    Requirements: 11.2, 11.3, 11.5
    """
    if percentile is None:
        return "VIX Percentile: Unavailable"

    color = get_percentile_color(percentile)
    warning = " ⚠️ ELEVATED VOLATILITY" if should_show_volatility_warning(percentile) else ""

    return f"VIX Percentile: {percentile:.1f}% [{color.upper()}]{warning}"



@dataclass
class CompressionThresholds:
    """Adaptive compression thresholds based on VIX regime."""

    lower_percentile: float  # e.g., 20, 25, or 30
    upper_percentile: float  # e.g., 80, 75, or 70
    regime: str  # "tight" | "moderate" | "wide"
    vix_90d_percentile: float

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "regime": self.regime,
            "vix_90d_percentile": round(self.vix_90d_percentile, 2),
        }


def calculate_compression_thresholds(
    conn: sqlite3.Connection
) -> CompressionThresholds:
    """
    Calculate adaptive compression thresholds based on 90-day VIX percentile.

    Threshold regimes:
    - Tight (low vol): 20th/80th percentile when VIX percentile < 30
    - Moderate (normal vol): 25th/75th percentile when VIX percentile 30-70
    - Wide (high vol): 30th/70th percentile when VIX percentile > 70

    Args:
        conn: Database connection

    Returns:
        CompressionThresholds with regime-appropriate thresholds

    Requirements: 13.1, 13.2, 13.3, 13.4
    """
    # Calculate 90-day VIX percentile
    vix_90d_percentile = calculate_vix_percentile(conn, window_days=90)

    if vix_90d_percentile is None:
        logger.warning("Unable to calculate VIX percentile, using moderate thresholds")
        vix_90d_percentile = 50.0  # Default to moderate

    # Determine threshold regime based on VIX percentile
    if vix_90d_percentile < 30:
        # Low volatility - use tight thresholds (Requirement 13.2)
        lower, upper = 20.0, 80.0
        regime = "tight"
        logger.info(
            f"Using TIGHT compression thresholds (20/80) - "
            f"VIX 90d percentile: {vix_90d_percentile:.1f}%"
        )
    elif vix_90d_percentile <= 70:
        # Normal volatility - use moderate thresholds (Requirement 13.3)
        lower, upper = 25.0, 75.0
        regime = "moderate"
        logger.info(
            f"Using MODERATE compression thresholds (25/75) - "
            f"VIX 90d percentile: {vix_90d_percentile:.1f}%"
        )
    else:
        # High volatility - use wide thresholds (Requirement 13.4)
        lower, upper = 30.0, 70.0
        regime = "wide"
        logger.info(
            f"Using WIDE compression thresholds (30/70) - "
            f"VIX 90d percentile: {vix_90d_percentile:.1f}%"
        )

    return CompressionThresholds(
        lower_percentile=lower,
        upper_percentile=upper,
        regime=regime,
        vix_90d_percentile=vix_90d_percentile,
    )


def detect_compression_signal(
    bb_width_percentile: Optional[float],
    thresholds: CompressionThresholds
) -> tuple[str, str]:
    """
    Detect compression/expansion state using adaptive thresholds.

    Args:
        bb_width_percentile: Bollinger Band width percentile (0-100)
        thresholds: Adaptive compression thresholds

    Returns:
        Tuple of (compression_state, labeled_state)
        - compression_state: "compression" | "expansion" | "normal"
        - labeled_state: State with regime label, e.g., "compression (tight)"

    Requirements: 13.6
    """
    if bb_width_percentile is None:
        return "normal", "normal"

    if bb_width_percentile <= thresholds.lower_percentile:
        state = "compression"
        labeled = f"compression ({thresholds.regime})"
    elif bb_width_percentile >= thresholds.upper_percentile:
        state = "expansion"
        labeled = f"expansion ({thresholds.regime})"
    else:
        state = "normal"
        labeled = "normal"

    logger.debug(
        f"Compression detection: BB width percentile={bb_width_percentile:.1f}%, "
        f"thresholds={thresholds.lower_percentile}/{thresholds.upper_percentile} "
        f"({thresholds.regime}), state={labeled}"
    )

    return state, labeled


def format_compression_thresholds_display(
    thresholds: CompressionThresholds
) -> str:
    """
    Format compression thresholds for display in VIX analysis tab.

    Args:
        thresholds: Compression thresholds to display

    Returns:
        Formatted display string

    Requirements: 13.5
    """
    regime_emoji = {
        "tight": "🔒",
        "moderate": "⚖️",
        "wide": "📏"
    }

    emoji = regime_emoji.get(thresholds.regime, "")

    lines = [
        f"{emoji} Compression Thresholds: {thresholds.regime.upper()} regime",
        f"  Lower: {thresholds.lower_percentile:.0f}th percentile",
        f"  Upper: {thresholds.upper_percentile:.0f}th percentile",
        f"  VIX 90-day percentile: {thresholds.vix_90d_percentile:.1f}%",
    ]

    return "\n".join(lines)
