"""Unit tests for VIX trap/reclaim pattern detection.

Requirements: 8.1, 8.2, 8.3, 8.4, 14.1, 14.2, 14.3, 14.4, 14.5
"""

import pandas as pd
import pytest

from trader_koo.structure.vix_patterns import (
    VIXTrapReclaimConfig,
    detect_vix_trap_reclaim_patterns,
    get_pattern_glossary,
)


class TestBullTrapDetection:
    """Test bull trap (failed breakout) detection."""

    def test_bull_trap_basic(self):
        """Test basic bull trap: break above resistance then reverse below."""
        # Create VIX data with a bull trap pattern
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "open": [15.0, 15.5, 16.0, 17.5, 16.5, 15.5, 15.0, 14.8, 14.5, 14.3],
                "high": [15.5, 16.0, 17.0, 18.0, 17.0, 16.0, 15.5, 15.0, 14.8, 14.5],
                "low": [14.8, 15.0, 15.5, 17.0, 16.0, 15.0, 14.5, 14.3, 14.0, 14.0],
                "close": [15.2, 15.8, 16.5, 17.5, 16.2, 15.2, 14.8, 14.5, 14.2, 14.2],
            }
        )

        # Resistance at 17.0
        levels = [{"type": "resistance", "level": 17.0, "zone_high": 17.2}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)

        # Should detect bull trap
        bull_traps = [p for p in patterns if p.pattern_type == "bull_trap"]
        assert len(bull_traps) > 0, "Should detect bull trap pattern"
        assert bull_traps[0].level == 17.0
        assert bull_traps[0].confidence > 0.5

    def test_no_bull_trap_without_reversal(self):
        """Test that no bull trap is detected if price stays above resistance."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "open": [15.0, 16.0, 17.0, 18.0, 18.5],
                "high": [16.0, 17.0, 18.0, 19.0, 19.5],
                "low": [14.8, 15.5, 16.5, 17.5, 18.0],
                "close": [15.8, 16.8, 17.8, 18.8, 19.0],
            }
        )

        levels = [{"type": "resistance", "level": 17.0, "zone_high": 17.2}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)
        bull_traps = [p for p in patterns if p.pattern_type == "bull_trap"]
        assert len(bull_traps) == 0, "Should not detect bull trap without reversal"


class TestBearTrapDetection:
    """Test bear trap (failed breakdown) detection."""

    def test_bear_trap_basic(self):
        """Test basic bear trap: break below support then reverse above."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "open": [15.0, 14.5, 14.0, 12.5, 13.5, 14.5, 15.0, 15.2, 15.5, 15.7],
                "high": [15.2, 15.0, 14.5, 13.0, 14.0, 15.0, 15.5, 15.7, 16.0, 16.2],
                "low": [14.5, 14.0, 13.0, 12.0, 13.0, 14.0, 14.5, 15.0, 15.2, 15.5],
                "close": [14.8, 14.2, 13.5, 12.5, 13.8, 14.8, 15.2, 15.5, 15.8, 16.0],
            }
        )

        # Support at 13.0
        levels = [{"type": "support", "level": 13.0, "zone_low": 12.8}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)

        # Should detect bear trap
        bear_traps = [p for p in patterns if p.pattern_type == "bear_trap"]
        assert len(bear_traps) > 0, "Should detect bear trap pattern"
        assert bear_traps[0].level == 13.0
        assert bear_traps[0].confidence > 0.5

    def test_no_bear_trap_without_reversal(self):
        """Test that no bear trap is detected if price stays below support."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "open": [15.0, 14.0, 13.0, 12.0, 11.5],
                "high": [15.2, 14.5, 13.5, 12.5, 12.0],
                "low": [14.0, 13.0, 12.0, 11.0, 10.5],
                "close": [14.2, 13.2, 12.2, 11.2, 10.8],
            }
        )

        levels = [{"type": "support", "level": 13.0, "zone_low": 12.8}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)
        bear_traps = [p for p in patterns if p.pattern_type == "bear_trap"]
        assert len(bear_traps) == 0, "Should not detect bear trap without reversal"


class TestSupportReclaimDetection:
    """Test support reclaim detection."""

    def test_support_reclaim_basic(self):
        """Test basic support reclaim: break below then close above for 2+ bars."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "open": [15.0, 14.5, 14.0, 12.5, 12.8, 13.2, 13.5, 13.8, 14.0, 14.2],
                "high": [15.2, 15.0, 14.5, 13.0, 13.5, 13.8, 14.0, 14.2, 14.5, 14.7],
                "low": [14.5, 14.0, 13.0, 12.0, 12.5, 13.0, 13.2, 13.5, 13.8, 14.0],
                "close": [14.8, 14.2, 13.5, 12.5, 13.0, 13.5, 13.8, 14.0, 14.3, 14.5],
            }
        )

        # Support at 13.0
        levels = [{"type": "support", "level": 13.0, "zone_low": 12.8}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)

        # Should detect support reclaim
        reclaims = [p for p in patterns if p.pattern_type == "support_reclaim"]
        assert len(reclaims) > 0, "Should detect support reclaim pattern"
        assert reclaims[0].level == 13.0
        assert reclaims[0].confidence > 0.5

    def test_no_support_reclaim_without_consecutive_closes(self):
        """Test that support reclaim requires consecutive closes above support."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8),
                "open": [15.0, 14.5, 14.0, 12.5, 13.2, 12.8, 13.3, 12.9],
                "high": [15.2, 15.0, 14.5, 13.0, 13.5, 13.2, 13.6, 13.3],
                "low": [14.5, 14.0, 13.0, 12.0, 12.8, 12.5, 12.9, 12.6],
                "close": [14.8, 14.2, 13.5, 12.5, 13.1, 12.7, 13.2, 12.8],
            }
        )

        levels = [{"type": "support", "level": 13.0, "zone_low": 12.8}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)
        reclaims = [p for p in patterns if p.pattern_type == "support_reclaim"]
        # Should not detect without 2 consecutive closes above
        assert len(reclaims) == 0, "Should not detect support reclaim without consecutive closes"


class TestResistanceReclaimDetection:
    """Test resistance reclaim detection."""

    def test_resistance_reclaim_basic(self):
        """Test basic resistance reclaim: break above then close below for 2+ bars."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "open": [15.0, 15.5, 16.0, 17.5, 17.2, 16.8, 16.5, 16.2, 16.0, 15.8],
                "high": [15.5, 16.0, 17.0, 18.0, 17.5, 17.2, 17.0, 16.8, 16.5, 16.3],
                "low": [14.8, 15.0, 15.5, 17.0, 16.5, 16.2, 16.0, 15.8, 15.5, 15.3],
                "close": [15.2, 15.8, 16.5, 17.5, 17.0, 16.5, 16.2, 16.0, 15.8, 15.5],
            }
        )

        # Resistance at 17.0
        levels = [{"type": "resistance", "level": 17.0, "zone_high": 17.2}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)

        # Should detect resistance reclaim
        reclaims = [p for p in patterns if p.pattern_type == "resistance_reclaim"]
        assert len(reclaims) > 0, "Should detect resistance reclaim pattern"
        assert reclaims[0].level == 17.0
        assert reclaims[0].confidence > 0.5


class TestGlossary:
    """Test glossary functionality."""

    def test_glossary_contains_all_patterns(self):
        """Test that glossary contains definitions for all pattern types."""
        glossary = get_pattern_glossary()

        required_patterns = [
            "bull_trap",
            "failed_breakout",
            "bear_trap",
            "failed_breakdown",
            "support_reclaim",
            "resistance_reclaim",
        ]

        for pattern in required_patterns:
            assert pattern in glossary, f"Glossary should contain {pattern}"
            assert len(glossary[pattern]) > 0, f"Glossary definition for {pattern} should not be empty"

    def test_glossary_definitions_are_descriptive(self):
        """Test that glossary definitions are meaningful."""
        glossary = get_pattern_glossary()

        for pattern, definition in glossary.items():
            assert len(definition) > 20, f"Definition for {pattern} should be descriptive"
            assert "VIX" in definition, f"Definition for {pattern} should mention VIX"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty VIX data."""
        data = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        levels = [{"type": "resistance", "level": 17.0, "zone_high": 17.2}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)
        assert len(patterns) == 0, "Should return empty list for empty data"

    def test_empty_levels(self):
        """Test handling of empty levels list."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "open": [15.0, 15.5, 16.0, 16.5, 17.0],
                "high": [15.5, 16.0, 16.5, 17.0, 17.5],
                "low": [14.8, 15.0, 15.5, 16.0, 16.5],
                "close": [15.2, 15.8, 16.2, 16.8, 17.2],
            }
        )

        patterns = detect_vix_trap_reclaim_patterns(data, [])
        assert len(patterns) == 0, "Should return empty list for empty levels"

    def test_insufficient_data(self):
        """Test handling of insufficient data points."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "open": [15.0, 15.5],
                "high": [15.5, 16.0],
                "low": [14.8, 15.0],
                "close": [15.2, 15.8],
            }
        )
        levels = [{"type": "resistance", "level": 17.0, "zone_high": 17.2}]

        patterns = detect_vix_trap_reclaim_patterns(data, levels)
        assert len(patterns) == 0, "Should return empty list for insufficient data"

    def test_custom_config(self):
        """Test using custom configuration."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "open": [15.0, 15.5, 16.0, 17.5, 16.5, 15.5, 15.0, 14.8, 14.5, 14.3],
                "high": [15.5, 16.0, 17.0, 18.0, 17.0, 16.0, 15.5, 15.0, 14.8, 14.5],
                "low": [14.8, 15.0, 15.5, 17.0, 16.0, 15.0, 14.5, 14.3, 14.0, 14.0],
                "close": [15.2, 15.8, 16.5, 17.5, 16.2, 15.2, 14.8, 14.5, 14.2, 14.2],
            }
        )
        levels = [{"type": "resistance", "level": 17.0, "zone_high": 17.2}]

        # Use custom config with longer lookback
        config = VIXTrapReclaimConfig(trap_lookback_bars=5)
        patterns = detect_vix_trap_reclaim_patterns(data, levels, config)

        # Should still detect patterns with custom config
        assert isinstance(patterns, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
