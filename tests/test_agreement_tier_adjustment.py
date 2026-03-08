"""Unit tests for agreement score tier adjustment functionality."""
import pytest
from trader_koo.scripts.generate_daily_report import (
    _apply_agreement_tier_adjustment,
    _downgrade_tier,
)


class TestDowngradeTier:
    """Test the _downgrade_tier helper function."""

    def test_downgrade_tier_a_to_b(self):
        """Test tier A downgrades to B."""
        assert _downgrade_tier("A") == "B"

    def test_downgrade_tier_b_to_c(self):
        """Test tier B downgrades to C."""
        assert _downgrade_tier("B") == "C"

    def test_downgrade_tier_c_to_d(self):
        """Test tier C downgrades to D."""
        assert _downgrade_tier("C") == "D"

    def test_downgrade_tier_d_stays_d(self):
        """Test tier D stays at D (no tier E)."""
        assert _downgrade_tier("D") == "D"

    def test_downgrade_tier_lowercase(self):
        """Test tier downgrade handles lowercase input."""
        assert _downgrade_tier("a") == "B"
        assert _downgrade_tier("b") == "C"

    def test_downgrade_tier_with_whitespace(self):
        """Test tier downgrade handles whitespace."""
        assert _downgrade_tier(" A ") == "B"
        assert _downgrade_tier(" B ") == "C"

    def test_downgrade_tier_invalid_defaults_to_d(self):
        """Test invalid tier defaults to D."""
        assert _downgrade_tier("X") == "D"
        assert _downgrade_tier("") == "D"


class TestApplyAgreementTierAdjustment:
    """Test the _apply_agreement_tier_adjustment function."""

    def test_agreement_below_50_downgrades_tier_a(self):
        """Test agreement < 50% downgrades tier A to B."""
        row = {
            "ticker": "AAPL",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 45.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "B"

    def test_agreement_below_50_downgrades_tier_b(self):
        """Test agreement < 50% downgrades tier B to C."""
        row = {
            "ticker": "MSFT",
            "setup_tier": "B",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 40.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "C"

    def test_agreement_below_50_downgrades_tier_c(self):
        """Test agreement < 50% downgrades tier C to D."""
        row = {
            "ticker": "GOOGL",
            "setup_tier": "C",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 49.9
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "D"

    def test_agreement_below_50_tier_d_stays_d(self):
        """Test agreement < 50% keeps tier D at D."""
        row = {
            "ticker": "TSLA",
            "setup_tier": "D",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 40.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "D"

    def test_agreement_at_50_no_change(self):
        """Test agreement = 50% does not change tier."""
        row = {
            "ticker": "NVDA",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 50.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"

    def test_agreement_above_50_no_change(self):
        """Test agreement > 50% does not change tier."""
        row = {
            "ticker": "META",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 75.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"

    def test_agreement_above_75_no_change(self):
        """Test agreement > 75% does not change tier."""
        row = {
            "ticker": "AMZN",
            "setup_tier": "B",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 85.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "B"

    def test_missing_agreement_score_no_change(self):
        """Test missing agreement score does not change tier."""
        row = {
            "ticker": "NFLX",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {}
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"

    def test_null_agreement_score_no_change(self):
        """Test null agreement score does not change tier."""
        row = {
            "ticker": "AMD",
            "setup_tier": "B",
            "debate_v1": {
                "consensus": {
                    "agreement_score": None
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "B"

    def test_missing_consensus_no_change(self):
        """Test missing consensus object does not change tier."""
        row = {
            "ticker": "INTC",
            "setup_tier": "A",
            "debate_v1": {}
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"

    def test_missing_debate_v1_no_change(self):
        """Test missing debate_v1 does not change tier."""
        row = {
            "ticker": "ORCL",
            "setup_tier": "A"
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"

    def test_invalid_debate_v1_type_no_change(self):
        """Test invalid debate_v1 type does not change tier."""
        row = {
            "ticker": "IBM",
            "setup_tier": "A",
            "debate_v1": "invalid"
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"

    def test_invalid_consensus_type_no_change(self):
        """Test invalid consensus type does not change tier."""
        row = {
            "ticker": "CSCO",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": "invalid"
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"

    def test_agreement_score_clamping_negative(self, capsys):
        """Test negative agreement score is clamped to 0."""
        row = {
            "ticker": "CRM",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": -10.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        # Clamped to 0, which is < 50, so tier should downgrade
        assert row["setup_tier"] == "B"
        # Check that clamping was logged
        captured = capsys.readouterr()
        assert "[AGREEMENT]" in captured.out
        assert "clamped" in captured.out

    def test_agreement_score_clamping_above_100(self, capsys):
        """Test agreement score > 100 is clamped to 100."""
        row = {
            "ticker": "ADBE",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 150.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        # Clamped to 100, which is >= 50, so tier should not change
        assert row["setup_tier"] == "A"
        # Check that clamping was logged
        captured = capsys.readouterr()
        assert "[AGREEMENT]" in captured.out
        assert "clamped" in captured.out

    def test_tier_downgrade_logging(self, capsys):
        """Test tier downgrade is logged."""
        row = {
            "ticker": "PYPL",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 45.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        captured = capsys.readouterr()
        assert "[AGREEMENT]" in captured.out
        assert "Tier downgraded A→B" in captured.out
        assert "PYPL" in captured.out
        assert "45.0%" in captured.out

    def test_no_logging_when_tier_unchanged(self, capsys):
        """Test no downgrade logging when tier doesn't change."""
        row = {
            "ticker": "SQ",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 75.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        captured = capsys.readouterr()
        # Should not log tier downgrade
        assert "Tier downgraded" not in captured.out

    def test_edge_case_agreement_49_9(self):
        """Test edge case: agreement 49.9% triggers downgrade."""
        row = {
            "ticker": "SHOP",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 49.9
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "B"

    def test_edge_case_agreement_50_0(self):
        """Test edge case: agreement 50.0% does not trigger downgrade."""
        row = {
            "ticker": "UBER",
            "setup_tier": "A",
            "debate_v1": {
                "consensus": {
                    "agreement_score": 50.0
                }
            }
        }
        _apply_agreement_tier_adjustment(row)
        assert row["setup_tier"] == "A"
