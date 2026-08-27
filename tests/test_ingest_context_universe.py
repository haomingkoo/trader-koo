from trader_koo.scripts.update_market_db import DEFAULT_SOFT_FAIL_TICKERS


def test_vix_term_structure_uses_provider_symbols_only():
    assert "^VIX3M" in DEFAULT_SOFT_FAIL_TICKERS
    assert "^VIX6M" in DEFAULT_SOFT_FAIL_TICKERS
    assert "VIX3M" not in DEFAULT_SOFT_FAIL_TICKERS
    assert "VIX6M" not in DEFAULT_SOFT_FAIL_TICKERS
