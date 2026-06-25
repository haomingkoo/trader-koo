from __future__ import annotations

from trader_koo.ml.external_data import _redact_url_secrets


def test_redact_url_secrets_hides_fred_api_key():
    text = (
        "500 Server Error for url: "
        "https://api.stlouisfed.org/fred/series/observations?"
        "series_id=T10Y2Y&api_key=secret-123&file_type=json"
    )

    redacted = _redact_url_secrets(text)

    assert "secret-123" not in redacted
    assert "api_key=<redacted>" in redacted
    assert "series_id=T10Y2Y" in redacted
