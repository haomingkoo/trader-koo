"""Property-based tests for CORS origin validation.

**Validates: Requirements 4.2, 4.3, 4.4, 4.5**

This module tests the CORS origin validation logic using property-based testing
to ensure that only valid origin patterns are accepted and invalid origins are
rejected across a wide range of inputs.
"""

import pytest
from hypothesis import given, strategies as st

from trader_koo.config import Config


# Strategy for generating various origin strings
@st.composite
def origin_strings(draw):
    """Generate diverse origin strings for testing."""
    protocol = draw(st.sampled_from(["http://", "https://", "ftp://", "ws://", ""]))

    # Generate different host patterns
    host_type = draw(st.sampled_from(["localhost", "domain", "ip", "invalid"]))

    if host_type == "localhost":
        host = "localhost"
        port = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=65535)))
        if port:
            host = f"{host}:{port}"
    elif host_type == "domain":
        # Generate domain-like strings — exclude commas to avoid confusing
        # the comma-separated parser.
        domain = draw(st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="-"),
            min_size=1,
            max_size=20,
        ))
        tld = draw(st.sampled_from(["com", "org", "net", "io", "dev"]))
        host = f"{domain}.{tld}"
    elif host_type == "ip":
        octets = [draw(st.integers(min_value=0, max_value=255)) for _ in range(4)]
        host = ".".join(map(str, octets))
    else:
        # Generate invalid/random strings — exclude commas so one origin
        # does not split into multiple during CSV parsing.
        host = draw(st.text(
            alphabet=st.characters(
                blacklist_characters=",",
                blacklist_categories=("Cs",),
            ),
            min_size=0,
            max_size=30,
        ))

    return f"{protocol}{host}"


class TestCORSOriginValidation:
    """Property-based tests for CORS origin validation.

    **Property 6: CORS Origin Validation**

    For any origin string in the CORS configuration, the platform should validate
    it matches the pattern ``https://`` or ``http://localhost``, and for any incoming
    CORS request, the platform should reject requests from origins not in the
    allowed list and log the rejection.
    """

    @given(origin_strings())
    def test_valid_https_origins_accepted(self, origin: str):
        """Property: All HTTPS origins should be accepted."""
        if origin.startswith("https://") and len(origin) > len("https://"):
            assert Config._is_valid_cors_origin(origin), \
                f"HTTPS origin should be valid: {origin}"

    @given(origin_strings())
    def test_localhost_origins_accepted(self, origin: str):
        """Property: All http://localhost origins should be accepted."""
        if origin.startswith("http://localhost"):
            assert Config._is_valid_cors_origin(origin), \
                f"Localhost origin should be valid: {origin}"

    @given(origin_strings())
    def test_non_https_non_localhost_rejected(self, origin: str):
        """Property: Origins that are not HTTPS or localhost should be rejected."""
        if not origin.startswith("https://") and not origin.startswith("http://localhost"):
            assert not Config._is_valid_cors_origin(origin), \
                f"Non-HTTPS non-localhost origin should be invalid: {origin}"

    @given(st.lists(origin_strings(), min_size=0, max_size=10))
    def test_origin_list_parsing(self, origins: list[str]):
        """Property: Origin list parsing should handle various inputs.

        Because ``_parse_cors_origins`` splits on commas, generated origin
        strings must not contain commas (ensured by our ``origin_strings``
        strategy).  After joining with commas and re-parsing, each
        non-empty-after-strip origin should survive the round-trip.
        """
        origins_str = ",".join(origins)
        parsed = Config._parse_cors_origins(origins_str)

        assert isinstance(parsed, list)

        expected_count = sum(1 for o in origins if o.strip())
        assert len(parsed) == expected_count

    def test_empty_origin_list_default(self):
        """Property: Empty CORS origins should default to empty list."""
        assert Config._parse_cors_origins("") == []
        assert Config._parse_cors_origins("   ") == []
        assert Config._parse_cors_origins(None) == []

    @given(st.lists(
        st.sampled_from([
            "https://example.com",
            "https://app.example.com",
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8080",
        ]),
        min_size=1,
        max_size=5,
    ))
    def test_valid_origins_pass_validation(self, valid_origins: list[str]):
        """Property: Lists of valid origins should pass validation."""
        origins_str = ",".join(valid_origins)

        import os
        old_value = os.environ.get("TRADER_KOO_CORS_ORIGINS")
        try:
            os.environ["TRADER_KOO_CORS_ORIGINS"] = origins_str
            os.environ["ADMIN_STRICT_API_KEY"] = "0"

            config = Config()
            assert len(config.cors_allowed_origins) == len(valid_origins)
        finally:
            if old_value is not None:
                os.environ["TRADER_KOO_CORS_ORIGINS"] = old_value
            else:
                os.environ.pop("TRADER_KOO_CORS_ORIGINS", None)
            os.environ.pop("ADMIN_STRICT_API_KEY", None)

    @given(st.lists(
        st.sampled_from([
            "http://example.com",
            "ftp://example.com",
            "example.com",
            "http://192.168.1.1",
        ]),
        min_size=1,
        max_size=3,
    ))
    def test_invalid_origins_fail_validation(self, invalid_origins: list[str]):
        """Property: Lists containing invalid origins should fail validation."""
        origins_str = ",".join(invalid_origins)

        import os
        old_value = os.environ.get("TRADER_KOO_CORS_ORIGINS")
        try:
            os.environ["TRADER_KOO_CORS_ORIGINS"] = origins_str
            os.environ["ADMIN_STRICT_API_KEY"] = "0"

            with pytest.raises(Exception):
                Config()
        finally:
            if old_value is not None:
                os.environ["TRADER_KOO_CORS_ORIGINS"] = old_value
            else:
                os.environ.pop("TRADER_KOO_CORS_ORIGINS", None)
            os.environ.pop("ADMIN_STRICT_API_KEY", None)

    def test_mixed_valid_invalid_origins_fail(self):
        """Property: Mixed valid/invalid origins should fail validation."""
        origins_str = "https://example.com,http://badorigin.com,http://localhost:3000"

        import os
        old_value = os.environ.get("TRADER_KOO_CORS_ORIGINS")
        try:
            os.environ["TRADER_KOO_CORS_ORIGINS"] = origins_str
            os.environ["ADMIN_STRICT_API_KEY"] = "0"

            with pytest.raises(Exception):
                Config()
        finally:
            if old_value is not None:
                os.environ["TRADER_KOO_CORS_ORIGINS"] = old_value
            else:
                os.environ.pop("TRADER_KOO_CORS_ORIGINS", None)
            os.environ.pop("ADMIN_STRICT_API_KEY", None)
