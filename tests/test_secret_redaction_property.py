"""Property-based tests for secret redaction.

This module uses hypothesis to generate random data structures and verify
that secret redaction works correctly across all inputs.

**Validates: Requirements 6.2, 6.3**

Property 8: Secret Redaction in Logs
For any data structure being logged, if it contains keys matching secret
patterns (API keys, passwords, tokens), then the platform should replace
those values with [REDACTED] in the log output.
"""

import pytest
from hypothesis import given, strategies as st

from trader_koo.security.redaction import (
    redact_secrets,
    _is_secret_key,
    REDACTED_VALUE,
    SECRET_PATTERNS,
)


# Strategy for generating secret-like keys
secret_key_strategy = st.sampled_from([
    "API_KEY",
    "api_key",
    "TRADER_KOO_API_KEY",
    "PASSWORD",
    "password",
    "DB_PASSWORD",
    "TOKEN",
    "token",
    "JWT_SECRET_KEY",
    "SECRET",
    "secret",
    "AWS_SECRET_ACCESS_KEY",
    "SMTP_PASSWORD",
])

# Strategy for generating non-secret keys
non_secret_key_strategy = st.sampled_from([
    "username",
    "email",
    "ticker",
    "count",
    "status",
    "message",
    "timestamp",
    "id",
])

# Strategy for generating values
value_strategy = st.one_of(
    st.text(min_size=0, max_size=100),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)


@given(secret_key=secret_key_strategy, value=value_strategy)
def test_property_secret_keys_always_redacted(secret_key, value):
    """Property: Secret keys are always redacted regardless of value.

    For any key that matches a secret pattern, the value should be
    replaced with [REDACTED] after redaction.
    """
    data = {secret_key: value}
    redacted = redact_secrets(data)

    # The key should still exist
    assert secret_key in redacted

    # The value should be redacted
    assert redacted[secret_key] == REDACTED_VALUE


@given(non_secret_key=non_secret_key_strategy, value=value_strategy)
def test_property_non_secret_keys_preserved(non_secret_key, value):
    """Property: Non-secret keys preserve their values.

    For any key that does NOT match a secret pattern, the value should
    remain unchanged after redaction.
    """
    data = {non_secret_key: value}
    redacted = redact_secrets(data)

    # The key should still exist
    assert non_secret_key in redacted

    # The value should be unchanged (unless it's a very long string that looks like a secret)
    if isinstance(value, str) and len(value) >= 40 and value.replace('_', '').replace('-', '').isalnum():
        # Very long alphanumeric strings might be redacted as potential secrets
        pass
    else:
        assert redacted[non_secret_key] == value


@given(
    secret_key=secret_key_strategy,
    non_secret_key=non_secret_key_strategy,
    secret_value=st.text(min_size=1, max_size=100),
    non_secret_value=st.text(min_size=1, max_size=100),
)
def test_property_mixed_dict_partial_redaction(
    secret_key, non_secret_key, secret_value, non_secret_value
):
    """Property: Mixed dictionaries have only secret keys redacted.

    For any dictionary containing both secret and non-secret keys,
    only the secret keys should be redacted.
    """
    data = {
        secret_key: secret_value,
        non_secret_key: non_secret_value,
    }
    redacted = redact_secrets(data)

    # Both keys should exist
    assert secret_key in redacted
    assert non_secret_key in redacted

    # Secret key should be redacted
    assert redacted[secret_key] == REDACTED_VALUE

    # Non-secret key should be preserved (unless it looks like a secret)
    if len(non_secret_value) >= 40 and non_secret_value.replace('_', '').replace('-', '').isalnum():
        # Very long alphanumeric strings might be redacted
        pass
    else:
        assert redacted[non_secret_key] == non_secret_value


@given(
    depth=st.integers(min_value=1, max_value=5),
    secret_key=secret_key_strategy,
    value=value_strategy,
)
def test_property_nested_secrets_redacted(depth, secret_key, value):
    """Property: Secrets are redacted at any nesting depth.

    For any nested dictionary structure, secrets should be redacted
    regardless of how deeply they are nested.
    """
    # Build nested structure
    data = {secret_key: value}
    for _ in range(depth):
        data = {"nested": data}

    redacted = redact_secrets(data)

    # Navigate to the secret key
    current = redacted
    for _ in range(depth):
        current = current["nested"]

    # The secret should be redacted
    assert current[secret_key] == REDACTED_VALUE


@given(
    items=st.lists(
        st.dictionaries(
            keys=st.one_of(secret_key_strategy, non_secret_key_strategy),
            values=value_strategy,
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=10,
    )
)
def test_property_list_of_dicts_redacted(items):
    """Property: Secrets in lists of dictionaries are redacted.

    For any list containing dictionaries, all secret keys in all
    dictionaries should be redacted.
    """
    redacted = redact_secrets(items)

    # Check each item in the list
    for i, item in enumerate(items):
        for key, value in item.items():
            if _is_secret_key(key):
                # Secret keys should be redacted
                assert redacted[i][key] == REDACTED_VALUE
            else:
                # Non-secret keys should be preserved (with heuristic exception)
                if isinstance(value, str) and len(value) >= 40 and value.replace('_', '').replace('-', '').isalnum():
                    pass
                else:
                    assert redacted[i][key] == value


@given(data=st.recursive(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.text(min_size=0, max_size=50),
        st.none(),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(
            keys=st.one_of(secret_key_strategy, non_secret_key_strategy),
            values=children,
            max_size=5,
        ),
    ),
    max_leaves=20,
))
def test_property_arbitrary_structure_no_crash(data):
    """Property: Redaction never crashes on arbitrary data structures.

    For any arbitrarily nested data structure, the redaction function
    should complete without raising an exception.
    """
    try:
        redacted = redact_secrets(data)
        # Should complete without exception
        # Note: redacted can be None if data is None, which is valid
        assert True
    except RecursionError:
        # Acceptable if we hit max recursion depth
        pass


@given(
    key=st.text(min_size=1, max_size=50),
    value=st.text(min_size=1, max_size=100),
)
def test_property_redaction_idempotent(key, value):
    """Property: Redaction is idempotent.

    For any data structure, applying redaction multiple times should
    produce the same result as applying it once.
    """
    data = {key: value}

    redacted_once = redact_secrets(data)
    redacted_twice = redact_secrets(redacted_once)

    # Should be the same
    assert redacted_once == redacted_twice


@given(
    secret_key=secret_key_strategy,
    value=value_strategy,
)
def test_property_redacted_value_never_original(secret_key, value):
    """Property: Redacted values never equal original secret values.

    For any secret key, after redaction, the value should never be
    the same as the original value (unless it was already [REDACTED]).
    """
    data = {secret_key: value}
    redacted = redact_secrets(data)

    # If the original value was not already [REDACTED], it should be different
    if value != REDACTED_VALUE:
        assert redacted[secret_key] != value

    # The redacted value should always be [REDACTED]
    assert redacted[secret_key] == REDACTED_VALUE
