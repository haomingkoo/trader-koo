"""Logging filter for secret redaction.

This module provides a logging filter that automatically redacts secrets
from all log messages before they are written to log files or console.

Requirements: 6.2, 6.3
"""

import logging
from typing import Any

from trader_koo.security.redaction import redact_secrets, redact_url_tokens


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts secrets from log records.

    This filter intercepts all log records and redacts any secret values
    before they are formatted and written to the log output.

    Usage:
        # Add to root logger
        logging.getLogger().addFilter(SecretRedactionFilter())

        # Add to specific logger
        logger = logging.getLogger("my_module")
        logger.addFilter(SecretRedactionFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter a log record to redact secrets.

        Args:
            record: The log record to filter.

        Returns:
            True (always allow the record, but modify it in place).
        """
        # Redact secrets from the message
        if hasattr(record, 'msg') and record.msg:
            if isinstance(record.msg, str):
                record.msg = redact_url_tokens(record.msg)
            else:
                # For non-string messages (e.g., dicts), redact secrets
                record.msg = redact_secrets(record.msg)

        # Redact secrets from args
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = redact_secrets(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(redact_secrets(list(record.args)))

        # Redact secrets from extra fields
        if hasattr(record, '__dict__'):
            # Create a list of keys to check (avoid modifying dict during iteration)
            extra_keys = [
                key for key in record.__dict__.keys()
                if key not in {
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'message', 'pathname', 'process', 'processName',
                    'relativeCreated', 'thread', 'threadName', 'exc_info',
                    'exc_text', 'stack_info', 'taskName'
                }
            ]

            for key in extra_keys:
                value = getattr(record, key, None)
                if value is not None:
                    setattr(record, key, redact_secrets(value))

        return True


def install_secret_redaction_filter() -> None:
    """Install the secret redaction filter on the root logger.

    This function should be called early in application startup to ensure
    all log messages are filtered for secrets.
    """
    root_logger = logging.getLogger()

    # Check if filter is already installed
    for filter_obj in root_logger.filters:
        if isinstance(filter_obj, SecretRedactionFilter):
            return

    # Install the filter
    root_logger.addFilter(SecretRedactionFilter())
