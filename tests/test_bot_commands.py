"""Tests for TelegramCommandHandler command dispatch and response formatting."""
from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader_koo.notifications.bot_commands import TelegramCommandHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def handler(tmp_path: Path) -> TelegramCommandHandler:
    """Create a handler with dummy credentials and tmp dirs."""
    return TelegramCommandHandler(
        bot_token="test-token",
        chat_id="12345",
        db_path=tmp_path / "test.db",
        report_dir=tmp_path / "reports",
        finnhub_api_key="test-key",
        alert_engine=None,
    )


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

class TestHelpCommand:
    @pytest.mark.asyncio
    async def test_help_returns_command_list(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        result = await handler._cmd_help()
        assert "/status" in result
        assert "/top" in result
        assert "/price" in result
        assert "/vix" in result
        assert "/alerts" in result
        assert "/help" in result


# ---------------------------------------------------------------------------
# /alerts — empty DB
# ---------------------------------------------------------------------------

class TestAlertsCommand:
    @pytest.mark.asyncio
    async def test_alerts_no_db(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        result = await handler._cmd_alerts()
        assert "not available" in result.lower() or "no alerts" in result.lower()

    @pytest.mark.asyncio
    async def test_alerts_empty_table(
        self,
        handler: TelegramCommandHandler,
        tmp_path: Path,
    ) -> None:
        """When the DB exists but has no alerts today."""
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_alerts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker     TEXT    NOT NULL,
                level      REAL    NOT NULL,
                price      REAL    NOT NULL,
                alert_type TEXT    NOT NULL,
                setup_tier TEXT,
                bias       TEXT,
                sent_at    TEXT    NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

        handler._db_path = db_path
        result = await handler._cmd_alerts()
        assert "no alerts" in result.lower()


# ---------------------------------------------------------------------------
# /price — missing ticker arg
# ---------------------------------------------------------------------------

class TestPriceCommand:
    @pytest.mark.asyncio
    async def test_price_no_ticker(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        result = await handler._cmd_price("")
        assert "usage" in result.lower()

    @pytest.mark.asyncio
    async def test_price_sanitizes_input(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        """Ticker arg is sanitized to alphanumeric + dot + caret."""
        with patch.object(
            handler,
            "_fetch_finnhub_quote",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_quote:
            result = await handler._cmd_price("AA$PL!!")
            mock_quote.assert_called_once_with("AAPL")
            assert "could not fetch" in result.lower()


# ---------------------------------------------------------------------------
# Update dispatch — security
# ---------------------------------------------------------------------------

class TestUpdateDispatch:
    @pytest.mark.asyncio
    async def test_ignores_wrong_chat_id(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        """Messages from unauthorized chats are silently ignored."""
        update = {
            "update_id": 100,
            "message": {
                "chat": {"id": 99999},
                "text": "/help",
            },
        }
        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock
        ) as mock_reply:
            await handler._handle_update(update)
            mock_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_known_command(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        update = {
            "update_id": 101,
            "message": {
                "chat": {"id": 12345},
                "text": "/help",
            },
        }
        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock
        ) as mock_reply:
            await handler._handle_update(update)
            mock_reply.assert_called_once()
            sent_text = mock_reply.call_args[0][0]
            assert "/status" in sent_text

    @pytest.mark.asyncio
    async def test_unknown_command(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        update = {
            "update_id": 102,
            "message": {
                "chat": {"id": 12345},
                "text": "/foo",
            },
        }
        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock
        ) as mock_reply:
            await handler._handle_update(update)
            mock_reply.assert_called_once()
            assert "unknown command" in mock_reply.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_updates_last_update_id(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        update = {
            "update_id": 200,
            "message": {
                "chat": {"id": 12345},
                "text": "/help",
            },
        }
        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock
        ):
            await handler._handle_update(update)
        assert handler._last_update_id == 200

    @pytest.mark.asyncio
    async def test_strips_bot_username_from_command(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        """Commands like /help@TraderKooBot should work."""
        update = {
            "update_id": 103,
            "message": {
                "chat": {"id": 12345},
                "text": "/help@TraderKooBot",
            },
        }
        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock
        ) as mock_reply:
            await handler._handle_update(update)
            mock_reply.assert_called_once()
            assert "/status" in mock_reply.call_args[0][0]

    @pytest.mark.asyncio
    async def test_ignores_non_command_text(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        """Regular text messages (no leading /) are ignored."""
        update = {
            "update_id": 104,
            "message": {
                "chat": {"id": 12345},
                "text": "hello world",
            },
        }
        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock
        ) as mock_reply:
            await handler._handle_update(update)
            mock_reply.assert_not_called()


# ---------------------------------------------------------------------------
# /top — mocked report
# ---------------------------------------------------------------------------

class TestTopCommand:
    @pytest.mark.asyncio
    async def test_top_no_report(
        self,
        handler: TelegramCommandHandler,
    ) -> None:
        with patch(
            "trader_koo.notifications.bot_commands."
            "TelegramCommandHandler._cmd_top",
            wraps=handler._cmd_top,
        ):
            result = await handler._cmd_top()
            assert (
                "no daily report" in result.lower()
                or "no signals" in result.lower()
                or "no top setups" in result.lower()
            )
