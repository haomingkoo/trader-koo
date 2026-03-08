"""Webhook service for event-driven notifications."""

import asyncio
import hashlib
import hmac
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Supported webhook event types."""
    
    PATTERN_DETECTED = "pattern_detected"
    REGIME_CHANGE = "regime_change"
    ALERT_TRIGGERED = "alert_triggered"
    REPORT_GENERATED = "report_generated"


class WebhookService:
    """Service for managing and delivering webhooks."""
    
    def __init__(self, db_path: str | Path):
        """Initialize webhook service.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = str(db_path)
    
    def register_webhook(
        self,
        user_id: str,
        url: str,
        events: list[str],
        auth_headers: dict[str, str] | None = None,
        hmac_secret: str | None = None,
    ) -> dict[str, Any]:
        """Register a new webhook.
        
        Args:
            user_id: User ID who owns the webhook
            url: Webhook URL to POST to
            events: List of event types to subscribe to
            auth_headers: Optional custom authentication headers
            hmac_secret: Optional HMAC secret for signature verification
            
        Returns:
            Webhook record with id and configuration
            
        Raises:
            ValueError: If URL is invalid or events list is empty
        """
        # Validate URL
        if not self._validate_url(url):
            raise ValueError(
                "Webhook URL must use HTTPS (except localhost for testing)"
            )
        
        # Validate events
        if not events:
            raise ValueError("At least one event type must be specified")
        
        valid_events = {e.value for e in WebhookEvent}
        invalid_events = set(events) - valid_events
        if invalid_events:
            raise ValueError(f"Invalid event types: {invalid_events}")
        
        webhook_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO webhooks (
                    id, user_id, url, events, auth_headers, hmac_secret
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    webhook_id,
                    user_id,
                    url,
                    json.dumps(events),
                    json.dumps(auth_headers) if auth_headers else None,
                    hmac_secret,
                ),
            )
            conn.commit()
            
            return {
                "id": webhook_id,
                "user_id": user_id,
                "url": url,
                "events": events,
                "is_active": True,
            }
        finally:
            conn.close()
    
    def get_webhook(self, webhook_id: str) -> dict[str, Any] | None:
        """Get webhook by ID.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            Webhook record or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT id, user_id, url, events, auth_headers, hmac_secret,
                       is_active, created_at, updated_at
                FROM webhooks
                WHERE id = ?
                """,
                (webhook_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "url": row["url"],
                "events": json.loads(row["events"]),
                "auth_headers": json.loads(row["auth_headers"]) if row["auth_headers"] else None,
                "hmac_secret": row["hmac_secret"],
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()
    
    def list_webhooks(self, user_id: str) -> list[dict[str, Any]]:
        """List all webhooks for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of webhook records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT id, user_id, url, events, is_active, created_at, updated_at
                FROM webhooks
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            
            webhooks = []
            for row in cursor.fetchall():
                webhooks.append({
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "url": row["url"],
                    "events": json.loads(row["events"]),
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                })
            
            return webhooks
        finally:
            conn.close()
    
    def update_webhook(
        self,
        webhook_id: str,
        url: str | None = None,
        events: list[str] | None = None,
        auth_headers: dict[str, str] | None = None,
        hmac_secret: str | None = None,
        is_active: bool | None = None,
    ) -> bool:
        """Update webhook configuration.
        
        Args:
            webhook_id: Webhook ID
            url: New URL (optional)
            events: New event list (optional)
            auth_headers: New auth headers (optional)
            hmac_secret: New HMAC secret (optional)
            is_active: New active status (optional)
            
        Returns:
            True if updated, False if webhook not found
        """
        updates = []
        params = []
        
        if url is not None:
            if not self._validate_url(url):
                raise ValueError(
                    "Webhook URL must use HTTPS (except localhost for testing)"
                )
            updates.append("url = ?")
            params.append(url)
        
        if events is not None:
            if not events:
                raise ValueError("At least one event type must be specified")
            valid_events = {e.value for e in WebhookEvent}
            invalid_events = set(events) - valid_events
            if invalid_events:
                raise ValueError(f"Invalid event types: {invalid_events}")
            updates.append("events = ?")
            params.append(json.dumps(events))
        
        if auth_headers is not None:
            updates.append("auth_headers = ?")
            params.append(json.dumps(auth_headers) if auth_headers else None)
        
        if hmac_secret is not None:
            updates.append("hmac_secret = ?")
            params.append(hmac_secret)
        
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        
        if not updates:
            return True  # Nothing to update
        
        updates.append("updated_at = datetime('now')")
        params.append(webhook_id)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                f"""
                UPDATE webhooks
                SET {', '.join(updates)}
                WHERE id = ?
                """,
                params,
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM webhooks WHERE id = ?",
                (webhook_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    async def trigger_webhook(
        self,
        event: WebhookEvent,
        payload: dict[str, Any],
        user_id: str | None = None,
    ) -> None:
        """Trigger webhooks for an event.
        
        Args:
            event: Event type
            payload: Event payload data
            user_id: Optional user ID to filter webhooks (if None, triggers for all users)
        """
        # Get all active webhooks subscribed to this event
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = """
                SELECT id, url, events, auth_headers, hmac_secret
                FROM webhooks
                WHERE is_active = 1
            """
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            cursor = conn.execute(query, params)
            webhooks = []
            
            for row in cursor.fetchall():
                events = json.loads(row["events"])
                if event.value in events:
                    webhooks.append({
                        "id": row["id"],
                        "url": row["url"],
                        "auth_headers": json.loads(row["auth_headers"]) if row["auth_headers"] else {},
                        "hmac_secret": row["hmac_secret"],
                    })
        finally:
            conn.close()
        
        # Deliver to all matching webhooks
        tasks = []
        for webhook in webhooks:
            task = self._deliver_webhook(
                webhook_id=webhook["id"],
                url=webhook["url"],
                event=event,
                payload=payload,
                auth_headers=webhook["auth_headers"],
                hmac_secret=webhook["hmac_secret"],
            )
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _deliver_webhook(
        self,
        webhook_id: str,
        url: str,
        event: WebhookEvent,
        payload: dict[str, Any],
        auth_headers: dict[str, str],
        hmac_secret: str | None,
    ) -> None:
        """Deliver webhook with retry logic.
        
        Args:
            webhook_id: Webhook ID
            url: Webhook URL
            event: Event type
            payload: Event payload
            auth_headers: Authentication headers
            hmac_secret: HMAC secret for signature
        """
        delivery_id = str(uuid.uuid4())
        
        # Prepare payload with event metadata
        full_payload = {
            "event": event.value,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": payload,
        }
        
        # Prepare headers
        headers = auth_headers.copy()
        headers["Content-Type"] = "application/json"
        
        # Add HMAC signature if configured
        if hmac_secret:
            signature = hmac.new(
                hmac_secret.encode(),
                json.dumps(full_payload).encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        # Retry with exponential backoff
        max_attempts = 3
        backoff_delays = [1, 2, 4]  # seconds
        
        for attempt in range(max_attempts):
            start_time = time.time()
            status_code = None
            error = None
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url,
                        json=full_payload,
                        headers=headers,
                    )
                    status_code = response.status_code
                    response.raise_for_status()
                    
                    # Success
                    response_time_ms = int((time.time() - start_time) * 1000)
                    self._log_delivery(
                        delivery_id=delivery_id,
                        webhook_id=webhook_id,
                        event=event.value,
                        payload=full_payload,
                        status_code=status_code,
                        response_time_ms=response_time_ms,
                        error=None,
                        attempts=attempt + 1,
                        delivered_at=datetime.utcnow().isoformat() + "Z",
                    )
                    logger.info(
                        f"Webhook delivered successfully: {webhook_id} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    return
                    
            except httpx.TimeoutException as e:
                error = f"Timeout after 10 seconds: {str(e)}"
                logger.warning(
                    f"Webhook delivery timeout: {webhook_id} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error = f"HTTP {status_code}: {str(e)}"
                logger.warning(
                    f"Webhook delivery failed with HTTP {status_code}: {webhook_id} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
            except Exception as e:
                error = f"Unexpected error: {str(e)}"
                logger.error(
                    f"Webhook delivery error: {webhook_id} "
                    f"(attempt {attempt + 1}/{max_attempts}): {e}"
                )
            
            # If not last attempt, wait before retry
            if attempt < max_attempts - 1:
                await asyncio.sleep(backoff_delays[attempt])
        
        # All attempts failed
        response_time_ms = int((time.time() - start_time) * 1000)
        self._log_delivery(
            delivery_id=delivery_id,
            webhook_id=webhook_id,
            event=event.value,
            payload=full_payload,
            status_code=status_code,
            response_time_ms=response_time_ms,
            error=error,
            attempts=max_attempts,
            delivered_at=None,
        )
        logger.error(
            f"Webhook delivery failed after {max_attempts} attempts: {webhook_id}"
        )
    
    def _log_delivery(
        self,
        delivery_id: str,
        webhook_id: str,
        event: str,
        payload: dict[str, Any],
        status_code: int | None,
        response_time_ms: int,
        error: str | None,
        attempts: int,
        delivered_at: str | None,
    ) -> None:
        """Log webhook delivery attempt.
        
        Args:
            delivery_id: Unique delivery ID
            webhook_id: Webhook ID
            event: Event type
            payload: Event payload
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            error: Error message if failed
            attempts: Number of attempts made
            delivered_at: Delivery timestamp if successful
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO webhook_deliveries (
                    id, webhook_id, event, payload, status_code,
                    response_time_ms, error, attempts, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery_id,
                    webhook_id,
                    event,
                    json.dumps(payload),
                    status_code,
                    response_time_ms,
                    error,
                    attempts,
                    delivered_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_delivery_history(
        self,
        webhook_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get delivery history for a webhook.
        
        Args:
            webhook_id: Webhook ID
            limit: Maximum number of records to return
            
        Returns:
            List of delivery records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT id, webhook_id, event, status_code, response_time_ms,
                       error, attempts, delivered_at, created_at
                FROM webhook_deliveries
                WHERE webhook_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (webhook_id, limit),
            )
            
            deliveries = []
            for row in cursor.fetchall():
                deliveries.append({
                    "id": row["id"],
                    "webhook_id": row["webhook_id"],
                    "event": row["event"],
                    "status_code": row["status_code"],
                    "response_time_ms": row["response_time_ms"],
                    "error": row["error"],
                    "attempts": row["attempts"],
                    "delivered_at": row["delivered_at"],
                    "created_at": row["created_at"],
                })
            
            return deliveries
        finally:
            conn.close()
    
    def get_delivery_stats(self, webhook_id: str) -> dict[str, Any]:
        """Get delivery statistics for a webhook.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            Statistics including success rate, average response time, etc.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_deliveries,
                    SUM(CASE WHEN delivered_at IS NOT NULL THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN delivered_at IS NULL THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN delivered_at IS NOT NULL THEN response_time_ms END) as avg_response_time_ms
                FROM webhook_deliveries
                WHERE webhook_id = ?
                """,
                (webhook_id,),
            )
            row = cursor.fetchone()
            
            total = row[0] or 0
            successful = row[1] or 0
            failed = row[2] or 0
            avg_response_time = row[3] or 0
            
            return {
                "total_deliveries": total,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "avg_response_time_ms": round(avg_response_time, 2),
            }
        finally:
            conn.close()
    
    def _validate_url(self, url: str) -> bool:
        """Validate webhook URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            
            # Allow localhost with HTTP for testing
            if parsed.hostname in ("localhost", "127.0.0.1"):
                return parsed.scheme in ("http", "https")
            
            # Require HTTPS for all other URLs
            return parsed.scheme == "https"
        except Exception:
            return False
