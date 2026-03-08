"""Database schema for webhook notifications."""

import sqlite3
from pathlib import Path


def ensure_webhook_schema(db_path: str | Path) -> None:
    """Create webhook-related tables if they don't exist.
    
    Creates:
    - webhooks: Registered webhook endpoints with configuration
    - webhook_deliveries: Delivery attempt history and status
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(str(db_path))
    try:
        # Webhooks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhooks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                url TEXT NOT NULL,
                events TEXT NOT NULL,
                auth_headers TEXT,
                hmac_secret TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_webhooks_user_id ON webhooks(user_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_webhooks_is_active ON webhooks(is_active)
        """)
        
        # Webhook deliveries table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhook_deliveries (
                id TEXT PRIMARY KEY,
                webhook_id TEXT NOT NULL,
                event TEXT NOT NULL,
                payload TEXT NOT NULL,
                status_code INTEGER,
                response_time_ms INTEGER,
                error TEXT,
                attempts INTEGER DEFAULT 1,
                delivered_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (webhook_id) REFERENCES webhooks(id) ON DELETE CASCADE
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_id 
            ON webhook_deliveries(webhook_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_created_at 
            ON webhook_deliveries(created_at)
        """)
        
        conn.commit()
    finally:
        conn.close()
