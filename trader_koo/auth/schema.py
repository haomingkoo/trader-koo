"""Database schema for authentication and user management."""

import sqlite3
from pathlib import Path


def ensure_auth_schema(db_path: str | Path) -> None:
    """Create authentication-related tables if they don't exist.
    
    Creates:
    - users: User accounts with roles and credentials
    - sessions: JWT session tracking
    - email_tokens: Email-based authentication tokens
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(str(db_path))
    try:
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'analyst', 'viewer')),
                is_active INTEGER DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login TEXT
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
        """)
        
        # Sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_activity TEXT NOT NULL DEFAULT (datetime('now')),
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)
        """)
        
        # Email tokens table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_tokens (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                used_at TEXT,
                revoked_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_email_tokens_token_hash ON email_tokens(token_hash)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_email_tokens_expires_at ON email_tokens(expires_at)
        """)
        
        conn.commit()
    finally:
        conn.close()
