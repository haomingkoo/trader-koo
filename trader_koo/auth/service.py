"""Authentication service for multi-user RBAC."""

import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import uuid

import jwt

from trader_koo.auth.models import User, UserRole, Session, EmailToken
from trader_koo.auth.password import hash_password, verify_password, validate_password_complexity


class AuthService:
    """Authentication service handling JWT, API key, and email token auth."""
    
    def __init__(
        self,
        db_path: str | Path,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        jwt_expiration_hours: int = 24,
        email_token_expiration_days: int = 7,
        api_key: str | None = None,
    ):
        """Initialize authentication service.
        
        Args:
            db_path: Path to SQLite database
            jwt_secret: Secret key for JWT signing
            jwt_algorithm: JWT algorithm (default: HS256)
            jwt_expiration_hours: JWT token expiration in hours
            email_token_expiration_days: Email token expiration in days
            api_key: Legacy API key for backward compatibility
        """
        self.db_path = str(db_path)
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiration_hours = jwt_expiration_hours
        self.email_token_expiration_days = email_token_expiration_days
        self.legacy_api_key = api_key
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using legacy API key (Requirement 16.3).
        
        Args:
            api_key: API key to validate
            
        Returns:
            User object if valid, None otherwise
        """
        if not self.legacy_api_key or api_key != self.legacy_api_key:
            return None
        
        # Legacy API key grants admin access
        # Return a synthetic admin user
        return User(
            id="legacy-admin",
            username="legacy_admin",
            email="admin@trader-koo.local",
            password_hash="",
            role=UserRole.ADMIN,
            is_active=True,
            failed_login_attempts=0,
            locked_until=None,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
        )
    
    def authenticate_jwt(self, token: str) -> Optional[User]:
        """Authenticate using JWT token (Requirement 16.3).
        
        Args:
            token: JWT token to validate
            
        Returns:
            User object if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # Verify session exists and is not expired
            conn = self._get_conn()
            try:
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                cursor = conn.execute(
                    """
                    SELECT user_id, expires_at
                    FROM sessions
                    WHERE token_hash = ? AND expires_at > datetime('now')
                    """,
                    (token_hash,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Update last activity
                conn.execute(
                    """
                    UPDATE sessions
                    SET last_activity = datetime('now')
                    WHERE token_hash = ?
                    """,
                    (token_hash,)
                )
                conn.commit()
            finally:
                conn.close()
            
            return self.get_user_by_id(user_id)
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def authenticate_email_token(self, token: str) -> Optional[User]:
        """Authenticate using email token (Requirement 16.3).
        
        Args:
            token: Email token to validate
            
        Returns:
            User object if valid, None otherwise
        """
        conn = self._get_conn()
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            cursor = conn.execute(
                """
                SELECT id, user_id, expires_at, used_at, revoked_at
                FROM email_tokens
                WHERE token_hash = ?
                """,
                (token_hash,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            token_id, user_id, expires_at_str, used_at, revoked_at = row
            
            # Check if token is valid
            expires_at = datetime.fromisoformat(expires_at_str)
            now = datetime.utcnow()
            
            if used_at or revoked_at or now >= expires_at:
                return None
            
            # Mark token as used
            conn.execute(
                """
                UPDATE email_tokens
                SET used_at = datetime('now')
                WHERE id = ?
                """,
                (token_id,)
            )
            conn.commit()
            
            return self.get_user_by_id(user_id)
        finally:
            conn.close()
    
    def create_jwt(self, user: User, ip_address: str | None = None, user_agent: str | None = None) -> str:
        """Create JWT token for user.
        
        Args:
            user: User to create token for
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.jwt_expiration_hours)
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "iat": now,
            "exp": expires_at,
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Store session
        session_id = str(uuid.uuid4())
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO sessions (id, user_id, token_hash, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user.id, token_hash, expires_at.isoformat(), ip_address, user_agent)
            )
            conn.commit()
        finally:
            conn.close()
        
        return token
    
    def create_email_token(self, user: User) -> str:
        """Create email authentication token.
        
        Args:
            user: User to create token for
            
        Returns:
            Email token string
        """
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        token_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=self.email_token_expiration_days)
        
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO email_tokens (id, user_id, token_hash, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token_id, user.id, token_hash, expires_at.isoformat())
            )
            conn.commit()
        finally:
            conn.close()
        
        return token
    
    def revoke_token(self, token: str) -> bool:
        """Revoke email token (Requirement 3.6).
        
        Args:
            token: Token to revoke
            
        Returns:
            True if token was revoked
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                UPDATE email_tokens
                SET revoked_at = datetime('now')
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (token_hash,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, username, email, password_hash, role, is_active,
                       failed_login_attempts, locked_until, created_at, last_login
                FROM users
                WHERE id = ?
                """,
                (user_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_user(row)
        finally:
            conn.close()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User object if found
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, username, email, password_hash, role, is_active,
                       failed_login_attempts, locked_until, created_at, last_login
                FROM users
                WHERE username = ?
                """,
                (username,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_user(row)
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authentication successful
        """
        user = self.get_user_by_username(username)
        if not user:
            return None
        
        if user.is_locked():
            return None
        
        if not verify_password(password, user.password_hash):
            # Increment failed login attempts
            self._increment_failed_login(user.id)
            return None
        
        # Reset failed login attempts and update last login
        conn = self._get_conn()
        try:
            conn.execute(
                """
                UPDATE users
                SET failed_login_attempts = 0,
                    locked_until = NULL,
                    last_login = datetime('now')
                WHERE id = ?
                """,
                (user.id,)
            )
            conn.commit()
        finally:
            conn.close()
        
        return user
    
    def _increment_failed_login(self, user_id: str) -> None:
        """Increment failed login attempts and lock account if needed."""
        conn = self._get_conn()
        try:
            # Increment counter
            conn.execute(
                """
                UPDATE users
                SET failed_login_attempts = failed_login_attempts + 1
                WHERE id = ?
                """,
                (user_id,)
            )
            
            # Check if we need to lock the account (5 failed attempts)
            cursor = conn.execute(
                "SELECT failed_login_attempts FROM users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row and row[0] >= 5:
                # Lock for 15 minutes
                locked_until = datetime.utcnow() + timedelta(minutes=15)
                conn.execute(
                    """
                    UPDATE users
                    SET locked_until = ?
                    WHERE id = ?
                    """,
                    (locked_until.isoformat(), user_id)
                )
            
            conn.commit()
        finally:
            conn.close()
    
    def _row_to_user(self, row: tuple) -> User:
        """Convert database row to User object."""
        (
            user_id, username, email, password_hash, role, is_active,
            failed_login_attempts, locked_until_str, created_at_str, last_login_str
        ) = row
        
        locked_until = None
        if locked_until_str:
            locked_until = datetime.fromisoformat(locked_until_str)
        
        last_login = None
        if last_login_str:
            last_login = datetime.fromisoformat(last_login_str)
        
        return User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=UserRole(role),
            is_active=bool(is_active),
            failed_login_attempts=failed_login_attempts,
            locked_until=locked_until,
            created_at=datetime.fromisoformat(created_at_str),
            last_login=last_login,
        )
