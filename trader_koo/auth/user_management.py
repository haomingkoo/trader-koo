"""User management service for RBAC."""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from trader_koo.auth.models import User, UserRole
from trader_koo.auth.password import hash_password, validate_password_complexity


class UserManagementService:
    """Service for managing user accounts (Requirement 16.8)."""
    
    def __init__(self, db_path: str | Path):
        """Initialize user management service.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = str(db_path)
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole,
    ) -> tuple[Optional[User], Optional[str]]:
        """Create a new user account.
        
        Args:
            username: Username (must be unique)
            email: Email address (must be unique)
            password: Plain text password
            role: User role
            
        Returns:
            Tuple of (User object, error message)
        """
        # Validate password complexity
        is_valid, error = validate_password_complexity(password)
        if not is_valid:
            return None, error
        
        # Hash password
        password_hash = hash_password(password)
        
        # Create user
        user_id = str(uuid.uuid4())
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, username, email, password_hash, role.value)
            )
            conn.commit()
            
            # Fetch created user
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
                return None, "Failed to create user"
            
            user = self._row_to_user(row)
            return user, None
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return None, "Username already exists"
            elif "email" in str(e):
                return None, "Email already exists"
            else:
                return None, f"Database error: {e}"
        finally:
            conn.close()
    
    def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        role: Optional[UserRole] = None,
    ) -> tuple[Optional[User], Optional[str]]:
        """Update user account.
        
        Args:
            user_id: User ID to update
            username: New username (optional)
            email: New email (optional)
            password: New password (optional)
            role: New role (optional)
            
        Returns:
            Tuple of (User object, error message)
        """
        updates = []
        params = []
        
        if username is not None:
            updates.append("username = ?")
            params.append(username)
        
        if email is not None:
            updates.append("email = ?")
            params.append(email)
        
        if password is not None:
            # Validate password complexity
            is_valid, error = validate_password_complexity(password)
            if not is_valid:
                return None, error
            
            password_hash = hash_password(password)
            updates.append("password_hash = ?")
            params.append(password_hash)
        
        if role is not None:
            updates.append("role = ?")
            params.append(role.value)
        
        if not updates:
            return None, "No updates provided"
        
        params.append(user_id)
        
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                f"""
                UPDATE users
                SET {', '.join(updates)}
                WHERE id = ?
                """,
                params
            )
            
            if cursor.rowcount == 0:
                return None, "User not found"
            
            conn.commit()
            
            # Fetch updated user
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
                return None, "Failed to fetch updated user"
            
            user = self._row_to_user(row)
            return user, None
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return None, "Username already exists"
            elif "email" in str(e):
                return None, "Email already exists"
            else:
                return None, f"Database error: {e}"
        finally:
            conn.close()
    
    def deactivate_user(self, user_id: str) -> tuple[bool, Optional[str]]:
        """Deactivate user account.
        
        Args:
            user_id: User ID to deactivate
            
        Returns:
            Tuple of (success, error message)
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                UPDATE users
                SET is_active = 0
                WHERE id = ?
                """,
                (user_id,)
            )
            
            if cursor.rowcount == 0:
                return False, "User not found"
            
            conn.commit()
            return True, None
        finally:
            conn.close()
    
    def activate_user(self, user_id: str) -> tuple[bool, Optional[str]]:
        """Activate user account.
        
        Args:
            user_id: User ID to activate
            
        Returns:
            Tuple of (success, error message)
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                UPDATE users
                SET is_active = 1
                WHERE id = ?
                """,
                (user_id,)
            )
            
            if cursor.rowcount == 0:
                return False, "User not found"
            
            conn.commit()
            return True, None
        finally:
            conn.close()
    
    def list_users(self) -> list[User]:
        """List all users.
        
        Returns:
            List of User objects
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT id, username, email, password_hash, role, is_active,
                       failed_login_attempts, locked_until, created_at, last_login
                FROM users
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
            return [self._row_to_user(row) for row in rows]
        finally:
            conn.close()
    
    def get_user(self, user_id: str) -> Optional[User]:
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
