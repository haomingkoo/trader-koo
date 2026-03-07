"""Unit tests for RBAC authentication."""

import sqlite3
import tempfile
from pathlib import Path
import pytest

from trader_koo.auth.schema import ensure_auth_schema
from trader_koo.auth.service import AuthService
from trader_koo.auth.models import UserRole
from trader_koo.auth.user_management import UserManagementService


@pytest.fixture
def db_path():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    # Initialize schema
    ensure_auth_schema(db_path)
    
    yield db_path
    
    # Cleanup
    db_path.unlink()


@pytest.fixture
def auth_service(db_path):
    """Create auth service for testing."""
    return AuthService(
        db_path=db_path,
        jwt_secret="test-secret-key-at-least-32-chars-long",
        api_key="test-api-key-at-least-32-chars-long",
    )


@pytest.fixture
def user_mgmt_service(db_path):
    """Create user management service for testing."""
    return UserManagementService(db_path=db_path)


class TestAuthentication:
    """Test authentication methods."""
    
    def test_api_key_authentication(self, auth_service):
        """Test API key authentication."""
        # Valid API key
        user = auth_service.authenticate_api_key("test-api-key-at-least-32-chars-long")
        assert user is not None
        assert user.role == UserRole.ADMIN
        
        # Invalid API key
        user = auth_service.authenticate_api_key("invalid-key")
        assert user is None
    
    def test_jwt_authentication(self, auth_service, user_mgmt_service):
        """Test JWT token authentication."""
        # Create a test user
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert error is None
        
        # Create JWT token
        token = auth_service.create_jwt(user)
        assert token is not None
        
        # Authenticate with token
        authenticated_user = auth_service.authenticate_jwt(token)
        assert authenticated_user is not None
        assert authenticated_user.id == user.id
        assert authenticated_user.username == user.username
        
        # Invalid token
        invalid_user = auth_service.authenticate_jwt("invalid-token")
        assert invalid_user is None
    
    def test_email_token_authentication(self, auth_service, user_mgmt_service):
        """Test email token authentication."""
        # Create a test user
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.VIEWER,
        )
        assert error is None
        
        # Create email token
        token = auth_service.create_email_token(user)
        assert token is not None
        
        # Authenticate with token
        authenticated_user = auth_service.authenticate_email_token(token)
        assert authenticated_user is not None
        assert authenticated_user.id == user.id
        
        # Token should be marked as used, can't use again
        second_auth = auth_service.authenticate_email_token(token)
        assert second_auth is None
    
    def test_email_token_revocation(self, auth_service, user_mgmt_service):
        """Test email token revocation."""
        # Create a test user
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.VIEWER,
        )
        assert error is None
        
        # Create email token
        token = auth_service.create_email_token(user)
        
        # Revoke token
        success = auth_service.revoke_token(token)
        assert success is True
        
        # Can't authenticate with revoked token
        authenticated_user = auth_service.authenticate_email_token(token)
        assert authenticated_user is None
    
    def test_user_password_authentication(self, auth_service, user_mgmt_service):
        """Test username/password authentication."""
        # Create a test user
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert error is None
        
        # Authenticate with correct password
        authenticated_user = auth_service.authenticate_user("testuser", "TestPassword123!")
        assert authenticated_user is not None
        assert authenticated_user.username == "testuser"
        
        # Authenticate with wrong password
        failed_auth = auth_service.authenticate_user("testuser", "WrongPassword")
        assert failed_auth is None
    
    def test_account_lockout(self, auth_service, user_mgmt_service):
        """Test account lockout after failed login attempts."""
        # Create a test user
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert error is None
        
        # Fail 5 times
        for _ in range(5):
            result = auth_service.authenticate_user("testuser", "WrongPassword")
            assert result is None
        
        # Account should be locked now
        locked_user = auth_service.get_user_by_username("testuser")
        assert locked_user.is_locked() is True
        
        # Can't authenticate even with correct password
        result = auth_service.authenticate_user("testuser", "TestPassword123!")
        assert result is None


class TestAuthorization:
    """Test role-based authorization."""
    
    def test_admin_permissions(self, user_mgmt_service):
        """Test admin role has full access."""
        user, error = user_mgmt_service.create_user(
            username="admin",
            email="admin@example.com",
            password="AdminPassword123!",
            role=UserRole.ADMIN,
        )
        assert error is None
        
        # Admin can access everything
        assert user.has_permission("/api/admin/users", "read") is True
        assert user.has_permission("/api/admin/users", "write") is True
        assert user.has_permission("/api/admin/users", "delete") is True
        assert user.has_permission("/api/analysis", "read") is True
        assert user.has_permission("/api/analysis", "write") is True
    
    def test_analyst_permissions(self, user_mgmt_service):
        """Test analyst role permissions."""
        user, error = user_mgmt_service.create_user(
            username="analyst",
            email="analyst@example.com",
            password="AnalystPassword123!",
            role=UserRole.ANALYST,
        )
        assert error is None
        
        # Analyst can read/write analysis
        assert user.has_permission("/api/analysis", "read") is True
        assert user.has_permission("/api/analysis", "write") is True
        assert user.has_permission("/api/reports", "read") is True
        assert user.has_permission("/api/reports", "write") is True
        
        # Analyst cannot access admin endpoints
        assert user.has_permission("/api/admin/users", "read") is False
        assert user.has_permission("/api/admin/users", "write") is False
    
    def test_viewer_permissions(self, user_mgmt_service):
        """Test viewer role is read-only."""
        user, error = user_mgmt_service.create_user(
            username="viewer",
            email="viewer@example.com",
            password="ViewerPassword123!",
            role=UserRole.VIEWER,
        )
        assert error is None
        
        # Viewer can only read
        assert user.has_permission("/api/dashboard", "read") is True
        assert user.has_permission("/api/dashboard", "write") is False
        assert user.has_permission("/api/analysis", "read") is True
        assert user.has_permission("/api/analysis", "write") is False
        
        # Viewer cannot access admin endpoints
        assert user.has_permission("/api/admin/users", "read") is False


class TestRoleEnforcement:
    """Test role enforcement in user management."""
    
    def test_create_user_with_different_roles(self, user_mgmt_service):
        """Test creating users with different roles."""
        # Create admin
        admin, error = user_mgmt_service.create_user(
            username="admin",
            email="admin@example.com",
            password="AdminPassword123!",
            role=UserRole.ADMIN,
        )
        assert error is None
        assert admin.role == UserRole.ADMIN
        
        # Create analyst
        analyst, error = user_mgmt_service.create_user(
            username="analyst",
            email="analyst@example.com",
            password="AnalystPassword123!",
            role=UserRole.ANALYST,
        )
        assert error is None
        assert analyst.role == UserRole.ANALYST
        
        # Create viewer
        viewer, error = user_mgmt_service.create_user(
            username="viewer",
            email="viewer@example.com",
            password="ViewerPassword123!",
            role=UserRole.VIEWER,
        )
        assert error is None
        assert viewer.role == UserRole.VIEWER
    
    def test_update_user_role(self, user_mgmt_service):
        """Test updating user role."""
        # Create user as viewer
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.VIEWER,
        )
        assert error is None
        assert user.role == UserRole.VIEWER
        
        # Update to analyst
        updated_user, error = user_mgmt_service.update_user(
            user_id=user.id,
            role=UserRole.ANALYST,
        )
        assert error is None
        assert updated_user.role == UserRole.ANALYST
        
        # Update to admin
        updated_user, error = user_mgmt_service.update_user(
            user_id=user.id,
            role=UserRole.ADMIN,
        )
        assert error is None
        assert updated_user.role == UserRole.ADMIN


class TestPasswordValidation:
    """Test password complexity validation."""
    
    def test_password_length_requirement(self, user_mgmt_service):
        """Test minimum 12 character requirement."""
        # Too short
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="Short1!",
            role=UserRole.VIEWER,
        )
        assert user is None
        assert "12 characters" in error
        
        # Long enough
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="LongEnough1!",
            role=UserRole.VIEWER,
        )
        assert user is not None
        assert error is None
    
    def test_password_uppercase_requirement(self, user_mgmt_service):
        """Test uppercase letter requirement."""
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="nouppercase123!",
            role=UserRole.VIEWER,
        )
        assert user is None
        assert "uppercase" in error
    
    def test_password_lowercase_requirement(self, user_mgmt_service):
        """Test lowercase letter requirement."""
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="NOLOWERCASE123!",
            role=UserRole.VIEWER,
        )
        assert user is None
        assert "lowercase" in error
    
    def test_password_number_requirement(self, user_mgmt_service):
        """Test number requirement."""
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="NoNumbersHere!",
            role=UserRole.VIEWER,
        )
        assert user is None
        assert "number" in error
    
    def test_password_special_char_requirement(self, user_mgmt_service):
        """Test special character requirement."""
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="NoSpecialChar123",
            role=UserRole.VIEWER,
        )
        assert user is None
        assert "special character" in error
    
    def test_valid_password(self, user_mgmt_service):
        """Test valid password meeting all requirements."""
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPassword123!",
            role=UserRole.VIEWER,
        )
        assert user is not None
        assert error is None


class TestUserManagement:
    """Test user management operations."""
    
    def test_create_user(self, user_mgmt_service):
        """Test user creation."""
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert user is not None
        assert error is None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.ANALYST
        assert user.is_active is True
    
    def test_duplicate_username(self, user_mgmt_service):
        """Test duplicate username rejection."""
        # Create first user
        user1, error = user_mgmt_service.create_user(
            username="testuser",
            email="test1@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert user1 is not None
        
        # Try to create with same username
        user2, error = user_mgmt_service.create_user(
            username="testuser",
            email="test2@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert user2 is None
        assert "Username already exists" in error
    
    def test_duplicate_email(self, user_mgmt_service):
        """Test duplicate email rejection."""
        # Create first user
        user1, error = user_mgmt_service.create_user(
            username="testuser1",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert user1 is not None
        
        # Try to create with same email
        user2, error = user_mgmt_service.create_user(
            username="testuser2",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert user2 is None
        assert "Email already exists" in error
    
    def test_update_user(self, user_mgmt_service):
        """Test user update."""
        # Create user
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.VIEWER,
        )
        assert user is not None
        
        # Update user
        updated_user, error = user_mgmt_service.update_user(
            user_id=user.id,
            username="newusername",
            email="newemail@example.com",
            role=UserRole.ANALYST,
        )
        assert updated_user is not None
        assert error is None
        assert updated_user.username == "newusername"
        assert updated_user.email == "newemail@example.com"
        assert updated_user.role == UserRole.ANALYST
    
    def test_deactivate_user(self, user_mgmt_service):
        """Test user deactivation."""
        # Create user
        user, error = user_mgmt_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.ANALYST,
        )
        assert user is not None
        assert user.is_active is True
        
        # Deactivate user
        success, error = user_mgmt_service.deactivate_user(user.id)
        assert success is True
        assert error is None
        
        # Verify user is deactivated
        deactivated_user = user_mgmt_service.get_user(user.id)
        assert deactivated_user.is_active is False
    
    def test_list_users(self, user_mgmt_service):
        """Test listing all users."""
        # Create multiple users
        for i in range(3):
            user_mgmt_service.create_user(
                username=f"user{i}",
                email=f"user{i}@example.com",
                password="TestPassword123!",
                role=UserRole.VIEWER,
            )
        
        # List users
        users = user_mgmt_service.list_users()
        assert len(users) == 3
        assert all(isinstance(u.role, UserRole) for u in users)
