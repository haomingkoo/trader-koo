"""Example of integrating RBAC system with FastAPI backend.

This example shows how to integrate the multi-user RBAC system
into the trader_koo backend.
"""

import os
from pathlib import Path
from fastapi import FastAPI
from trader_koo.auth.integration import initialize_rbac, create_default_admin_user

# Example integration in backend/main.py:

def setup_rbac_example():
    """Example of setting up RBAC in the main application."""
    
    # 1. Set required environment variables
    os.environ["JWT_SECRET_KEY"] = "your-secret-key-at-least-32-characters-long"
    os.environ["JWT_EXPIRATION_HOURS"] = "24"
    os.environ["EMAIL_TOKEN_EXPIRATION_DAYS"] = "7"
    
    # 2. Create FastAPI app
    app = FastAPI(title="trader_koo API")
    
    # 3. Initialize RBAC system
    db_path = Path("trader_koo.db")
    auth_service, user_mgmt_service = initialize_rbac(app, db_path)
    
    # 4. Create default admin user if no users exist
    create_default_admin_user(user_mgmt_service)
    
    # 5. RBAC is now integrated! The following endpoints are available:
    # - POST /api/login - Authenticate and get JWT token
    # - POST /api/admin/users - Create user (admin only)
    # - GET /api/admin/users - List users (admin only)
    # - GET /api/admin/users/{user_id} - Get user (admin only)
    # - PATCH /api/admin/users/{user_id} - Update user (admin only)
    # - POST /api/admin/users/{user_id}/deactivate - Deactivate user (admin only)
    
    return app


# Example API usage:

def example_api_usage():
    """Example of using the RBAC API endpoints."""
    
    import requests
    
    base_url = "http://localhost:8000"
    
    # 1. Login to get JWT token
    response = requests.post(
        f"{base_url}/api/login",
        json={
            "username": "admin",
            "password": "AdminPassword123!"
        }
    )
    token = response.json()["token"]
    
    # 2. Create a new analyst user
    response = requests.post(
        f"{base_url}/api/admin/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "username": "analyst1",
            "email": "analyst1@example.com",
            "password": "AnalystPassword123!",
            "role": "analyst"
        }
    )
    new_user = response.json()
    print(f"Created user: {new_user['username']} with role {new_user['role']}")
    
    # 3. List all users
    response = requests.get(
        f"{base_url}/api/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )
    users = response.json()
    print(f"Total users: {len(users)}")
    
    # 4. Update user role
    response = requests.patch(
        f"{base_url}/api/admin/users/{new_user['id']}",
        headers={"Authorization": f"Bearer {token}"},
        json={"role": "admin"}
    )
    updated_user = response.json()
    print(f"Updated user role to: {updated_user['role']}")
    
    # 5. Deactivate user
    response = requests.post(
        f"{base_url}/api/admin/users/{new_user['id']}/deactivate",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"User deactivated: {response.json()}")


# Example programmatic usage:

def example_programmatic_usage():
    """Example of using RBAC services programmatically."""
    
    from trader_koo.auth.service import AuthService
    from trader_koo.auth.user_management import UserManagementService
    from trader_koo.auth.models import UserRole
    
    db_path = Path("trader_koo.db")
    
    # Initialize services
    auth_service = AuthService(
        db_path=db_path,
        jwt_secret="your-secret-key-at-least-32-characters-long",
    )
    
    user_mgmt_service = UserManagementService(db_path=db_path)
    
    # Create a user
    user, error = user_mgmt_service.create_user(
        username="testuser",
        email="test@example.com",
        password="TestPassword123!",
        role=UserRole.ANALYST,
    )
    
    if error:
        print(f"Error creating user: {error}")
        return
    
    print(f"Created user: {user.username}")
    
    # Authenticate user
    authenticated_user = auth_service.authenticate_user(
        "testuser",
        "TestPassword123!"
    )
    
    if authenticated_user:
        print(f"Authentication successful for: {authenticated_user.username}")
        
        # Create JWT token
        token = auth_service.create_jwt(authenticated_user)
        print(f"JWT token: {token[:50]}...")
        
        # Check permissions
        can_write_analysis = authenticated_user.has_permission("/api/analysis", "write")
        can_access_admin = authenticated_user.has_permission("/api/admin/users", "read")
        
        print(f"Can write analysis: {can_write_analysis}")  # True for analyst
        print(f"Can access admin: {can_access_admin}")      # False for analyst
    else:
        print("Authentication failed")


if __name__ == "__main__":
    print("RBAC Integration Examples")
    print("=" * 60)
    print("\nThis file contains examples of integrating the RBAC system.")
    print("See the functions above for usage examples.")
    print("\nTo use RBAC in your application:")
    print("1. Set JWT_SECRET_KEY environment variable")
    print("2. Call initialize_rbac(app, db_path)")
    print("3. Call create_default_admin_user(user_mgmt_service)")
    print("\nFor more details, see trader_koo/auth/README.md")
