# Multi-User RBAC System

This module implements comprehensive role-based access control (RBAC) for the trader_koo platform.

## Features

- **Three User Roles** (Requirement 16.1):

  - **Admin**: Full access to all endpoints including user management
  - **Analyst**: Read/write access to analysis and reports, no admin access
  - **Viewer**: Read-only access to dashboards and reports

- **Multiple Authentication Methods** (Requirement 16.3):

  - JWT tokens (24-hour expiration)
  - API keys (legacy support)
  - Email tokens (7-day expiration)

- **Password Security** (Requirement 16.10):

  - Minimum 12 characters
  - Must contain uppercase, lowercase, number, and special character
  - Bcrypt hashing with 12 rounds
  - Account lockout after 5 failed attempts (15-minute cooldown)

- **Audit Logging** (Requirement 16.9):
  - All authentication attempts logged
  - Role changes logged
  - User management actions logged

## Configuration

Set these environment variables:

```bash
# Required
JWT_SECRET_KEY=<random-256-bit-key>  # Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'

# Optional
JWT_ALGORITHM=HS256                   # Default: HS256
JWT_EXPIRATION_HOURS=24               # Default: 24
EMAIL_TOKEN_EXPIRATION_DAYS=7         # Default: 7

# Legacy API key (for backward compatibility)
TRADER_KOO_API_KEY=<your-api-key>

# Default admin user (created if no users exist)
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_EMAIL=admin@trader-koo.local
DEFAULT_ADMIN_PASSWORD=<secure-password>  # If not set, random password is generated
```

## Database Schema

The RBAC system creates three tables:

### users

- `id`: UUID primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Bcrypt hashed password
- `role`: User role (admin, analyst, viewer)
- `is_active`: Account active status
- `failed_login_attempts`: Failed login counter
- `locked_until`: Account lock expiration
- `created_at`: Account creation timestamp
- `last_login`: Last successful login timestamp

### sessions

- `id`: UUID primary key
- `user_id`: Foreign key to users
- `token_hash`: SHA256 hash of JWT token
- `expires_at`: Session expiration
- `created_at`: Session creation timestamp
- `last_activity`: Last activity timestamp
- `ip_address`: Client IP address
- `user_agent`: Client user agent

### email_tokens

- `id`: UUID primary key
- `user_id`: Foreign key to users
- `token_hash`: SHA256 hash of email token
- `expires_at`: Token expiration (7 days)
- `created_at`: Token creation timestamp
- `used_at`: Token usage timestamp
- `revoked_at`: Token revocation timestamp

## API Endpoints

### Authentication

#### POST /api/login

Authenticate user and return JWT token.

**Request:**

```json
{
  "username": "testuser",
  "password": "TestPassword123!"
}
```

**Response:**

```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "user": {
    "id": "uuid",
    "username": "testuser",
    "email": "test@example.com",
    "role": "analyst"
  }
}
```

### User Management (Admin Only)

#### POST /api/admin/users

Create a new user.

**Request:**

```json
{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "SecurePassword123!",
  "role": "analyst"
}
```

**Response:**

```json
{
  "id": "uuid",
  "username": "newuser",
  "email": "newuser@example.com",
  "role": "analyst",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00",
  "last_login": null
}
```

#### GET /api/admin/users

List all users.

**Response:**

```json
[
  {
    "id": "uuid",
    "username": "admin",
    "email": "admin@example.com",
    "role": "admin",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00",
    "last_login": "2024-01-15T10:00:00"
  }
]
```

#### GET /api/admin/users/{user_id}

Get user by ID.

#### PATCH /api/admin/users/{user_id}

Update user.

**Request:**

```json
{
  "username": "updatedname",
  "email": "newemail@example.com",
  "password": "NewPassword123!",
  "role": "admin"
}
```

#### POST /api/admin/users/{user_id}/deactivate

Deactivate user account.

## Usage Examples

### Creating Users

```python
from trader_koo.auth.user_management import UserManagementService
from trader_koo.auth.models import UserRole

user_mgmt = UserManagementService(db_path="trader_koo.db")

# Create admin user
admin, error = user_mgmt.create_user(
    username="admin",
    email="admin@example.com",
    password="AdminPassword123!",
    role=UserRole.ADMIN,
)

# Create analyst user
analyst, error = user_mgmt.create_user(
    username="analyst",
    email="analyst@example.com",
    password="AnalystPassword123!",
    role=UserRole.ANALYST,
)
```

### Authenticating Users

```python
from trader_koo.auth.service import AuthService

auth_service = AuthService(
    db_path="trader_koo.db",
    jwt_secret="your-secret-key-at-least-32-chars",
)

# Authenticate with username/password
user = auth_service.authenticate_user("admin", "AdminPassword123!")
if user:
    # Create JWT token
    token = auth_service.create_jwt(user)
    print(f"Token: {token}")
```

### Checking Permissions

```python
# Check if user has permission
if user.has_permission("/api/admin/users", "write"):
    print("User can create/update users")
else:
    print("User does not have permission")
```

### Using Authentication in API Requests

```bash
# Login to get token
curl -X POST http://localhost:8000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "AdminPassword123!"}'

# Use token in subsequent requests
curl -X GET http://localhost:8000/api/admin/users \
  -H "Authorization: Bearer <token>"

# Or use legacy API key
curl -X GET http://localhost:8000/api/admin/users \
  -H "X-API-Key: <api-key>"
```

## Integration with FastAPI

To integrate RBAC with your FastAPI application:

```python
from fastapi import FastAPI
from trader_koo.auth.integration import initialize_rbac, create_default_admin_user

app = FastAPI()

# Initialize RBAC
auth_service, user_mgmt_service = initialize_rbac(app, db_path="trader_koo.db")

# Create default admin user if no users exist
create_default_admin_user(user_mgmt_service)
```

## Security Considerations

1. **JWT Secret**: Use a strong random secret key (at least 32 characters)
2. **Password Storage**: Passwords are hashed with bcrypt (12 rounds)
3. **Account Lockout**: Accounts are locked after 5 failed login attempts
4. **Token Expiration**: JWT tokens expire after 24 hours, email tokens after 7 days
5. **Audit Logging**: All authentication and authorization events are logged
6. **HTTPS**: Always use HTTPS in production to protect tokens in transit

## Testing

Run the test suite:

```bash
pytest tests/test_rbac_auth.py -v
```

The test suite covers:

- API key authentication
- JWT authentication
- Email token authentication
- Token revocation
- Password authentication
- Account lockout
- Role-based permissions
- Password complexity validation
- User management operations
