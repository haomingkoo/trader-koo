"""API endpoints for authentication and user management."""

from typing import Optional
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel, EmailStr

from trader_koo.auth.models import UserRole, User
from trader_koo.auth.service import AuthService
from trader_koo.auth.user_management import UserManagementService
from trader_koo.audit.logger import AuditLogger


# Request/Response models
class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""
    token: str
    user: dict


class CreateUserRequest(BaseModel):
    """Create user request."""
    username: str
    email: EmailStr
    password: str
    role: str


class UpdateUserRequest(BaseModel):
    """Update user request."""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[str] = None


class UserResponse(BaseModel):
    """User response."""
    id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str]


def create_auth_router(
    auth_service: AuthService,
    user_mgmt_service: UserManagementService,
    audit_logger: AuditLogger,
) -> APIRouter:
    """Create authentication and user management router.
    
    Args:
        auth_service: Authentication service
        user_mgmt_service: User management service
        audit_logger: Audit logger
        
    Returns:
        FastAPI router
    """
    router = APIRouter()
    
    @router.post("/api/login", response_model=LoginResponse)
    async def login(request: Request, body: LoginRequest):
        """Authenticate user and return JWT token."""
        user = auth_service.authenticate_user(body.username, body.password)
        if not user:
            # Log failed authentication
            audit_logger.log_event(
                event_type="auth_failure",
                user_id=None,
                resource="/api/login",
                action="login",
                details={"username": body.username},
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create JWT token
        token = auth_service.create_jwt(
            user,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        # Log successful authentication
        audit_logger.log_event(
            event_type="auth_success",
            user_id=user.id,
            resource="/api/login",
            action="login",
            details={"username": user.username},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        return LoginResponse(
            token=token,
            user={
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            }
        )
    
    @router.post("/api/admin/users", response_model=UserResponse)
    async def create_user(request: Request, body: CreateUserRequest):
        """Create a new user (admin only)."""
        # Check if requester is admin
        user = getattr(request.state, "user", None)
        if not user or user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Validate role
        try:
            role = UserRole(body.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {body.role}"
            )
        
        # Create user
        new_user, error = user_mgmt_service.create_user(
            username=body.username,
            email=body.email,
            password=body.password,
            role=role,
        )
        
        if error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )
        
        # Log user creation (Requirement 16.9)
        audit_logger.log_event(
            event_type="admin_action",
            user_id=user.id,
            resource="/api/admin/users",
            action="create_user",
            details={
                "new_user_id": new_user.id,
                "new_user_username": new_user.username,
                "new_user_role": new_user.role.value,
            },
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            role=new_user.role.value,
            is_active=new_user.is_active,
            created_at=new_user.created_at.isoformat(),
            last_login=new_user.last_login.isoformat() if new_user.last_login else None,
        )
    
    @router.patch("/api/admin/users/{user_id}", response_model=UserResponse)
    async def update_user(request: Request, user_id: str, body: UpdateUserRequest):
        """Update user (admin only)."""
        # Check if requester is admin
        user = getattr(request.state, "user", None)
        if not user or user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Validate role if provided
        role = None
        if body.role:
            try:
                role = UserRole(body.role)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid role: {body.role}"
                )
        
        # Get old user for audit log
        old_user = user_mgmt_service.get_user(user_id)
        if not old_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update user
        updated_user, error = user_mgmt_service.update_user(
            user_id=user_id,
            username=body.username,
            email=body.email,
            password=body.password,
            role=role,
        )
        
        if error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )
        
        # Log role change if role was updated (Requirement 16.9)
        if role and old_user.role != role:
            audit_logger.log_event(
                event_type="role_change",
                user_id=user.id,
                resource=f"/api/admin/users/{user_id}",
                action="update_role",
                details={
                    "target_user_id": user_id,
                    "target_username": updated_user.username,
                    "old_role": old_user.role.value,
                    "new_role": updated_user.role.value,
                },
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
        
        # Log user update
        audit_logger.log_event(
            event_type="admin_action",
            user_id=user.id,
            resource=f"/api/admin/users/{user_id}",
            action="update_user",
            details={
                "target_user_id": user_id,
                "updated_fields": [k for k, v in body.dict().items() if v is not None],
            },
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        return UserResponse(
            id=updated_user.id,
            username=updated_user.username,
            email=updated_user.email,
            role=updated_user.role.value,
            is_active=updated_user.is_active,
            created_at=updated_user.created_at.isoformat(),
            last_login=updated_user.last_login.isoformat() if updated_user.last_login else None,
        )
    
    @router.post("/api/admin/users/{user_id}/deactivate")
    async def deactivate_user(request: Request, user_id: str):
        """Deactivate user (admin only)."""
        # Check if requester is admin
        user = getattr(request.state, "user", None)
        if not user or user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Get user for audit log
        target_user = user_mgmt_service.get_user(user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Deactivate user
        success, error = user_mgmt_service.deactivate_user(user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error or "Failed to deactivate user"
            )
        
        # Log deactivation
        audit_logger.log_event(
            event_type="admin_action",
            user_id=user.id,
            resource=f"/api/admin/users/{user_id}/deactivate",
            action="deactivate_user",
            details={
                "target_user_id": user_id,
                "target_username": target_user.username,
            },
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        return {"success": True}
    
    @router.get("/api/admin/users", response_model=list[UserResponse])
    async def list_users(request: Request):
        """List all users (admin only)."""
        # Check if requester is admin
        user = getattr(request.state, "user", None)
        if not user or user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        users = user_mgmt_service.list_users()
        return [
            UserResponse(
                id=u.id,
                username=u.username,
                email=u.email,
                role=u.role.value,
                is_active=u.is_active,
                created_at=u.created_at.isoformat(),
                last_login=u.last_login.isoformat() if u.last_login else None,
            )
            for u in users
        ]
    
    @router.get("/api/admin/users/{user_id}", response_model=UserResponse)
    async def get_user(request: Request, user_id: str):
        """Get user by ID (admin only)."""
        # Check if requester is admin
        user = getattr(request.state, "user", None)
        if not user or user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        target_user = user_mgmt_service.get_user(user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=target_user.id,
            username=target_user.username,
            email=target_user.email,
            role=target_user.role.value,
            is_active=target_user.is_active,
            created_at=target_user.created_at.isoformat(),
            last_login=target_user.last_login.isoformat() if target_user.last_login else None,
        )
    
    return router
