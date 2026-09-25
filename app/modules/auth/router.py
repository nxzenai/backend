from app.modules.auth.models import UserModel
from fastapi import APIRouter, Depends, status

from app.modules.auth.dependencies import get_auth_service
from app.modules.auth.schemas import RegisterRequest, UserResponse
from app.modules.auth.service import AuthService

from app.shared.responses import APIResponse
from app.modules.auth.schemas import (
    LoginRequest,
    TokenResponse,
)
from app.modules.auth.dependencies import (
    get_auth_service,
    get_current_user,
)
from app.core.audit import log_event

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=APIResponse,
)
async def register_user(
    request: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
):

    user = await service.register(request)

    return APIResponse(
        success=True,
        message=("User registered successfully." if user.role == "super_admin" else
                 "Registration received. Your account is pending administrator approval."),
        data=UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            account_status=user.account_status,
        ).model_dump(),
    )
@router.post(
    "/login",
    response_model=APIResponse[TokenResponse],
    summary="Login User",
)
async def login(
    request: LoginRequest,
    service: AuthService = Depends(get_auth_service),
):

    token = await service.login(request)

    return APIResponse(
        success=True,
        message="Login successful.",
        data=token,
    )
@router.get(
    "/me",
    response_model=APIResponse[UserResponse],
    summary="Get Current User",
)
async def me(
    current_user: UserModel = Depends(get_current_user),
):

    return APIResponse(
        success=True,
        message="Current user retrieved successfully.",
        data=UserResponse(
            id=current_user.id,
            email=current_user.email,
            username=current_user.username,
            full_name=current_user.full_name,
            role=current_user.role,
            is_active=current_user.is_active,
            is_verified=current_user.is_verified,
            account_status=current_user.account_status,
            organization_id=current_user.organization_id,
            course_id=current_user.course_id,
            batch_id=current_user.batch_id,
            access_start_at=current_user.access_start_at.isoformat() if current_user.access_start_at else None,
            access_end_at=current_user.access_end_at.isoformat() if current_user.access_end_at else None,
            effective_modules=current_user.effective_modules,
        ),
    )


@router.post("/logout", response_model=APIResponse)
async def logout(current_user: UserModel = Depends(get_current_user)):
    await log_event(
        "auth_logs", "logout", "auth", owner_id=current_user.id,
        organization_id=current_user.organization_id, batch_id=current_user.batch_id,
    )
    return APIResponse(success=True, message="Logged out.", data=None)
