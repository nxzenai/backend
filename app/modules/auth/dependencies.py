from fastapi import Depends
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.core.exceptions.custom import AIStudioException
from app.core.security.jwt import decode_access_token

from app.modules.auth.models import UserModel
from app.modules.auth.repository import AuthRepository
from app.modules.auth.service import AuthService

# --------------------------------------------------
# HTTP Bearer Security
# --------------------------------------------------

security = HTTPBearer(auto_error=True)


# --------------------------------------------------
# Auth Service Dependency
# --------------------------------------------------

def get_auth_service(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> AuthService:
    return AuthService(db)


# --------------------------------------------------
# Current Authenticated User
# --------------------------------------------------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> UserModel:
    """
    Returns the currently authenticated user.
    """

    token = credentials.credentials

    payload = decode_access_token(token)

    if payload is None:
        raise AIStudioException(
            message="Invalid or expired token.",
            status_code=401,
            error_code="INVALID_TOKEN",
        )

    user_id = payload.get("sub")

    if not user_id:
        raise AIStudioException(
            message="Invalid token.",
            status_code=401,
            error_code="INVALID_TOKEN",
        )

    repository = AuthRepository(db)

    user = await repository.get_by_id(user_id)

    if user is None:
        raise AIStudioException(
            message="User not found.",
            status_code=404,
            error_code="USER_NOT_FOUND",
        )

    if not user.is_active:
        raise AIStudioException(
            message="User account is disabled.",
            status_code=403,
            error_code="ACCOUNT_DISABLED",
        )

    return user