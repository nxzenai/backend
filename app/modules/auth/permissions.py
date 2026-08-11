from fastapi import Depends

from app.core.exceptions.custom import AIStudioException
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel


async def require_super_admin(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:

    if current_user.role != "super_admin":
        raise AIStudioException(
            message="Access denied.",
            status_code=403,
            error_code="INSUFFICIENT_PERMISSIONS",
        )

    return current_user


async def require_admin(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:

    if current_user.role not in (
        "admin",
        "super_admin",
    ):
        raise AIStudioException(
            message="Access denied.",
            status_code=403,
            error_code="INSUFFICIENT_PERMISSIONS",
        )

    return current_user