from fastapi import Depends

from app.core.dependencies.auth import get_current_user
from fastapi import APIRouter

from app.core.database.mongodb import MongoDB
from app.core.exceptions.custom import AIStudioException
from app.core.security.jwt import create_access_token
from app.shared.responses import APIResponse
router = APIRouter(prefix="/system", tags=["System"])


@router.get(
    "/health/database",
    response_model=APIResponse,
)
async def database_health():

    await MongoDB.client.admin.command("ping")

    return APIResponse(
        success=True,
        message="Database connection is healthy.",
        data={
            "database": "healthy"
        },
    )
@router.get("/test-error")
async def test_error():

    raise AIStudioException(
        message="This is a test exception.",
        status_code=400,
        error_code="TEST_ERROR",
    )
@router.get("/protected")
async def protected_route(
    current_user=Depends(get_current_user),
):

    return {
        "authenticated": True,
        "user": current_user,
    }
@router.get("/token")
async def generate_test_token():

    token = create_access_token(
        {
            "sub": "admin",
            "role": "super_admin",
        }
    )

    return {
        "access_token": token
    }