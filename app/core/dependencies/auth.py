from fastapi import Depends

from app.core.exceptions.custom import AIStudioException
from app.core.security.jwt import decode_access_token
from app.core.security.oauth import oauth2_scheme


async def get_current_user(
    token: str = Depends(oauth2_scheme),
):
    payload = decode_access_token(token)

    if payload is None:
        raise AIStudioException(
            message="Invalid authentication token.",
            status_code=401,
            error_code="INVALID_TOKEN",
        )

    return payload