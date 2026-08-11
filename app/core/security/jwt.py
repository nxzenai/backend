from datetime import datetime, timedelta, UTC

from jose import JWTError, jwt

from app.core.config.settings import settings


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
):
    payload = data.copy()

    expire = datetime.now(UTC) + (
        expires_delta
        if expires_delta
        else timedelta(
            minutes=settings.access_token_expire_minutes
        )
    )

    payload.update({"exp": expire})

    return jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def decode_access_token(token: str) -> dict | None:
    """
    Decode and validate a JWT access token.
    """

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )

        return payload

    except JWTError:
        return None