from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
from app.core.audit import actor, log_event
from app.core.config.settings import settings
from app.core.logging.logger import logger
from app.modules.auth.access_control import ensure_account_active
from services.access_email import send_access_request_admin
import asyncio

from app.core.exceptions.custom import AIStudioException
from app.core.security.jwt import create_access_token
from app.core.security.password import (
    hash_password,
    verify_password,
)

from app.modules.auth.constants import (
    SUPER_ADMIN,
    USER,
    SUPER_ADMIN_EMAILS,
)

from app.modules.auth.models import UserModel
from app.modules.auth.repository import AuthRepository
from app.modules.auth.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
)


class AuthService:
    """
    Business logic for Authentication.
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ):
        self.repository = AuthRepository(db)

    # --------------------------------------------------
    # Register User
    # --------------------------------------------------

    async def register(
        self,
        request: RegisterRequest,
    ) -> UserModel:

        existing_user = await self.repository.get_by_email(
            request.email
        )

        if existing_user:
            raise AIStudioException(
                message="Email already registered.",
                status_code=409,
                error_code="EMAIL_ALREADY_EXISTS",
            )

        # -----------------------------------------------
        # Assign Role
        # -----------------------------------------------

        role = (
            SUPER_ADMIN
            if request.email.lower() in SUPER_ADMIN_EMAILS
            else USER
        )

        # -----------------------------------------------
        # Create User
        # -----------------------------------------------

        user = UserModel(
            email=request.email,
            username=request.username,
            full_name=request.full_name,
            hashed_password=hash_password(
                request.password
            ),
            role=role,
            account_status="active" if role == SUPER_ADMIN else "pending_approval",
            is_active=role == SUPER_ADMIN,
            is_verified=role == SUPER_ADMIN,
        )

        try:
            created_user = await self.repository.create_user(user)
        except DuplicateKeyError as exc:
            raise AIStudioException(message="Email already registered.", status_code=409,
                                    error_code="EMAIL_ALREADY_EXISTS") from exc

        actor.set((created_user.id, created_user.role))
        if role != SUPER_ADMIN:
            try:
                await asyncio.to_thread(
                    send_access_request_admin, str(created_user.email), created_user.full_name,
                    settings.smtp_admin_recipients,
                )
            except Exception:
                logger.warning("Access request email could not be delivered")
        return created_user

    # --------------------------------------------------
    # Login User
    # --------------------------------------------------

    async def login(
        self,
        request: LoginRequest,
    ) -> TokenResponse:

        user = await self.repository.get_by_email(
            request.email
        )

        if not user:
            raise AIStudioException(
                message="Invalid email or password.",
                status_code=401,
                error_code="INVALID_CREDENTIALS",
            )

        if not verify_password(
            request.password,
            user.hashed_password,
        ):
            raise AIStudioException(
                message="Invalid email or password.",
                status_code=401,
                error_code="INVALID_CREDENTIALS",
            )

        ensure_account_active(user)

        actor.set((user.id, user.role))
        await log_event(
            "auth_logs", "login", "auth", owner_id=user.id,
            organization_id=user.organization_id, batch_id=user.batch_id,
        )

        access_token = create_access_token(
            {
                "sub": user.id,
                "email": user.email,
                "role": user.role,
            }
        )

        is_first_login = await self.repository.update_last_login(user.id)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            is_first_login=is_first_login,
        )
