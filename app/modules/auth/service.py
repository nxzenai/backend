from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.exceptions.custom import AIStudioException
from app.core.security.jwt import create_access_token
from app.core.security.password import (
    hash_password,
    verify_password,
    needs_rehash,
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
        )

        created_user = await self.repository.create_user(
            user
        )

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

        # -----------------------------------------------
        # User Not Found
        # -----------------------------------------------

        if not user:
            raise AIStudioException(
                message="Invalid email or password.",
                status_code=401,
                error_code="INVALID_CREDENTIALS",
            )

        # -----------------------------------------------
        # Verify Password
        # -----------------------------------------------

        password_valid = verify_password(
            request.password,
            user.hashed_password,
        )

        if not password_valid:
            raise AIStudioException(
                message="Invalid email or password.",
                status_code=401,
                error_code="INVALID_CREDENTIALS",
            )

        # -----------------------------------------------
        # Account Status
        # -----------------------------------------------

        if not user.is_active:
            raise AIStudioException(
                message="User account is disabled.",
                status_code=403,
                error_code="ACCOUNT_DISABLED",
            )

        # -----------------------------------------------
        # Password Migration
        #
        # If this user was created using bcrypt,
        # automatically migrate the password to Argon2.
        # -----------------------------------------------

        if needs_rehash(user.hashed_password):

            new_password_hash = hash_password(
                request.password
            )

            await self.repository.update_password(
                user.id,
                new_password_hash,
            )

        # -----------------------------------------------
        # Update Last Login
        # -----------------------------------------------

        await self.repository.update_last_login(
            user.id
        )

        # -----------------------------------------------
        # Create Access Token
        # -----------------------------------------------

        access_token = create_access_token(
            {
                "sub": user.id,
                "email": user.email,
                "role": user.role,
            }
        )

        # -----------------------------------------------
        # Return Token
        # -----------------------------------------------

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
        )

