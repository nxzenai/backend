from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
)


class UserModel(BaseModel):
    """
    MongoDB User Document
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    id: str | None = None

    email: EmailStr

    username: str

    full_name: str

    hashed_password: str

    # --------------------------------------------------
    # RBAC
    # --------------------------------------------------

    role: Literal[
        "super_admin",
        "admin",
        "user",
    ] = "user"

    # --------------------------------------------------
    # Organization
    # --------------------------------------------------

    organization_id: str | None = None

    # --------------------------------------------------
    # Account Status
    # --------------------------------------------------

    is_active: bool = True

    is_verified: bool = False

    # --------------------------------------------------
    # Audit Fields
    # --------------------------------------------------

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    last_login: datetime | None = None