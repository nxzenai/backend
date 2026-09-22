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
        "trainer",
        "trainee",
        "guest",
        "user",
    ] = "user"

    # --------------------------------------------------
    # Organization
    # --------------------------------------------------

    organization_id: str | None = None
    course_id: str | None = None
    batch_id: str | None = None

    allowed_modules: list[str] = Field(default_factory=list)
    denied_modules: list[str] = Field(default_factory=list)
    effective_modules: list[str] = Field(default_factory=list)

    # --------------------------------------------------
    # Account Status
    # --------------------------------------------------

    is_active: bool = True

    is_verified: bool = False

    account_status: Literal[
        "pending_approval", "active", "rejected", "suspended", "expired", "deleted"
    ] = "active"
    access_start_at: datetime | None = None
    access_end_at: datetime | None = None
    deleted_at: datetime | None = None

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
