from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):

    email: EmailStr

    username: str = Field(min_length=3)

    full_name: str

    password: str = Field(min_length=8)


class LoginRequest(BaseModel):

    email: EmailStr

    password: str


class UserResponse(BaseModel):

    id: str

    email: EmailStr

    username: str

    full_name: str

    role: str

    is_active: bool

    is_verified: bool

    account_status: str = "active"

    organization_id: str | None = None

    course_id: str | None = None

    batch_id: str | None = None

    access_start_at: str | None = None

    access_end_at: str | None = None

    effective_modules: list[str] = Field(default_factory=list)


class TokenResponse(BaseModel):

    access_token: str

    token_type: str = "bearer"
