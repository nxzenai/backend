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


class TokenResponse(BaseModel):

    access_token: str

    token_type: str = "bearer"