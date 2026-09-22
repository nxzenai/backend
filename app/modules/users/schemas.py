from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, model_validator


Role = Literal["super_admin", "admin", "trainer", "trainee", "guest"]
Status = Literal["pending_approval", "active", "rejected", "suspended", "expired"]


class AccessAssignment(BaseModel):
    role: Role
    organization_id: str | None = None
    course_id: str | None = None
    batch_id: str | None = None
    allowed_modules: list[str] = Field(default_factory=list)
    denied_modules: list[str] = Field(default_factory=list)
    access_start_at: datetime | None = None
    access_end_at: datetime | None = None

    @model_validator(mode="after")
    def validate_dates(self):
        if self.access_start_at and self.access_end_at and self.access_start_at >= self.access_end_at:
            raise ValueError("Access end must be after access start")
        if self.role == "guest" and not self.access_end_at:
            raise ValueError("Guest access must have an end date")
        return self


class ApprovalRequest(AccessAssignment):
    pass


class RejectionRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


class UserCreate(AccessAssignment):
    email: EmailStr
    username: str = Field(min_length=3, max_length=80)
    full_name: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=8, max_length=200)


class BulkUsers(BaseModel):
    users: list[UserCreate] = Field(min_length=1, max_length=500)


class UserUpdate(BaseModel):
    role: Role | None = None
    organization_id: str | None = None
    course_id: str | None = None
    batch_id: str | None = None
    allowed_modules: list[str] | None = None
    denied_modules: list[str] | None = None
    access_start_at: datetime | None = None
    access_end_at: datetime | None = None


class LifecycleRequest(BaseModel):
    action: Literal["suspend", "reactivate", "expire", "extend", "reset_access"]
    access_end_at: datetime | None = None


class OrganizationUpsert(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    enabled_modules: list[str] = Field(default_factory=list)
    user_limit: int = Field(default=1, ge=1, le=100000)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    organization_type: str = Field(default="customer", max_length=80)
    plan: str = Field(default="standard", max_length=80)
    is_active: bool = True

    @model_validator(mode="after")
    def validate_dates(self):
        if self.valid_from and self.valid_until and self.valid_from >= self.valid_until:
            raise ValueError("Organization validity end must be after start")
        return self


class CourseUpsert(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    organization_id: str | None = None
    default_modules: list[str] = Field(default_factory=list)
    default_duration_days: int = Field(default=30, ge=1, le=3650)
    description: str = Field(default="", max_length=2000)
    is_active: bool = True


class BatchUpsert(BaseModel):
    organization_id: str
    course_id: str
    name: str = Field(min_length=1, max_length=150)
    start_at: datetime
    end_at: datetime
    trainer_ids: list[str] = Field(default_factory=list)
    max_students: int = Field(default=1, ge=1, le=100000)
    allowed_modules: list[str] = Field(default_factory=list)
    access_end_at: datetime
    is_active: bool = True

    @model_validator(mode="after")
    def validate_dates(self):
        if self.start_at >= self.end_at:
            raise ValueError("Batch end must be after start")
        if self.access_end_at < self.end_at:
            raise ValueError("Batch access expiry cannot be before batch end")
        return self


class UsageEvent(BaseModel):
    module: str = Field(max_length=80)
    event: Literal[
        "logout", "module_opened", "dataset_uploaded", "job_started", "job_completed",
        "job_failed", "model_trained", "prediction_run", "notebook_started",
        "notebook_executed", "query_executed", "genai_request",
    ]
    metadata: dict[str, Any] = Field(default_factory=dict)
