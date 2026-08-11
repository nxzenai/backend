from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .constants import (
    LeadPriority,
    LeadStatus,
)


class LeadModel(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)

    full_name: str
    email: str
    phone: Optional[str] = None

    profession: Optional[str] = None

    course: Optional[str] = None

    message: Optional[str] = None

    status: LeadStatus = LeadStatus.NEW

    priority: LeadPriority = LeadPriority.MEDIUM

    assigned_to: Optional[str] = None

    source: str = "Website"

    converted: bool = False

    student_id: Optional[str] = None

    notes: list = []

    activities: list = []

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    updated_at: datetime = Field(
        default_factory=datetime.utcnow
    )