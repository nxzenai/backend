from typing import Optional

from pydantic import BaseModel

from .constants import (
    LeadPriority,
    LeadStatus,
)


class LeadListQuery(BaseModel):
    page: int = 1
    limit: int = 20

    search: Optional[str] = None

    status: Optional[LeadStatus] = None

    priority: Optional[LeadPriority] = None


class UpdateLeadRequest(BaseModel):
    status: Optional[LeadStatus] = None

    priority: Optional[LeadPriority] = None

    assigned_to: Optional[str] = None


class LeadNoteRequest(BaseModel):
    note: str