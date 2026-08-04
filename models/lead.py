from typing import Optional

from pydantic import BaseModel, EmailStr


class LeadCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str

    profession: str
    program_interest: str

    # NEW FIELD
    preferred_demo_date: str

    message: Optional[str] = None

    notes: Optional[str] = ""
    priority: Optional[str] = "warm"
    follow_up_date: Optional[str] = ""
