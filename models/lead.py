from pydantic import BaseModel, EmailStr
from typing import Optional


class LeadCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str
    profession: str
    program_interest: str
    message: Optional[str] = None

    notes: Optional[str] = ""
    priority: Optional[str] = "warm"
    follow_up_date: Optional[str] = ""