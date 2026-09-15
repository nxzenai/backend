from datetime import date
import re
from typing import Optional, Literal

from pydantic import BaseModel, EmailStr, model_validator, Field


class LeadCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str

    profession: str
    program_interest: str

    # NEW FIELD
    preferred_demo_date: str = ""

    source: str = Field(default="website", max_length=100)
    city: str = Field(default="", max_length=200)
    qualification: str = Field(default="", max_length=300)
    organization: str = Field(default="", max_length=300)
    experience: str = ""
    referral_source: str = Field(default="", max_length=300)
    consent: bool = False

    @model_validator(mode="after")
    def validate_training_registration(self):
        # Keep legacy demo/contact validation and payloads compatible.
        if self.source == "training_registration":
            for field in ("name", "phone", "city", "profession", "program_interest", "qualification"):
                value = getattr(self, field).strip()
                if not value or len(value) > 300:
                    raise ValueError(f"{field} is required and must be at most 300 characters")
                setattr(self, field, value)
            if self.profession not in {"Student", "Working Professional", "Entrepreneur", "Career Switcher", "Other"}:
                raise ValueError("Select a valid profession")
            if self.experience not in {"", "Fresher", "0–2 years", "2–5 years", "5+ years"}:
                raise ValueError("Select a valid experience level")
            if not re.fullmatch(r"\+?[0-9 ()-]+", self.phone) or not 10 <= len(re.sub(r"\D", "", self.phone)) <= 15:
                raise ValueError("Enter a valid mobile number with 10 to 15 digits")
            if not self.consent:
                raise ValueError("Consent to be contacted is required")
            if self.preferred_demo_date:
                date.fromisoformat(self.preferred_demo_date)
        return self

    message: Optional[str] = None

    notes: Optional[str] = ""
    priority: Optional[str] = "warm"
    follow_up_date: Optional[str] = ""


class VerificationUpdate(BaseModel):
    verification_status: Literal["pending", "verified", "not_verified"]
    verification_notes: str = Field(default="", max_length=4000)
