from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from app.schemas.user import validate_strong_password

class UserUpdate(BaseModel):
    name: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    old_password: str

class ChangePasswordVerify(BaseModel):
    otp: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v):
        return validate_strong_password(v)

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v):
        return validate_strong_password(v)
