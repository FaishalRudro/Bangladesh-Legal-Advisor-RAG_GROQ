from pydantic import BaseModel
from typing import List, Optional
from app.models.user import UserRole
from app.schemas.chat_schema import FeedbackResponse
from app.schemas.user import UserResponse

class UserStatusUpdate(BaseModel):
    is_active: bool

class UserRoleUpdate(BaseModel):
    role: UserRole

class PaginatedFeedbacks(BaseModel):
    total: int
    page: int
    size: int
    feedbacks: List[FeedbackResponse]

class PaginatedUsers(BaseModel):
    total: int
    page: int
    size: int
    users: List[UserResponse]
