from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List
from datetime import datetime

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500, description="Question in Bengali or English only")
    law_id: Optional[str] = Field(default=None, description="Search specific law. None = search all laws")
    top_k: int = Field(default=5, ge=1, le=10)
    session_id: Optional[int] = Field(default=None, description="Optional session ID to continue a chat")

    @field_validator('query')
    @classmethod
    def validate_query_not_arabic(cls, v):
        arabic_chars = sum(1 for c in v if '\u0600' <= c <= '\u06FF')
        if arabic_chars > len(v) * 0.3:
            raise ValueError("Query must be in Bengali or English. আরবিতে প্রশ্ন করবেন না।")
        return v.strip()

class Source(BaseModel):
    law_name: str
    year: str
    link: str
    repealed: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    relevance_label: Literal["high", "medium", "low"]
    context_text: str = Field(default="", description="The actual original legal text from the chunk")

class ChatResponse(BaseModel):
    message_id: Optional[int] = None
    session_id: Optional[int] = None
    answer: str
    sources: list[Source]
    confidence: Literal["high", "medium", "low", "not_found"]
    query_language: Literal["bn", "en"]
    legal_query: str
    laws_searched: list[str]
    total_chunks_retrieved: int

class FeedbackResponse(BaseModel):
    id: int
    lawyer_id: int
    mufti_name: Optional[str] = None
    is_good: Optional[bool]
    feedback_text: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    user_query: str
    ai_response: dict
    created_at: datetime
    feedbacks: List[FeedbackResponse] = []

    class Config:
        from_attributes = True

class ChatSessionListResponse(BaseModel):
    id: int
    user_id: int
    title: Optional[str]
    created_at: datetime
    is_pinned: bool

    class Config:
        from_attributes = True

class PaginatedSessions(BaseModel):
    total: int
    page: int
    size: int
    sessions: List[ChatSessionListResponse]

class ChatSessionDetailResponse(ChatSessionListResponse):
    total_messages: int
    page: int
    size: int
    messages: List[ChatMessageResponse] = []

class FeedbackRequest(BaseModel):
    is_good: Optional[bool] = None
    feedback_text: Optional[str] = None

class PinSessionRequest(BaseModel):
    is_pinned: bool
