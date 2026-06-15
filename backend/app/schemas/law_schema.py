from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class LawIngestResponse(BaseModel):
    law_id: str
    law_name: str
    year: str
    link: str
    repealed: str
    total_chunks: int
    status: Literal["ingested", "already_exists", "failed"]
    message: str
    ingested_at: datetime

class LawInfo(BaseModel):
    law_id: str
    law_name: str
    year: str
    link: str
    repealed: str
    total_chunks: int
    ingested_at: str

class LawListResponse(BaseModel):
    total: int
    page: int
    size: int
    laws: list[LawInfo]

class DeleteResponse(BaseModel):
    law_id: str
    status: str
    message: str
