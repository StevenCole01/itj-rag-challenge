from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    top_k: Optional[int] = 5


class SourceCitation(BaseModel):
    """Model for a single source attribution."""
    source: str
    page: int
    text: str


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[SourceCitation]
