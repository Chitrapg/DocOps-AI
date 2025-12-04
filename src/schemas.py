# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any

class UploadResponse(BaseModel):
    doc_id: str
    chunk_count: int

class GenerateRequest(BaseModel):
    doc_id: Optional[str] = None
    feature_text: str
    k: int = Field(5, ge=1, le=20)

class TestCase(BaseModel):
    id: str
    title: str
    category: str
    priority: str
    preconditions: List[str] = []
    steps: List[str]
    expected_result: str
    test_data: Optional[Any] = None
    related_requirement_ids: List[str] = []
    tags: List[str] = []

class GenerateResponse(BaseModel):
    tests: List[TestCase]
    context_snippets: List[str] = []
