from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

# --- Project Schemas ---
class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    id: str # Allow client to provide ID for simplicity in hackathon

class Project(ProjectBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Knowledge Transfer (Old Member) Schemas ---
class StaticQARequest(BaseModel):
    project_id: str
    questions_answers: List[Dict[str, str]] # [{"question": "...", "answer": "..."}]

class FileUploadResponse(BaseModel):
    project_id: str
    file_name: str
    message: str
    document_id: str

class InteractiveQASessionStartResponse(BaseModel):
    session_id: str
    project_id: str
    question: str
    is_complete: bool

class InteractiveQAResponseRequest(BaseModel):
    session_id: str
    project_id: str
    answer: str

class InteractiveQAResponse(BaseModel):
    session_id: str
    project_id: str
    next_question: Optional[str]
    is_complete: bool
    message: str

class DocumentQAGenerateQuestionsRequest(BaseModel):
    project_id: str
    document_id: str

class DocumentQAGenerateQuestionsResponse(BaseModel):
    project_id: str
    document_id: str
    suggested_questions: List[str]
    message: str

# --- Knowledge Retrieval (New Member) Schemas ---
class ChatRole(str, Enum):
    HUMAN = "human"
    AI = "ai"

class ChatMessage(BaseModel):
    role: ChatRole
    content: str

class ChatRequest(BaseModel):
    project_id: str
    query: str
    chat_history: Optional[List[ChatMessage]] = []

class SourceDocument(BaseModel):
    file_name: str
    question: Optional[str] = None # For static Q&A
    context: str
    document_id: Optional[str] = None
    page_number: Optional[int] = None

class ChatResponse(BaseModel):
    project_id: str
    answer: str
    source_documents: List[SourceDocument]
