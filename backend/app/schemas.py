from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

# --- Project Schemas ---
class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    pass 

class Project(ProjectBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Knowledge Transfer (Old Member) Schemas ---

# For bulk Q&A ingestion (can be general or document-related)
class StaticQARequest(BaseModel):
    project_id: str
    questions_answers: List[Dict[str, str]] 
    document_id: Optional[str] = None 

class FileUploadResponse(BaseModel):
    project_id: str
    file_name: str
    message: str
    document_id: str

# For Project-wide Interactive Q&A (pre-defined or dynamically generated questions)
class ProjectQASessionStartResponse(BaseModel): 
    session_id: str
    project_id: str
    question: Optional[str]
    question_entry_id: Optional[str] # New: ID of the TextKnowledgeEntry for the question
    is_complete: bool
    message: str = "Session started."

class ProjectQAResponseRequest(BaseModel): 
    session_id: str
    project_id: str
    answer: str

class ProjectQAResponse(BaseModel): 
    session_id: str
    project_id: str
    next_question: Optional[str]
    next_question_entry_id: Optional[str] # New: ID of the TextKnowledgeEntry for the next question
    is_complete: bool
    message: str

# For Document-specific Interactive Q&A
class DocumentQAGenerateQuestionsRequest(BaseModel):
    project_id: str
    document_id: str

class DocumentQAGenerateQuestionsResponse(BaseModel):
    project_id: str
    document_id: str
    question_entry_ids: List[str] 
    message: str

class DocumentQASessionStartResponse(BaseModel): 
    session_id: str
    project_id: str
    document_id: str
    question: Optional[str] 
    question_entry_id: Optional[str] 
    is_complete: bool
    message: str

class DocumentQAResponseRequest(BaseModel): 
    session_id: str
    project_id: str
    answer: str

class DocumentQAResponse(BaseModel): 
    session_id: str
    project_id: str
    next_question: Optional[str]
    next_question_entry_id: Optional[str]
    is_complete: bool
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
    question: Optional[str] = None 
    context: str
    document_id: Optional[str] = None
    page_number: Optional[int] = None

class ChatResponse(BaseModel):
    project_id: str
    answer: str
    source_documents: List[SourceDocument]