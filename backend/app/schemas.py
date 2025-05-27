from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import datetime
from enum import Enum 

# --- Base Models ---
class Timestamps(BaseModel):
    created_at: datetime.datetime
    # Removed updated_at

# --- Project Schemas ---
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase, Timestamps):
    id: str

    class Config:
        from_attributes = True 

# --- Knowledge Entry Schemas ---

class TextKnowledgeEntryBase(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    source_context: Optional[str] = None
    is_interactive_qa: bool = False

class TextKnowledgeEntryResponse(TextKnowledgeEntryBase, Timestamps):
    id: str
    project_id: str
    document_knowledge_entry_id: Optional[str] = None

    class Config:
        from_attributes = True

class DocumentKnowledgeEntryBase(BaseModel):
    file_name: str
    file_path: str 

class DocumentKnowledgeEntryResponse(DocumentKnowledgeEntryBase, Timestamps):
    id: str
    project_id: str

    class Config:
        from_attributes = True

# --- Ingestion Schemas ---

# For Document Upload & Chunking
class FileUploadResponse(BaseModel):
    project_id: str
    file_name: str
    message: str
    document_id: str 

# For Static Q&A (bulk ingestion)
class StaticQAPair(BaseModel):
    question: Optional[str] = None 
    answer: str

class StaticQAIngestRequest(BaseModel):
    project_id: str
    questions_answers: List[StaticQAPair]
    document_id: Optional[str] = Field(None, description="Optional: Link these Q&A to a specific document ID.")

class StaticQAIngestResponse(BaseModel):
    message: str

# For Document-specific Q&A: Generate Questions
class DocumentQAGenerateQuestionsRequest(BaseModel):
    project_id: str
    document_id: str

class DocumentQAGenerateQuestionsResponse(BaseModel):
    project_id: str
    document_id: str
    question_entry_ids: List[str] 
    message: str

# --- Interactive Q&A Session Schemas (Old Member) ---

# Project-wide Q&A
class ProjectQASessionStartResponse(BaseModel):
    session_id: str
    project_id: str
    question: Optional[str] 
    question_entry_id: Optional[str] = None 
    is_complete: bool
    message: str

class ProjectQARespondRequest(BaseModel):
    session_id: str
    project_id: str
    answer: str

class ProjectQAResponse(BaseModel):
    session_id: str
    project_id: str
    next_question: Optional[str]
    next_question_entry_id: Optional[str] = None 
    is_complete: bool
    message: str

# Document-specific Q&A
class DocumentQASessionStartResponse(BaseModel):
    session_id: str
    project_id: str
    document_id: str
    question: Optional[str]
    question_entry_id: Optional[str] = None 
    is_complete: bool
    message: str

class DocumentQARespondRequest(BaseModel):
    session_id: str
    project_id: str
    answer: str

class DocumentQAResponse(BaseModel):
    session_id: str
    project_id: str
    next_question: Optional[str]
    next_question_entry_id: Optional[str] = None 
    is_complete: bool
    message: str

# --- Retrieval Schemas (New Member) ---

class ChatRole(str, Enum):
    HUMAN = "human"
    AI = "ai"

class ChatMessage(BaseModel):
    role: ChatRole
    content: str

class SourceDocument(BaseModel):
    file_name: str
    question: Optional[str] = None 
    context: str 
    document_id: Optional[str] = None 
    page_number: Optional[int] = None 

class RetrievalRequest(BaseModel):
    project_id: str
    query: str
    chat_history: Optional[List[ChatMessage]] = [] 

class ChatResponse(BaseModel): 
    project_id: str
    answer: str
    source_documents: List[SourceDocument] 

class Chunk(BaseModel):
    content: str
    metadata: Dict[str, Any]

class RelevantChunksResponse(BaseModel):
    chunks: List[Chunk]