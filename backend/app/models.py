from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document_entries = relationship("DocumentKnowledgeEntry", back_populates="project", cascade="all, delete-orphan")
    text_entries = relationship("TextKnowledgeEntry", back_populates="project", cascade="all, delete-orphan")
    interactive_qa_sessions = relationship("InteractiveQASession", back_populates="project", cascade="all, delete-orphan")


class DocumentKnowledgeEntry(Base):
    __tablename__ = "document_knowledge_entries"
    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    file_name = Column(String, nullable=False)
    # file_path is more for internal reference, ChromaDB handles its own storage
    file_path = Column(String, nullable=True) # Can be null if file not stored locally
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="document_entries")
    text_chunks = relationship("TextKnowledgeEntry", back_populates="document_entry", cascade="all, delete-orphan")


class TextKnowledgeEntry(Base):
    __tablename__ = "text_knowledge_entries"
    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    document_knowledge_entry_id = Column(String, ForeignKey("document_knowledge_entries.id"), nullable=True) # Optional for Q&A
    question = Column(Text, nullable=True) # Question from Q&A or text chunk
    answer = Column(Text, nullable=True)   # Answer from Q&A or text chunk content
    source_context = Column(Text, nullable=True) # To store context for retrieved answers
    is_interactive_qa = Column(Boolean, default=False) # Flag for interactive Q&A
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="text_entries")
    document_entry = relationship("DocumentKnowledgeEntry", back_populates="text_chunks")

class InteractiveQASession(Base):
    __tablename__ = "interactive_qa_sessions"
    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    current_question_index = Column(Integer, default=0)
    status = Column(String, default="active") # "active", "completed"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("Project", back_populates="interactive_qa_sessions")
    # Store history of questions asked and answers given within this session
    qa_history = Column(Text, nullable=True) # JSON string of [{"q": "...", "a": "..."}]

