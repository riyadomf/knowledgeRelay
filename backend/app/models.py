import datetime
import uuid
from typing import Optional, List # Added List for relationships

from sqlalchemy.orm import Mapped, mapped_column, relationship # Added relationship
from sqlalchemy import String, Text, DateTime, func, ForeignKey # Import func and ForeignKey

from .database import Base

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    # Removed updated_at for simplicity

    # Relationships
    document_entries: Mapped[List["DocumentKnowledgeEntry"]] = relationship("DocumentKnowledgeEntry", back_populates="project", cascade="all, delete-orphan")
    text_entries: Mapped[List["TextKnowledgeEntry"]] = relationship("TextKnowledgeEntry", back_populates="project", cascade="all, delete-orphan")
    project_qa_sessions: Mapped[List["ProjectQASession"]] = relationship("ProjectQASession", back_populates="project", cascade="all, delete-orphan")
    document_qa_sessions: Mapped[List["DocumentQASession"]] = relationship("DocumentQASession", back_populates="project", cascade="all, delete-orphan")


    def __repr__(self) -> str:
        return f"<Project(id='{self.id}', name='{self.name}')>"

class DocumentKnowledgeEntry(Base):
    __tablename__ = "document_knowledge_entries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), index=True)
    file_name: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String) # Path to where the file is stored
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    # Removed updated_at for simplicity

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="document_entries")
    # text_chunks: Mapped[List["TextKnowledgeEntry"]] = relationship("TextKnowledgeEntry", back_populates="document_entry") # Removed this relationship here, as chunks are not stored as TKE
    document_qa_sessions: Mapped[List["DocumentQASession"]] = relationship("DocumentQASession", back_populates="document_entry", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<DocumentKnowledgeEntry(id='{self.id}', file_name='{self.file_name}', project_id='{self.project_id}')>"


class TextKnowledgeEntry(Base):
    __tablename__ = "text_knowledge_entries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), index=True)
    document_knowledge_entry_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("document_knowledge_entries.id"), nullable=True, index=True)
    question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # Original text context from which QA was derived or chunked
    is_interactive_qa: Mapped[bool] = mapped_column(default=False) # True if this question was generated for an old member to answer

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    # Removed updated_at for simplicity

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="text_entries")
    document_entry: Mapped[Optional["DocumentKnowledgeEntry"]] = relationship("DocumentKnowledgeEntry") # No back_populates as TKE is not a list in DocKE


    def __repr__(self) -> str:
        return f"<TextKnowledgeEntry(id='{self.id}', project_id='{self.project_id}', question='{self.question[:30]}...')>"


class ProjectQASession(Base):
    __tablename__ = "project_qa_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), index=True)
    status: Mapped[str] = mapped_column(String) # e.g., "active", "completed", "aborted"
    current_question_text_entry_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("text_knowledge_entries.id"), nullable=True) # ID of the TextKnowledgeEntry being answered
    current_question_index: Mapped[int] = mapped_column(default=0) # Index in the list of predefined questions to answer
    qa_history: Mapped[str] = mapped_column(Text) # JSON string of Q&A pairs for the session

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    # Removed updated_at for simplicity

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="project_qa_sessions")
    current_question_entry: Mapped[Optional["TextKnowledgeEntry"]] = relationship("TextKnowledgeEntry", foreign_keys=[current_question_text_entry_id])


    def __repr__(self) -> str:
        return f"<ProjectQASession(id='{self.id}', project_id='{self.project_id}', status='{self.status}')>"

class DocumentQASession(Base):
    __tablename__ = "document_qa_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), index=True)
    document_id: Mapped[str] = mapped_column(String, ForeignKey("document_knowledge_entries.id"), index=True) # ID of the document being queried
    status: Mapped[str] = mapped_column(String) # e.g., "active", "completed", "aborted"
    current_question_text_entry_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("text_knowledge_entries.id"), nullable=True) # ID of the TextKnowledgeEntry being answered

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    # Removed updated_at for simplicity

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="document_qa_sessions")
    document_entry: Mapped["DocumentKnowledgeEntry"] = relationship("DocumentKnowledgeEntry", back_populates="document_qa_sessions")
    current_question_entry: Mapped[Optional["TextKnowledgeEntry"]] = relationship("TextKnowledgeEntry", foreign_keys=[current_question_text_entry_id])


    def __repr__(self) -> str:
        return f"<DocumentQASession(id='{self.id}', document_id='{self.document_id}', status='{self.status}')>"
