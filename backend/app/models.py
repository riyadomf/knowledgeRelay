from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, Integer
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
    project_qa_sessions = relationship("ProjectQASession", back_populates="project", cascade="all, delete-orphan")
    document_qa_sessions = relationship("DocumentQASession", back_populates="project", cascade="all, delete-orphan")


class DocumentKnowledgeEntry(Base):
    __tablename__ = "document_knowledge_entries"
    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=True) 
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="document_entries")
    text_chunks = relationship("TextKnowledgeEntry", back_populates="document_entry", cascade="all, delete-orphan")
    document_qa_sessions = relationship("DocumentQASession", back_populates="document_entry", cascade="all, delete-orphan")

class TextKnowledgeEntry(Base):
    __tablename__ = "text_knowledge_entries"
    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    document_knowledge_entry_id = Column(String, ForeignKey("document_knowledge_entries.id"), nullable=True) 
    question = Column(Text, nullable=True) 
    answer = Column(Text, nullable=True)   
    source_context = Column(Text, nullable=True) 
    is_interactive_qa = Column(Boolean, default=False) 
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="text_entries")
    document_entry = relationship("DocumentKnowledgeEntry", back_populates="text_chunks")


class ProjectQASession(Base): 
    __tablename__ = "project_qa_sessions" 
    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    current_question_index = Column(Integer, default=0) 
    current_question_text_entry_id = Column(String, ForeignKey("text_knowledge_entries.id"), nullable=True) # ID of the TextKnowledgeEntry whose question is being asked
    status = Column(String, default="active") 
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("Project", back_populates="project_qa_sessions")
    qa_history = Column(Text, nullable=True) 
    current_question_entry = relationship("TextKnowledgeEntry", foreign_keys=[current_question_text_entry_id])


class DocumentQASession(Base): 
    __tablename__ = "document_qa_sessions"
    id = Column(String, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    document_id = Column(String, ForeignKey("document_knowledge_entries.id"), nullable=False)
    current_question_text_entry_id = Column(String, ForeignKey("text_knowledge_entries.id"), nullable=True) 
    status = Column(String, default="active") 
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="document_qa_sessions")
    document_entry = relationship("DocumentKnowledgeEntry", back_populates="document_qa_sessions")
    current_question_entry = relationship("TextKnowledgeEntry", foreign_keys=[current_question_text_entry_id])
