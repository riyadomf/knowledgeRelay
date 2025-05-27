from sqlalchemy.orm import Session
from app import models, schemas
from typing import List, Optional, Dict
import uuid
import json
from sqlalchemy import Integer


def create_project(db: Session, project: schemas.ProjectCreate) -> models.Project:
    # Generate UUID for the new project ID
    db_project = models.Project(id=str(uuid.uuid4()), name=project.name)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def get_project(db: Session, project_id: str) -> Optional[models.Project]:
    return db.query(models.Project).filter(models.Project.id == project_id).first()

def get_projects(db: Session, skip: int = 0, limit: int = 100) -> List[models.Project]:
    return db.query(models.Project).offset(skip).limit(limit).all()

def create_text_knowledge_entry(
    db: Session,
    project_id: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    document_knowledge_entry_id: Optional[str] = None,
    source_context: Optional[str] = None,
    is_interactive_qa: bool = False
) -> models.TextKnowledgeEntry:
    db_entry = models.TextKnowledgeEntry(
        id=str(uuid.uuid4()),
        project_id=project_id,
        question=question,
        answer=answer,
        document_knowledge_entry_id=document_knowledge_entry_id,
        source_context=source_context,
        is_interactive_qa=is_interactive_qa
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

def create_document_knowledge_entry(
    db: Session,
    project_id: str,
    file_name: str,
    file_path: Optional[str] = None # Make optional as ChromaDB handles storage
) -> models.DocumentKnowledgeEntry:
    db_entry = models.DocumentKnowledgeEntry(
        id=str(uuid.uuid4()),
        project_id=project_id,
        file_name=file_name,
        file_path=file_path
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

def get_document_knowledge_entry(db: Session, document_id: str) -> Optional[models.DocumentKnowledgeEntry]:
    return db.query(models.DocumentKnowledgeEntry).filter(models.DocumentKnowledgeEntry.id == document_id).first()

def get_text_knowledge_entries_by_document_id(db: Session, document_id: str) -> List[models.TextKnowledgeEntry]:
    return db.query(models.TextKnowledgeEntry).filter(models.TextKnowledgeEntry.document_knowledge_entry_id == document_id).all()

def create_interactive_qa_session(db: Session, project_id: str) -> models.InteractiveQASession:
    db_session = models.InteractiveQASession(
        id=str(uuid.uuid4()),
        project_id=project_id,
        current_question_index=0,
        status="active",
        qa_history=json.dumps([])
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_interactive_qa_session(db: Session, session_id: str) -> Optional[models.InteractiveQASession]:
    return db.query(models.InteractiveQASession).filter(models.InteractiveQASession.id == session_id).first()

def update_interactive_qa_session(
    db: Session,
    session: models.InteractiveQASession,
    current_question_index: Optional[int] = None,
    status: Optional[str] = None,
    qa_history: Optional[List[Dict]] = None
) -> models.InteractiveQASession:
    if current_question_index is not None:
        session.current_question_index = current_question_index
    if status is not None:
        session.status = status
    if qa_history is not None:
        session.qa_history = json.dumps(qa_history)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session
