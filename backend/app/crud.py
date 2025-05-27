from sqlalchemy.orm import Session
from app import models, schemas
from typing import List, Optional, Dict
import uuid
import json
from sqlalchemy import Integer, and_

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

def get_text_knowledge_entry_by_id(db: Session, entry_id: str) -> Optional[models.TextKnowledgeEntry]:
    return db.query(models.TextKnowledgeEntry).filter(models.TextKnowledgeEntry.id == entry_id).first()

def update_text_knowledge_entry_answer(db: Session, entry_id: str, answer: str) -> Optional[models.TextKnowledgeEntry]:
    db_entry = db.query(models.TextKnowledgeEntry).filter(models.TextKnowledgeEntry.id == entry_id).first()
    if db_entry:
        db_entry.answer = answer
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
    return db_entry

def get_unanswered_questions_for_document(db: Session, project_id: str, document_id: str) -> List[models.TextKnowledgeEntry]:
    """Retrieves TextKnowledgeEntry records that have a question but no answer, linked to a specific document."""
    return db.query(models.TextKnowledgeEntry).filter(
        and_(
            models.TextKnowledgeEntry.project_id == project_id,
            models.TextKnowledgeEntry.document_knowledge_entry_id == document_id,
            models.TextKnowledgeEntry.question.isnot(None),
            models.TextKnowledgeEntry.answer.is_(None)
        )
    ).order_by(models.TextKnowledgeEntry.created_at).all()

def get_unanswered_project_questions(db: Session, project_id: str) -> List[models.TextKnowledgeEntry]:
    """Retrieves TextKnowledgeEntry records that are general project questions (no document_id) and are unanswered."""
    return db.query(models.TextKnowledgeEntry).filter(
        and_(
            models.TextKnowledgeEntry.project_id == project_id,
            models.TextKnowledgeEntry.document_knowledge_entry_id.is_(None), # Not linked to a document
            models.TextKnowledgeEntry.question.isnot(None),
            models.TextKnowledgeEntry.answer.is_(None),
            models.TextKnowledgeEntry.is_interactive_qa.is_(True) # Explicitly flagged as interactive Q&A
        )
    ).order_by(models.TextKnowledgeEntry.created_at).all()


def create_document_knowledge_entry(
    db: Session,
    project_id: str,
    file_name: str,
    file_path: Optional[str] = None 
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

# CRUD for ProjectQASession
def create_project_qa_session(db: Session, project_id: str, current_question_text_entry_id: Optional[str] = None) -> models.ProjectQASession:
    db_session = models.ProjectQASession(
        id=str(uuid.uuid4()),
        project_id=project_id,
        current_question_index=0, # This index is for predefined questions
        current_question_text_entry_id=current_question_text_entry_id, # Can be null if starting with predefined questions
        status="active",
        qa_history=json.dumps([])
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_project_qa_session(db: Session, session_id: str) -> Optional[models.ProjectQASession]:
    return db.query(models.ProjectQASession).filter(models.ProjectQASession.id == session_id).first()

def update_project_qa_session(
    db: Session,
    session: models.ProjectQASession,
    current_question_index: Optional[int] = None,
    current_question_text_entry_id: Optional[str] = None,
    status: Optional[str] = None,
    qa_history: Optional[List[Dict]] = None
) -> models.ProjectQASession:
    if current_question_index is not None:
        session.current_question_index = current_question_index
    if current_question_text_entry_id is not None:
        session.current_question_text_entry_id = current_question_text_entry_id
    if status is not None:
        session.status = status
    if qa_history is not None:
        session.qa_history = json.dumps(qa_history)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

# CRUD for DocumentQASession
def create_document_qa_session(db: Session, project_id: str, document_id: str, current_question_text_entry_id: Optional[str] = None) -> models.DocumentQASession:
    db_session = models.DocumentQASession(
        id=str(uuid.uuid4()),
        project_id=project_id,
        document_id=document_id,
        current_question_text_entry_id=current_question_text_entry_id,
        status="active" if current_question_text_entry_id else "no_questions",
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_document_qa_session(db: Session, session_id: str) -> Optional[models.DocumentQASession]:
    return db.query(models.DocumentQASession).filter(models.DocumentQASession.id == session_id).first()

def update_document_qa_session(
    db: Session,
    session: models.DocumentQASession,
    current_question_text_entry_id: Optional[str] = None,
    status: Optional[str] = None
) -> models.DocumentQASession:
    if current_question_text_entry_id is not None:
        session.current_question_text_entry_id = current_question_text_entry_id
    if status is not None:
        session.status = status
    db.add(session)
    db.commit()
    db.refresh(session)
    return session