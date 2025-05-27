import uuid
from typing import List, Optional, Dict

from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, and_

from app import models, schemas

def get_project(db: Session, project_id: str):
    """Retrieve a project by its ID."""
    return db.query(models.Project).filter(models.Project.id == project_id).first()

def get_text_knowledge_entry_by_id(db: Session, entry_id: str):
    """Retrieve a TextKnowledgeEntry by its ID."""
    return db.query(models.TextKnowledgeEntry).filter(models.TextKnowledgeEntry.id == entry_id).first()

def create_text_knowledge_entry(
    db: Session,
    project_id: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    document_knowledge_entry_id: Optional[str] = None,
    source_context: Optional[str] = None,
    is_interactive_qa: bool = False 
):
    """
    Create a new TextKnowledgeEntry.
    Can represent a raw text chunk, a Q&A pair, or an interactive Q&A question.
    """
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

def update_text_knowledge_entry_answer(db: Session, entry_id: str, answer: str):
    """Update the answer for an existing TextKnowledgeEntry."""
    db_entry = get_text_knowledge_entry_by_id(db, entry_id)
    if db_entry:
        db_entry.answer = answer
        db.commit()
        db.refresh(db_entry)
    return db_entry

def get_document_knowledge_entry(db: Session, document_id: str):
    """Retrieve a DocumentKnowledgeEntry by its ID."""
    return db.query(models.DocumentKnowledgeEntry).filter(models.DocumentKnowledgeEntry.id == document_id).first()

def create_document_knowledge_entry(
    db: Session,
    project_id: str,
    file_name: str,
    file_path: str 
):
    """Create a new DocumentKnowledgeEntry for an uploaded document."""
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


# Project Q&A Session CRUD operations
def get_project_qa_session(db: Session, session_id: str) -> Optional[models.ProjectQASession]:
    """Retrieve a project QA session by its ID."""
    return db.query(models.ProjectQASession).filter(models.ProjectQASession.id == session_id).first()

def create_project_qa_session(
    db: Session,
    project_id: str,
    current_question_text_entry_id: Optional[str] = None,
    current_question_index: int = 0
) -> models.ProjectQASession:
    """Create a new project QA session."""
    db_session = models.ProjectQASession(
        id=str(uuid.uuid4()),
        project_id=project_id,
        status="active",
        current_question_text_entry_id=current_question_text_entry_id,
        current_question_index=current_question_index,
        qa_history="[]"
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def update_project_qa_session(
    db: Session,
    session: models.ProjectQASession,
    status: Optional[str] = None,
    current_question_text_entry_id: Optional[str] = None,
    current_question_index: Optional[int] = None,
    qa_history: Optional[List[Dict[str, str]]] = None
) -> models.ProjectQASession:
    """Update an existing project QA session."""
    if status is not None:
        session.status = status
    if current_question_text_entry_id is not None:
        session.current_question_text_entry_id = current_question_text_entry_id
    if current_question_index is not None:
        session.current_question_index = current_question_index
    if qa_history is not None:
        import json
        session.qa_history = json.dumps(qa_history)

    db.commit()
    db.refresh(session)
    return session

def get_unanswered_project_questions(db: Session, project_id: str) -> List[models.TextKnowledgeEntry]:
    """
    Get all unanswered questions related to a project (not tied to a specific document)
    for interactive Q&A, ordered by creation time (oldest first).
    """
    return db.query(models.TextKnowledgeEntry).filter(
        models.TextKnowledgeEntry.project_id == project_id,
        models.TextKnowledgeEntry.document_knowledge_entry_id.is_(None), 
        models.TextKnowledgeEntry.question.isnot(None),
        models.TextKnowledgeEntry.answer.is_(None),
        models.TextKnowledgeEntry.is_interactive_qa == True
    ).order_by(models.TextKnowledgeEntry.created_at).all()

# Document Q&A Session CRUD operations removed

def get_unanswered_questions_for_document(db: Session, project_id: str, document_id: str) -> List[models.TextKnowledgeEntry]:
    """
    Get all unanswered questions for a specific document, generated for interactive QA,
    ordered by creation time (oldest first).
    """
    return db.query(models.TextKnowledgeEntry).filter(
        models.TextKnowledgeEntry.project_id == project_id,
        models.TextKnowledgeEntry.document_knowledge_entry_id == document_id,
        models.TextKnowledgeEntry.question.isnot(None),
        models.TextKnowledgeEntry.answer.is_(None),
        models.TextKnowledgeEntry.is_interactive_qa == True
    ).order_by(models.TextKnowledgeEntry.created_at).all()


def get_recent_text_knowledge_entries(db: Session, project_id: str, limit: int = 5):
    """
    Retrieves recent text knowledge entries for a project to provide context to the LLM.
    """
    return db.query(models.TextKnowledgeEntry).filter(
        models.TextKnowledgeEntry.project_id == project_id
    ).order_by(models.TextKnowledgeEntry.created_at.desc()).limit(limit).all()