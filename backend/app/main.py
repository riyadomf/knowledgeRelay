import logging
import uuid 
from typing import List, Dict, Union
import json
from fastapi import Form


from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import engine, get_db
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Knowledge Relay API",
    description="API for ingesting knowledge, managing projects, and enabling Q&A sessions.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection for services
def get_ingestion_service(db: Session = Depends(get_db)):
    return IngestionService(db)

def get_retrieval_service(db: Session = Depends(get_db)):
    return RetrievalService(db)

@app.get("/")
def read_root():
    return {"message": "Welcome to Knowledge Relay API"}

# Project Management Endpoints
@app.post("/projects/", response_model=schemas.ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    """
    Create a new project. The 'description' field is optional.
    """
    logger.info(f"Attempting to create project: {project.name}")
    try:
        db_project = models.Project(
            id=str(uuid.uuid4()), 
            name=project.name,
            description=project.description
        )
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        logger.info(f"Project '{project.name}' created with ID: {db_project.id}")
        return db_project
    except Exception as e:
        logger.error(f"Error creating project {project.name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create project: {e}")

@app.get("/projects/", response_model=List[schemas.ProjectResponse])
def get_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a list of all projects.
    """
    projects = db.query(models.Project).offset(skip).limit(limit).all()
    return projects

@app.get("/projects/{project_id}", response_model=schemas.ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a single project by its ID.
    """
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if project is None:
        logger.warning(f"Project with ID {project_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


# Knowledge Ingestion Endpoints

@app.post("/transfer/static-qa/", response_model=schemas.StaticQAIngestResponse, status_code=status.HTTP_200_OK)
def ingest_static_qa_data(
    request: schemas.StaticQAIngestRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)):
    """
    Ingest static Q&A pairs (e.g., from a CSV or predefined list).
    These are immediately available for new member retrieval.
    """
    try:
        response = ingestion_service.ingest_static_qa(
            project_id=request.project_id,
            questions_answers=request.questions_answers,
            document_knowledge_entry_id=request.document_id 
        )
        return response
    except ValueError as e:
        logger.error(f"Error ingesting static Q&A for project {request.project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error ingesting static Q&A for project {request.project_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest static Q&A: {e}")

@app.post("/transfer/document/", response_model=schemas.FileUploadResponse, status_code=status.HTTP_200_OK)
async def ingest_document(
    project_id: str,
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service)):
    """
    Upload a document (PDF, Word, TXT, MD). The document is chunked, and its chunks'
    embeddings are stored in the vector DB. A DocumentKnowledgeEntry is created.
    This step prepares the document for question generation, but doesn't create
    Q&A pairs yet.
    """
    allowed_file_types = ["pdf", "word", "txt", "md"]
    file_extension = file.filename.split(".")[-1].lower()

    if file_extension not in allowed_file_types:
        logger.warning(f"Unsupported file type uploaded for project {project_id}: {file_extension}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_file_types)}"
        )

    try:
        content = await file.read()
        response = ingestion_service.ingest_document(project_id, content, file.filename, file_extension)
        return response
    except ValueError as e:
        logger.error(f"Error ingesting document for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error ingesting document for project {project_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest document: {e}")

@app.post("/transfer/document-qa/generate-questions/", response_model=schemas.DocumentQAGenerateQuestionsResponse, status_code=status.HTTP_200_OK)
def generate_questions_from_document(
    request: schemas.DocumentQAGenerateQuestionsRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)):
    """
    Generate questions from an uploaded document's content using an LLM.
    These questions are stored as TextKnowledgeEntry records with null answers,
    awaiting an old member to provide answers.
    """
    try:
        response = ingestion_service.generate_questions_from_document(request.project_id, request.document_id)
        return response
    except ValueError as e:
        logger.error(f"Error generating questions from document {request.document_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error generating questions from document {request.document_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate questions from document: {e}")


# Interactive Q&A Session Endpoints (Old Member)

@app.post("/transfer/project-qa/start-session/", response_model=schemas.ProjectQASessionStartResponse, status_code=status.HTTP_200_OK)
def start_project_qa_session(
    project_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service)):
    """
    Initiate an interactive Q&A session for a project.
    This begins the process of asking an old member questions about the project.
    """
    try:
        response = ingestion_service.start_project_qa_session(project_id)
        return response
    except ValueError as e:
        logger.error(f"Error starting project Q&A session for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error starting project Q&A session for project {project_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to start project Q&A session: {e}")

@app.post("/transfer/project-qa/respond/", response_model=schemas.ProjectQAResponse, status_code=status.HTTP_200_OK)
def respond_to_project_qa(
    session_id: str = Form(...),
    project_id: str = Form(...),
    answer: str = Form(...),
    # request: schemas.ProjectQARespondRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)):
    """
    Submit an answer to the current question in a project-wide Q&A session.
    The answer is stored, and the next question (if any) is returned.
    """
    try:
        response = ingestion_service.respond_to_project_qa(session_id, project_id, answer)
        return response
    except ValueError as e:
        logger.error(f"Error responding to project Q&A session {request.session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException as e: 
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error responding to project Q&A session {request.session_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to respond to project Q&A: {e}")

# New endpoints for document-specific Q&A without sessions
@app.post("/transfer/document-qa/get-next-question/", response_model=schemas.GetNextDocumentQuestionResponse, status_code=status.HTTP_200_OK)
def get_next_document_question(
    request: schemas.GetNextDocumentQuestionRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)):
    """
    Retrieves the oldest unanswered question for a specific document.
    """
    try:
        response = ingestion_service.get_next_document_question(request.project_id, request.document_id)
        return response
    except ValueError as e:
        logger.error(f"Error getting next document question for project {request.project_id}, document {request.document_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error getting next document question for project {request.project_id}, document {request.document_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get next document question: {e}")

@app.post("/transfer/document-qa/answer-question/", response_model=schemas.AnswerDocumentQuestionResponse, status_code=status.HTTP_200_OK)
def answer_document_question(
    request: schemas.AnswerDocumentQuestionRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)):
    """
    Submits an answer to a specific document question (identified by question_entry_id).
    The answer is stored, and the Q&A pair is ingested into the knowledge base.
    """
    try:
        response = ingestion_service.answer_document_question(request.project_id, request.question_entry_id, request.answer)
        return response
    except ValueError as e:
        logger.error(f"Error answering document question {request.question_entry_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException as e: 
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error answering document question {request.question_entry_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to answer document question: {e}")


# Knowledge Retrieval Endpoints (New Member)

@app.post("/retrieve/answer/", response_model=schemas.ChatResponse, status_code=status.HTTP_200_OK) 
def retrieve_answer(
    request: schemas.RetrievalRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    """
    Retrieve an answer for a given query from the knowledge base of a specific project.
    Can include chat history for conversational context.
    """
    try:
        response = retrieval_service.answer_query(request.project_id, request.query, request.chat_history)
        return response
    except ValueError as e:
        logger.error(f"Retrieval error for project {request.project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected retrieval error for project {request.project_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Retrieval failed: {e}")

@app.post("/retrieve/relevant-chunks/", response_model=schemas.RelevantChunksResponse, status_code=status.HTTP_200_OK)
def retrieve_relevant_chunks(
    request: schemas.RetrievalRequest, 
    retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    """
    Retrieve raw relevant chunks of text from the knowledge base for a given query.
    Useful for debugging or showing underlying data.
    """
    try:
        chunks = retrieval_service.retrieve_relevant_chunks(request.project_id, request.query)
        return schemas.RelevantChunksResponse(chunks=chunks)
    except ValueError as e:
        logger.error(f"Relevant chunks retrieval error for project {request.project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected relevant chunks retrieval error for project {request.project_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Relevant chunks retrieval failed: {e}")

