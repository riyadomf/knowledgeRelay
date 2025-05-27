from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict
import logging

from app import models, schemas, crud
from app.database import engine, get_db, Base
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.core.vector_store import ChromaDBManager # For project deletion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)
logger.info("Database tables created/checked.")

app = FastAPI(
    title="KnowledgeRelay AI Backend",
    description="AI-based knowledge transfer agent for project onboarding.",
    version="1.0.0"
)

# Dependency to get IngestionService
def get_ingestion_service(db: Session = Depends(get_db)) -> IngestionService:
    return IngestionService(db)

# Dependency to get RetrievalService
def get_retrieval_service(db: Session = Depends(get_db)) -> RetrievalService:
    return RetrievalService(db)


@app.get("/")
async def root():
    return {"message": "Welcome to KnowledgeRelay AI Backend!"}

# --- Project Management Endpoints ---
@app.post("/projects/", response_model=schemas.Project, status_code=status.HTTP_201_CREATED)
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    db_project = crud.get_project(db, project.id)
    if db_project:
        logger.warning(f"Attempted to create project with existing ID: {project.id}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Project with this ID already exists.")
    new_project = crud.create_project(db=db, project=project)
    logger.info(f"Created new project: {new_project.id} - {new_project.name}")
    return new_project

@app.get("/projects/", response_model=List[schemas.Project])
def get_all_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    projects = crud.get_projects(db, skip=skip, limit=limit)
    logger.info(f"Retrieved {len(projects)} projects.")
    return projects

@app.get("/projects/{project_id}", response_model=schemas.Project)
def get_project_by_id(project_id: str, db: Session = Depends(get_db)):
    project = crud.get_project(db, project_id)
    if project is None:
        logger.warning(f"Project with ID {project_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    logger.info(f"Retrieved project: {project_id}")
    return project

@app.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    project = crud.get_project(db, project_id)
    if not project:
        logger.warning(f"Attempted to delete non-existent project: {project_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    try:
        # Deleting project cascades to related entries due to 'cascade="all, delete-orphan"' in models
        db.delete(project)
        db.commit()

        # Delete the corresponding ChromaDB collection
        chroma_manager = ChromaDBManager(project_id)
        chroma_manager.delete_collection()
        logger.info(f"Project {project_id} and its knowledge base deleted successfully.")
        return {"message": "Project and its knowledge base deleted successfully."}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete project: {e}")


# --- Old Member: Knowledge Transfer Endpoints ---
@app.post("/transfer/static-qa/", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
def ingest_static_qa(
    request: schemas.StaticQARequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    try:
        response = ingestion_service.ingest_static_qa(request.project_id, request.questions_answers)
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
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
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

@app.post("/transfer/interactive-qa/start/", response_model=schemas.InteractiveQASessionStartResponse, status_code=status.HTTP_200_OK)
def start_interactive_qa_session(
    project_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    try:
        response = ingestion_service.start_interactive_qa_session(project_id)
        return response
    except ValueError as e:
        logger.error(f"Error starting interactive Q&A session for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error starting interactive Q&A session for project {project_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to start interactive Q&A session: {e}")

@app.post("/transfer/interactive-qa/respond/", response_model=schemas.InteractiveQAResponse, status_code=status.HTTP_200_OK)
def respond_to_interactive_qa(
    request: schemas.InteractiveQAResponseRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    try:
        response = ingestion_service.respond_to_interactive_qa(
            request.session_id, request.project_id, request.answer
        )
        return response
    except ValueError as e:
        logger.error(f"Error responding to interactive Q&A session {request.session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error responding to interactive Q&A session {request.session_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to respond to interactive Q&A: {e}")

@app.post("/transfer/document-qa/generate-questions/", response_model=schemas.DocumentQAGenerateQuestionsResponse, status_code=status.HTTP_200_OK)
def generate_questions_from_document(
    request: schemas.DocumentQAGenerateQuestionsRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    try:
        response = ingestion_service.generate_questions_from_document(request.project_id, request.document_id)
        return response
    except ValueError as e:
        logger.error(f"Error generating questions from document {request.document_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error generating questions from document {request.document_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate questions from document: {e}")


# --- New Member: Knowledge Retrieval Endpoints ---
@app.post("/query/", response_model=schemas.ChatResponse, status_code=status.HTTP_200_OK)
def answer_query(
    request: schemas.ChatRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    try:
        response = retrieval_service.answer_query(
            project_id=request.project_id,
            query=request.query,
            chat_history=request.chat_history
        )
        return response
    except ValueError as e:
        logger.error(f"Error answering query for project {request.project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error answering query for project {request.project_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to answer query: {e}")

