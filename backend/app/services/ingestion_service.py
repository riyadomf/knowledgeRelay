from sqlalchemy.orm import Session
from app import schemas, crud, models
from app.core.vector_store import ChromaDBManager
from app.core.document_loaders import load_document
from app.core.text_splitters import split_documents
from app.services.llm_service import LLMService
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os
import uuid
import json
import logging

logger = logging.getLogger(__name__)

# Predefined initial questions for interactive Q&A
INITIAL_INTERACTIVE_QA_QUESTIONS = [
    "What is the primary purpose and mission of this project?",
    "What are the key technologies, frameworks, and libraries used in this project?",
    "Describe the overall architecture of the project (e.g., microservices, monolith, database choices).",
    "What is the standard deployment process for this project? (e.g., CI/CD, manual steps, environments)",
    "What are the most common pitfalls, tricky bugs, or unexpected behaviors new developers should be aware of?",
    "Who are the key stakeholders or contact persons for different parts of the project (e.g., frontend, backend, database, infrastructure)?",
    "Are there any specific team conventions, coding standards, or practices unique to this project?",
    "Where can a new developer find important documentation, runbooks, or troubleshooting guides?",
    "What are the major components or modules within the codebase, and what is their responsibility?",
    "Is there anything else critical a new team member should know to get up to speed quickly?"
]

class IngestionService:
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()

    def ingest_static_qa(self, project_id: str, questions_answers: List[Dict[str, str]], is_interactive_qa: bool = False):
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for static Q&A ingestion.")
            raise ValueError(f"Project with ID {project_id} not found.")

        chroma_manager = ChromaDBManager(project_id)
        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for qa_pair in questions_answers:
            question = qa_pair.get("question")
            answer = qa_pair.get("answer")
            if not answer:
                logger.warning(f"Skipping Q&A pair due to missing answer for project {project_id}: {qa_pair}")
                continue

            # Store in relational DB
            crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                question=question,
                answer=answer,
                source_context=answer, # For static Q&A, the answer itself is the context
                is_interactive_qa=is_interactive_qa
            )
            
            # Prepare for vector DB ingestion
            documents_to_add.append(answer) # Embed the answer for retrieval
            metadatas_to_add.append({
                "type": "static_qa",
                "project_id": project_id,
                "question": question,
                "answer": answer,
                "source_context": answer
            })
            ids_to_add.append(f"static_qa_{uuid.uuid4()}")

        if documents_to_add:
            chroma_manager.add_documents(documents_to_add, metadatas_to_add, ids_to_add)
        
        logger.info(f"Ingested {len(questions_answers)} static Q&A pairs for project {project_id}.")
        return {"message": "Static Q&A ingested successfully."}

    def ingest_document(self, project_id: str, file: bytes, file_name: str, file_type: str):
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for document ingestion.")
            raise ValueError(f"Project with ID {project_id} not found.")

        # Save file temporarily for processing (in a real app, use persistent storage)
        temp_dir = "/tmp/knowledge_relay_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file_name}")
        with open(temp_file_path, "wb") as f:
            f.write(file)
        logger.info(f"Temporarily saved uploaded file to: {temp_file_path}")

        # Load and split document
        loaded_documents = load_document(temp_file_path, file_type)
        split_docs = split_documents(loaded_documents)

        # Create a document entry in the relational DB
        document_db_entry = crud.create_document_knowledge_entry(
            db=self.db,
            project_id=project_id,
            file_name=file_name,
            file_path=temp_file_path # Store temp path for now, or a more permanent path
        )
        logger.info(f"Created document entry in DB: {document_db_entry.id}")
        
        # Prepare for ChromaDB ingestion and relational DB chunk storage
        texts_for_chroma = []
        metadatas_for_chroma = []
        ids_for_chroma = []

        for i, doc_chunk in enumerate(split_docs):
            # Store chunk in relational DB
            crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                document_knowledge_entry_id=document_db_entry.id,
                answer=doc_chunk.page_content, # Store chunk content as answer
                source_context=doc_chunk.page_content # Source context is the chunk itself
            )

            # Prepare for ChromaDB
            metadata = {
                "project_id": project_id,
                "document_id": document_db_entry.id,
                "file_name": file_name,
                "type": "document_chunk",
                "source_context": doc_chunk.page_content # Store chunk itself as source context
            }
            if 'page' in doc_chunk.metadata:
                metadata['page_number'] = doc_chunk.metadata['page']
            if 'source' in doc_chunk.metadata: # e.g., for markdown files
                metadata['source'] = doc_chunk.metadata['source']
            
            texts_for_chroma.append(doc_chunk.page_content)
            metadatas_for_chroma.append(metadata)
            ids_for_chroma.append(f"doc_chunk_{document_db_entry.id}_{i}")

        # Ingest into ChromaDB
        chroma_manager = ChromaDBManager(project_id)
        if texts_for_chroma:
            chroma_manager.add_documents(texts_for_chroma, metadatas_for_chroma, ids_for_chroma)
        else:
            logger.warning(f"No text chunks extracted from {file_name} for project {project_id}.")

        # Clean up temporary file
        os.remove(temp_file_path)
        logger.info(f"Cleaned up temporary file: {temp_file_path}")

        return schemas.FileUploadResponse(
            project_id=project_id,
            file_name=file_name,
            message="Document ingested successfully.",
            document_id=document_db_entry.id
        )

    def start_interactive_qa_session(self, project_id: str) -> schemas.InteractiveQASessionStartResponse:
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for starting interactive Q&A.")
            raise ValueError(f"Project with ID {project_id} not found.")

        # Create a new session
        session = crud.create_interactive_qa_session(self.db, project_id)
        
        # Get the first question from the predefined list
        if INITIAL_INTERACTIVE_QA_QUESTIONS:
            first_question = INITIAL_INTERACTIVE_QA_QUESTIONS[0]
            crud.update_interactive_qa_session(self.db, session, current_question_index=0)
            logger.info(f"Started interactive Q&A session {session.id} for project {project_id}.")
            return schemas.InteractiveQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                question=first_question,
                is_complete=False
            )
        else:
            logger.warning("No initial questions defined for interactive Q&A.")
            return schemas.InteractiveQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                question="No questions available. Session complete.",
                is_complete=True
            )

    def respond_to_interactive_qa(self, session_id: str, project_id: str, answer: str) -> schemas.InteractiveQAResponse:
        session = crud.get_interactive_qa_session(self.db, session_id)
        if not session or session.project_id != project_id:
            logger.error(f"Interactive Q&A session {session_id} not found or does not belong to project {project_id}.")
            raise ValueError("Interactive Q&A session not found or invalid for this project.")
        
        if session.status == "completed":
            logger.info(f"Interactive Q&A session {session_id} is already completed.")
            return schemas.InteractiveQAResponse(
                session_id=session.id,
                project_id=project_id,
                next_question=None,
                is_complete=True,
                message="Session already completed."
            )

        qa_history_list = json.loads(session.qa_history)
        current_question_index = session.current_question_index

        if current_question_index < len(INITIAL_INTERACTIVE_QA_QUESTIONS):
            current_question = INITIAL_INTERACTIVE_QA_QUESTIONS[current_question_index]
            
            # Store the Q&A pair as a TextKnowledgeEntry
            self.ingest_static_qa(
                project_id=project_id,
                questions_answers=[{"question": current_question, "answer": answer}],
                is_interactive_qa=True
            )
            
            # Update session history
            qa_history_list.append({"question": current_question, "answer": answer})
            
            next_question_index = current_question_index + 1
            if next_question_index < len(INITIAL_INTERACTIVE_QA_QUESTIONS):
                next_question = INITIAL_INTERACTIVE_QA_QUESTIONS[next_question_index]
                crud.update_interactive_qa_session(self.db, session, current_question_index=next_question_index, qa_history=qa_history_list)
                logger.info(f"Session {session.id}: Answered Q{current_question_index}, next Q{next_question_index}.")
                return schemas.InteractiveQAResponse(
                    session_id=session.id,
                    project_id=project_id,
                    next_question=next_question,
                    is_complete=False,
                    message="Answer recorded. Here's the next question."
                )
            else:
                crud.update_interactive_qa_session(self.db, session, status="completed", qa_history=qa_history_list)
                logger.info(f"Interactive Q&A session {session.id} completed.")
                return schemas.InteractiveQAResponse(
                    session_id=session.id,
                    project_id=project_id,
                    next_question=None,
                    is_complete=True,
                    message="All predefined questions answered. Session completed."
                )
        else:
            logger.warning(f"Session {session.id}: No more questions expected. Session status: {session.status}.")
            return schemas.InteractiveQAResponse(
                session_id=session.id,
                project_id=project_id,
                next_question=None,
                is_complete=True,
                message="No more questions available for this session."
            )

    def generate_questions_from_document(self, project_id: str, document_id: str) -> schemas.DocumentQAGenerateQuestionsResponse:
        document_entry = crud.get_document_knowledge_entry(self.db, document_id)
        if not document_entry or document_entry.project_id != project_id:
            logger.error(f"Document {document_id} not found or does not belong to project {project_id}.")
            raise ValueError("Document not found or invalid for this project.")

        # Retrieve text chunks associated with this document
        text_chunks = crud.get_text_knowledge_entries_by_document_id(self.db, document_id)
        
        suggested_questions = []
        # For simplicity, generate questions from the first few chunks or a sample
        # In a real app, you might sample chunks more intelligently or limit the number of LLM calls
        chunks_to_process = text_chunks[:3] # Process up to first 3 chunks for hackathon

        for chunk_entry in chunks_to_process:
            questions = self.llm_service.generate_questions_from_document_chunk(chunk_entry.answer) # 'answer' holds the chunk content
            suggested_questions.extend(questions)
        
        logger.info(f"Generated {len(suggested_questions)} questions from document {document_id}.")
        return schemas.DocumentQAGenerateQuestionsResponse(
            project_id=project_id,
            document_id=document_id,
            suggested_questions=list(set(suggested_questions)), # Remove duplicates
            message="Suggested questions generated from document content."
        )
