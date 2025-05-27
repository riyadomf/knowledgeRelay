import os
from pathlib import Path
import uuid
import json
import logging
from typing import List, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session
from langchain_core.documents import Document

from app import schemas, crud, models
from app.core.vector_store import ChromaDBManager
from app.core.document_loaders import load_document
from app.core.text_splitters import split_documents, load_and_chunk_file
from app.services.llm_service import LLMService
from app.config import Settings

logger = logging.getLogger(__name__)

# Predefined initial questions for project-wide interactive Q&A
INITIAL_PROJECT_QA_QUESTIONS = [ 
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

    def ingest_static_qa(self, project_id: str, questions_answers: List[Dict[str, str]], document_knowledge_entry_id: Optional[str] = None):
        """
        Ingests static (bulk) Q&A pairs into the knowledge base.
        Can be general project Q&A or linked to a specific document.
        """
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

            crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                question=question,
                answer=answer,
                document_knowledge_entry_id=document_knowledge_entry_id,
                source_context=answer, 
                is_interactive_qa=False 
            )
            
            documents_to_add.append(answer) 
            metadata = {
                "type": "static_qa",
                "project_id": project_id,
                "question": question,
                "answer": answer,
                "source_context": answer
            }
            if document_knowledge_entry_id:
                metadata["document_id"] = document_knowledge_entry_id
                doc_entry = crud.get_document_knowledge_entry(self.db, document_knowledge_entry_id)
                if doc_entry:
                    metadata["file_name"] = doc_entry.file_name
                else:
                    logger.warning(f"Document entry with ID {document_knowledge_entry_id} not found for metadata.")
                    metadata["file_name"] = "Unknown Document"
            metadatas_to_add.append(metadata)
            ids_to_add.append(f"static_qa_{uuid.uuid4()}")

        if documents_to_add:
            chroma_manager.add_documents(documents_to_add, metadatas_to_add, ids_to_add)
        
        logger.info(f"Ingested {len(questions_answers)} static Q&A pairs for project {project_id}.")
        return {"message": "Static Q&A ingested successfully."}

    def ingest_document(self, project_id: str, file: bytes, file_name: str, file_type: str):
        """
        Ingests a document: creates a DocumentKnowledgeEntry, chunks the document,
        and stores the chunks' embeddings in ChromaDB.
        """
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for document ingestion.")
            raise ValueError(f"Project with ID {project_id} not found.")

        temp_dir = "/tmp/knowledge_relay_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        unique_file_name = f"{uuid.uuid4()}_{file_name}"
        temp_file_path = os.path.join(temp_dir, unique_file_name)
        with open(temp_file_path, "wb") as f:
            f.write(file)
        logger.info(f"Temporarily saved uploaded file to: {temp_file_path}")

        loaded_documents = load_document(temp_file_path, file_type)
        split_docs = split_documents(loaded_documents)
        
        
        document_db_entry = crud.create_document_knowledge_entry(
            db=self.db,
            project_id=project_id,
            file_name=file_name,
            file_path=temp_file_path 
        )
        logger.info(f"Created document entry in DB: {document_db_entry.id}")

        texts_for_chroma = []
        metadatas_for_chroma = []
        ids_for_chroma = []

        for i, doc_chunk in enumerate(split_docs):
            metadata = {
                "project_id": project_id,
                "document_id": document_db_entry.id,
                "file_name": file_name,
                "type": "document_chunk", 
                "source_context": doc_chunk.page_content
            }
            if 'page' in doc_chunk.metadata:
                metadata['page_number'] = doc_chunk.metadata['page']
            if 'source' in doc_chunk.metadata:
                metadata['source'] = doc_chunk.metadata['source'] 
            
            texts_for_chroma.append(doc_chunk.page_content)
            metadatas_for_chroma.append(metadata)
            ids_for_chroma.append(f"doc_chunk_{document_db_entry.id}_{i}")

        chroma_manager = ChromaDBManager(project_id)
        if texts_for_chroma:
            chroma_manager.add_documents(texts_for_chroma, metadatas_for_chroma, ids_for_chroma)
            logger.info(f"Embedded and stored {len(texts_for_chroma)} chunks for document {document_db_entry.id} in ChromaDB.")
        else:
            logger.warning(f"No text chunks extracted from {file_name} for project {project_id}.")

        return schemas.FileUploadResponse(
            project_id=project_id,
            file_name=file_name,
            message="Document ingested and indexed for retrieval. Generate questions next.",
            document_id=document_db_entry.id
        )

    def generate_questions_from_document(self, project_id: str, document_id: str, num_questions_per_chunk: int = 2, max_total_questions: int = 10) -> schemas.DocumentQAGenerateQuestionsResponse:
        """
        Generates questions from a document's content and stores them as unanswered
        TextKnowledgeEntry records, returning their IDs. Aims for a total number of questions.
        This now re-loads and chunks the document from its stored file path.
        """
        document_entry = crud.get_document_knowledge_entry(self.db, document_id)
        if not document_entry or document_entry.project_id != project_id:
            logger.error(f"Document {document_id} not found or does not belong to project {project_id}.")
            raise ValueError("Document not found or invalid for this project.")
        
        if not document_entry.file_path or not os.path.exists(document_entry.file_path):
            logger.error(f"File for document {document_id} not found at path: {document_entry.file_path}")
            raise HTTPException(status_code=500, detail="Document file not found on server.")

        file_extension = document_entry.file_name.split(".")[-1].lower()
        loaded_documents = load_document(document_entry.file_path, file_extension)
        split_docs = split_documents(loaded_documents)
        
        generated_question_entry_ids = []
        
        for chunk_idx, doc_chunk in enumerate(split_docs):
            if len(generated_question_entry_ids) >= max_total_questions:
                logger.info(f"Reached max_total_questions ({max_total_questions}). Stopping question generation.")
                break
            
            questions = self.llm_service.generate_questions_from_document_chunk(doc_chunk.page_content, num_questions=num_questions_per_chunk)
            
            for question_text in questions:
                if len(generated_question_entry_ids) >= max_total_questions:
                    break
                
                new_qa_entry = crud.create_text_knowledge_entry(
                    db=self.db,
                    project_id=project_id,
                    document_knowledge_entry_id=document_id,
                    question=question_text,
                    answer=None, 
                    source_context=doc_chunk.page_content, 
                    is_interactive_qa=True 
                )
                generated_question_entry_ids.append(new_qa_entry.id)
                logger.info(f"Generated and stored question '{question_text}' for document {document_id}.")
        
        logger.info(f"Generated {len(generated_question_entry_ids)} questions from document {document_id} and stored them as unanswered entries.")
        return schemas.DocumentQAGenerateQuestionsResponse(
            project_id=project_id,
            document_id=document_id,
            question_entry_ids=generated_question_entry_ids,
            message=f"Generated {len(generated_question_entry_ids)} suggested questions from document content and stored for answering."
        )

    def start_project_qa_session(self, project_id: str) -> schemas.ProjectQASessionStartResponse: 
        """
        Starts a project-wide interactive Q&A session. Prioritizes unanswered questions
        from DB, then predefined, then new LLM-generated.
        """
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for starting project Q&A.")
            raise ValueError(f"Project with ID {project_id} not found.")

        unanswered_db_questions = crud.get_unanswered_project_questions(self.db, project_id)
        if unanswered_db_questions:
            first_question_entry = unanswered_db_questions[0]
            session = crud.create_project_qa_session(self.db, project_id, first_question_entry.id)
            logger.info(f"Started project Q&A session {session.id} for project {project_id} with existing unanswered question {first_question_entry.id}.")
            return schemas.ProjectQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                question=first_question_entry.question,
                question_entry_id=first_question_entry.id,
                is_complete=False,
                message="Continuing with existing unanswered project questions."
            )
        
        session = crud.create_project_qa_session(self.db, project_id) 

        if session.current_question_index < len(INITIAL_PROJECT_QA_QUESTIONS): 
            first_predefined_question = INITIAL_PROJECT_QA_QUESTIONS[session.current_question_index] 
            logger.info(f"Started project Q&A session {session.id} for project {project_id} with predefined question {session.current_question_index}.")
            return schemas.ProjectQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                question=first_predefined_question,
                question_entry_id=None, 
                is_complete=False,
                message="Starting with predefined project questions."
            )
        
        try:
            llm_generated_question = self.llm_service.generate_static_qa_question(existing_qa=[]) 
            new_qa_entry = crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                question=llm_generated_question,
                answer=None, 
                source_context=None, 
                is_interactive_qa=True 
            )
            crud.update_project_qa_session(self.db, session, current_question_text_entry_id=new_qa_entry.id, current_question_index=len(INITIAL_PROJECT_QA_QUESTIONS)) 
            logger.info(f"Started project Q&A session {session.id} for project {project_id} with new LLM-generated question {new_qa_entry.id}.")
            return schemas.ProjectQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                question=new_qa_entry.question,
                question_entry_id=new_qa_entry.id,
                is_complete=False,
                message="Generating new project questions."
            )
        except Exception as e:
            logger.error(f"Failed to generate initial LLM question for project {project_id}: {e}")
            crud.update_project_qa_session(self.db, session, status="completed")
            return schemas.ProjectQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                question="No questions available and failed to generate new ones. Session complete.",
                question_entry_id=None,
                is_complete=True,
                message="Failed to start session due to LLM error."
            )

    def respond_to_project_qa(self, session_id: str, project_id: str, answer: str) -> schemas.ProjectQAResponse: 
        """
        Handles old member's answers in a project-wide interactive Q&A session.
        Updates the answer in the corresponding TextKnowledgeEntry.
        """
        session = crud.get_project_qa_session(self.db, session_id) 
        if not session or session.project_id != project_id or session.status == "completed":
            logger.error(f"Project Q&A session {session_id} not found, invalid for project {project_id}, or already completed.")
            raise HTTPException(status_code=400, detail="Project Q&A session not found, invalid, or already completed.")
        
        qa_history_list = json.loads(session.qa_history)
        current_question = None
        current_question_entry_id = session.current_question_text_entry_id

        if current_question_entry_id:
            current_question_entry = crud.get_text_knowledge_entry_by_id(self.db, current_question_entry_id)
            if not current_question_entry:
                logger.error(f"TextKnowledgeEntry {current_question_entry_id} not found for project Q&A session {session_id}.")
                raise HTTPException(status_code=400, detail="Current question entry not found.")
            current_question = current_question_entry.question

            crud.update_text_knowledge_entry_answer(self.db, current_question_entry_id, answer)
            logger.info(f"Updated answer for TextKnowledgeEntry {current_question_entry_id}.")

            chroma_manager = ChromaDBManager(project_id)
            chroma_manager.add_documents(
                documents=[answer],
                metadatas=[{
                    "type": "project_qa",
                    "project_id": project_id,
                    "question": current_question,
                    "answer": answer,
                    "source_context": current_question_entry.source_context if current_question_entry.source_context else answer
                }],
                ids=[f"project_qa_{current_question_entry_id}"] 
            )

        elif session.current_question_index < len(INITIAL_PROJECT_QA_QUESTIONS):
            current_question = INITIAL_PROJECT_QA_QUESTIONS[session.current_question_index]
            
            answered_qa_entry = crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                question=current_question,
                answer=answer,
                source_context=answer,
                is_interactive_qa=True
            )
            current_question_entry_id = answered_qa_entry.id 

            chroma_manager = ChromaDBManager(project_id)
            chroma_manager.add_documents(
                documents=[answer],
                metadatas=[{
                    "type": "project_qa",
                    "project_id": project_id,
                    "question": current_question,
                    "answer": answer,
                    "source_context": answer
                }],
                ids=[f"project_qa_{current_question_entry_id}"]
            )
        else:
            logger.warning(f"Session {session_id}: No current question to answer or index out of bounds. Status: {session.status}.")
            raise HTTPException(status_code=400, detail="No current question to answer or session state invalid.")

        if current_question:
            qa_history_list.append({"question": current_question, "answer": answer})

        next_question_text = None
        next_question_entry_id = None
        session_is_complete = False

        unanswered_db_questions = crud.get_unanswered_project_questions(self.db, project_id)
        if unanswered_db_questions:
            remaining_unanswered = [q for q in unanswered_db_questions if q.id != current_question_entry_id]
            if remaining_unanswered:
                next_question_entry = remaining_unanswered[0]
                next_question_text = next_question_entry.question
                next_question_entry_id = next_question_entry.id
                crud.update_project_qa_session(self.db, session, current_question_text_entry_id=next_question_entry.id, qa_history=qa_history_list, status="active")
                logger.info(f"Session {session.id}: Next Q (DB) {next_question_entry.id}.")
            else: 
                next_predefined_index = session.current_question_index + 1
                if next_predefined_index < len(INITIAL_PROJECT_QA_QUESTIONS):
                    next_question_text = INITIAL_PROJECT_QA_QUESTIONS[next_predefined_index]
                    crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=None, qa_history=qa_history_list, status="active")
                    logger.info(f"Session {session.id}: Next Q (Predefined) {next_predefined_index}.")
                else: 
                    try:
                        llm_generated_question = self.llm_service.generate_static_qa_question(existing_qa=qa_history_list)
                        new_qa_entry = crud.create_text_knowledge_entry(
                            db=self.db,
                            project_id=project_id,
                            question=llm_generated_question,
                            answer=None,
                            source_context=None,
                            is_interactive_qa=True
                        )
                        next_question_text = new_qa_entry.question
                        next_question_entry_id = new_qa_entry.id
                        crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=new_qa_entry.id, qa_history=qa_history_list, status="active")
                        logger.info(f"Session {session.id}: Next Q (LLM Generated) {new_qa_entry.id}.")
                    except Exception as e:
                        logger.error(f"Failed to generate next LLM question for project {project_id}: {e}")
                        session_is_complete = True 
        else: 
            next_predefined_index = session.current_question_index + 1
            if next_predefined_index < len(INITIAL_PROJECT_QA_QUESTIONS):
                next_question_text = INITIAL_PROJECT_QA_QUESTIONS[next_predefined_index]
                crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=None, qa_history=qa_history_list, status="active")
                logger.info(f"Session {session.id}: Next Q (Predefined) {next_predefined_index}.")
            else: 
                try:
                    llm_generated_question = self.llm_service.generate_static_qa_question(existing_qa=qa_history_list)
                    new_qa_entry = crud.create_text_knowledge_entry(
                        db=self.db,
                        project_id=project_id,
                        question=llm_generated_question,
                        answer=None,
                        source_context=None,
                        is_interactive_qa=True
                    )
                    next_question_text = new_qa_entry.question
                    next_question_entry_id = new_qa_entry.id
                    crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=new_qa_entry.id, qa_history=qa_history_list, status="active")
                    logger.info(f"Session {session.id}: Next Q (LLM Generated) {new_qa_entry.id}.")
                except Exception as e:
                    logger.error(f"Failed to generate next LLM question for project {project_id}: {e}")
                    session_is_complete = True 
        
        if not next_question_text and not session_is_complete:
            session_is_complete = True
            crud.update_project_qa_session(self.db, session, status="completed", qa_history=qa_history_list, current_question_text_entry_id=None)
            logger.info(f"Project Q&A session {session.id} completed.")
        elif session_is_complete:
             crud.update_project_qa_session(self.db, session, status="completed", qa_history=qa_history_list, current_question_text_entry_id=None)


        return schemas.ProjectQAResponse( 
            session_id=session.id,
            project_id=project_id,
            next_question=next_question_text,
            next_question_entry_id=next_question_entry_id,
            is_complete=session_is_complete,
            message="Answer recorded." + (" Session completed." if session_is_complete else " Here's the next question.")
        )

    # Removed start_document_qa_session and respond_to_document_qa from IngestionService
    def get_next_document_question(self, project_id: str, document_id: str) -> schemas.GetNextDocumentQuestionResponse:
        """
        Retrieves the oldest unanswered question for a specific document.
        """
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for getting next document question.")
            raise ValueError(f"Project with ID {project_id} not found.")
        
        document_entry = crud.get_document_knowledge_entry(self.db, document_id)
        if not document_entry or document_entry.project_id != project_id:
            logger.error(f"Document {document_id} not found or does not belong to project {project_id}.")
            raise ValueError("Document not found or invalid for this project.")

        unanswered_questions = crud.get_unanswered_questions_for_document(self.db, project_id, document_id)
        
        if not unanswered_questions:
            logger.info(f"No unanswered questions found for document {document_id} in project {project_id}.")
            return schemas.GetNextDocumentQuestionResponse(
                project_id=project_id,
                document_id=document_id,
                question=None,
                question_entry_id=None,
                is_complete=True, 
                message="No unanswered questions found for this document."
            )
        
        next_question_entry = unanswered_questions[0] # Get the oldest one
        
        logger.info(f"Fetched next unanswered question {next_question_entry.id} for document {document_id}.")
        return schemas.GetNextDocumentQuestionResponse(
            project_id=project_id,
            document_id=document_id,
            question=next_question_entry.question,
            question_entry_id=next_question_entry.id,
            is_complete=False,
            message="Next unanswered question retrieved."
        )

    def answer_document_question(self, project_id: str, question_entry_id: str, answer: str) -> schemas.AnswerDocumentQuestionResponse:
        """
        Updates a specific document question with an answer and ingests it into ChromaDB.
        """
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for answering document question.")
            raise ValueError(f"Project with ID {project_id} not found.")

        question_entry = crud.get_text_knowledge_entry_by_id(self.db, question_entry_id)
        if not question_entry or question_entry.project_id != project_id or question_entry.answer is not None:
            logger.error(f"Question entry {question_entry_id} not found, invalid for project {project_id}, or already answered.")
            raise HTTPException(status_code=400, detail="Question entry not found, invalid, or already answered.")
        
        document_id = question_entry.document_knowledge_entry_id
        if not document_id:
            logger.error(f"Question entry {question_entry_id} is not linked to a document.")
            raise HTTPException(status_code=400, detail="Question is not linked to a document.")

        crud.update_text_knowledge_entry_answer(self.db, question_entry_id, answer)
        logger.info(f"Answered question {question_entry_id} for document {document_id}.")

        chroma_manager = ChromaDBManager(project_id)
        doc_entry_for_meta = crud.get_document_knowledge_entry(self.db, document_id)
        file_name_for_meta = doc_entry_for_meta.file_name if doc_entry_for_meta else "Unknown Document"

        chroma_manager.add_documents(
            documents=[answer], 
            metadatas=[{
                "type": "document_qa",
                "project_id": project_id,
                "document_id": document_id,
                "file_name": file_name_for_meta, 
                "question": question_entry.question,
                "answer": answer,
                "source_context": question_entry.source_context if question_entry.source_context else answer 
            }],
            ids=[f"doc_qa_{question_entry_id}"]
        )

        # Check if there are more unanswered questions for this document
        remaining_unanswered = crud.get_unanswered_questions_for_document(self.db, project_id, document_id)
        is_complete = not bool(remaining_unanswered)

        return schemas.AnswerDocumentQuestionResponse(
            project_id=project_id,
            question_entry_id=question_entry_id,
            message="Answer recorded and knowledge base updated." + (" All questions for this document are now answered." if is_complete else ""),
            is_complete=is_complete
        )
