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
from fastapi import HTTPException # For explicit HTTP exceptions

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

            # Store in relational DB
            crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                question=question,
                answer=answer,
                document_knowledge_entry_id=document_knowledge_entry_id,
                source_context=answer, # For static Q&A, the answer itself is the context
                is_interactive_qa=False 
            )
            
            # Prepare for vector DB ingestion
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
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for document ingestion.")
            raise ValueError(f"Project with ID {project_id} not found.")

        temp_dir = "/tmp/knowledge_relay_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file_name}")
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
            # Store chunk content as a TextKnowledgeEntry
            crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                document_knowledge_entry_id=document_db_entry.id,
                answer=doc_chunk.page_content, 
                source_context=doc_chunk.page_content, 
                is_interactive_qa=False 
            )

            # Prepare for ChromaDB: Embed the chunk content
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
        else:
            logger.warning(f"No text chunks extracted from {file_name} for project {project_id}.")

        os.remove(temp_file_path)
        logger.info(f"Cleaned up temporary file: {temp_file_path}")

        return schemas.FileUploadResponse(
            project_id=project_id,
            file_name=file_name,
            message="Document ingested successfully.",
            document_id=document_db_entry.id
        )

    def generate_questions_from_document(self, project_id: str, document_id: str, num_questions_per_chunk: int = 2, max_total_questions: int = 10) -> schemas.DocumentQAGenerateQuestionsResponse:
        """
        Generates questions from a document's content and stores them as unanswered
        TextKnowledgeEntry records, returning their IDs. Aims for a total number of questions.
        """
        document_entry = crud.get_document_knowledge_entry(self.db, document_id)
        if not document_entry or document_entry.project_id != project_id:
            logger.error(f"Document {document_id} not found or does not belong to project {project_id}.")
            raise ValueError("Document not found or invalid for this project.")

        text_chunks = crud.get_text_knowledge_entries_by_document_id(self.db, document_id)
        
        generated_question_entry_ids = []
        
        # Iterate over all chunks to generate questions, limiting total questions
        for chunk_entry in text_chunks:
            if len(generated_question_entry_ids) >= max_total_questions:
                break # Stop if we've generated enough questions
            
            questions = self.llm_service.generate_questions_from_document_chunk(chunk_entry.answer, num_questions=num_questions_per_chunk)
            
            for question_text in questions:
                if len(generated_question_entry_ids) >= max_total_questions:
                    break
                # Store each generated question as a new TextKnowledgeEntry with no answer yet
                new_qa_entry = crud.create_text_knowledge_entry(
                    db=self.db,
                    project_id=project_id,
                    document_knowledge_entry_id=document_id,
                    question=question_text,
                    answer=None, # Initially unanswered
                    source_context=chunk_entry.answer, # Context from which question was generated
                    is_interactive_qa=True # Part of interactive Q&A, but document-specific
                )
                generated_question_entry_ids.append(new_qa_entry.id)
        
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

        # 1. Check for existing unanswered questions in DB
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
        
        # 2. If no unanswered DB questions, check predefined questions
        # This will create a session without an initial current_question_text_entry_id
        # and rely on current_question_index for predefined questions.
        session = crud.create_project_qa_session(self.db, project_id) 

        if session.current_question_index < len(INITIAL_PROJECT_QA_QUESTIONS): 
            first_predefined_question = INITIAL_PROJECT_QA_QUESTIONS[session.current_question_index] 
            # No text_entry_id for predefined questions initially as they are not in TextKnowledgeEntry yet
            logger.info(f"Started project Q&A session {session.id} for project {project_id} with predefined question {session.current_question_index}.")
            return schemas.ProjectQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                question=first_predefined_question,
                question_entry_id=None, # For predefined, it's not a TextKnowledgeEntry yet
                is_complete=False,
                message="Starting with predefined project questions."
            )
        
        # 3. If all predefined questions exhausted and no unanswered DB questions, generate a new one
        try:
            # Pass empty list for existing_qa to start fresh LLM generation
            llm_generated_question = self.llm_service.generate_static_qa_question(existing_qa=[]) 
            new_qa_entry = crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                question=llm_generated_question,
                answer=None, 
                source_context=None, # No specific source context for LLM generated
                is_interactive_qa=True 
            )
            # Update the session to point to this newly generated question
            crud.update_project_qa_session(self.db, session, current_question_text_entry_id=new_qa_entry.id, current_question_index=len(INITIAL_PROJECT_QA_QUESTIONS)) # Set index beyond predefined
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
            # Fallback if LLM fails, mark session complete
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

        # Determine the current question based on session state
        if current_question_entry_id:
            # If we're answering an LLM-generated or previously existing unanswered question
            current_question_entry = crud.get_text_knowledge_entry_by_id(self.db, current_question_entry_id)
            if not current_question_entry:
                logger.error(f"TextKnowledgeEntry {current_question_entry_id} not found for project Q&A session {session_id}.")
                raise HTTPException(status_code=400, detail="Current question entry not found.")
            current_question = current_question_entry.question
            
            # Update the answer in the existing TextKnowledgeEntry
            crud.update_text_knowledge_entry_answer(self.db, current_question_entry_id, answer)
            logger.info(f"Updated answer for TextKnowledgeEntry {current_question_entry_id}.")
            
            # Ingest this Q&A into ChromaDB for retrieval
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
                ids=[f"project_qa_{current_question_entry_id}"] # Use entry ID for Chroma
            )

        elif session.current_question_index < len(INITIAL_PROJECT_QA_QUESTIONS):
            # If we're answering a predefined question
            current_question = INITIAL_PROJECT_QA_QUESTIONS[session.current_question_index]
            
            # For predefined questions, create a new TextKnowledgeEntry upon answering
            answered_qa_entry = crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                question=current_question,
                answer=answer,
                source_context=answer,
                is_interactive_qa=True
            )
            # This is the ID for the *newly created* entry, so ChromaDB will use this
            current_question_entry_id = answered_qa_entry.id 

            # Ingest this Q&A into ChromaDB for retrieval
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

        # Find the next question: Prioritize existing unanswered questions, then predefined, then LLM-generated
        unanswered_db_questions = crud.get_unanswered_project_questions(self.db, project_id)
        if unanswered_db_questions:
            # Filter out the current question if it's still in the list (shouldn't be if updated)
            remaining_unanswered = [q for q in unanswered_db_questions if q.id != current_question_entry_id]
            if remaining_unanswered:
                next_question_entry = remaining_unanswered[0]
                next_question_text = next_question_entry.question
                next_question_entry_id = next_question_entry.id
                crud.update_project_qa_session(self.db, session, current_question_text_entry_id=next_question_entry.id, qa_history=qa_history_list)
                logger.info(f"Session {session.id}: Next Q (DB) {next_question_entry.id}.")
            else: # All DB questions answered, move to predefined if not done
                next_predefined_index = session.current_question_index + 1
                if next_predefined_index < len(INITIAL_PROJECT_QA_QUESTIONS):
                    next_question_text = INITIAL_PROJECT_QA_QUESTIONS[next_predefined_index]
                    crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=None, qa_history=qa_history_list)
                    logger.info(f"Session {session.id}: Next Q (Predefined) {next_predefined_index}.")
                else: # Predefined exhausted, generate new LLM question
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
                        crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=new_qa_entry.id, qa_history=qa_history_list)
                        logger.info(f"Session {session.id}: Next Q (LLM Generated) {new_qa_entry.id}.")
                    except Exception as e:
                        logger.error(f"Failed to generate next LLM question for project {project_id}: {e}")
                        session_is_complete = True # Mark complete if LLM fails to generate
        else: # If no DB questions found at the start of next search, check predefined
            next_predefined_index = session.current_question_index + 1
            if next_predefined_index < len(INITIAL_PROJECT_QA_QUESTIONS):
                next_question_text = INITIAL_PROJECT_QA_QUESTIONS[next_predefined_index]
                crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=None, qa_history=qa_history_list)
                logger.info(f"Session {session.id}: Next Q (Predefined) {next_predefined_index}.")
            else: # Predefined exhausted, generate new LLM question
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
                    crud.update_project_qa_session(self.db, session, current_question_index=next_predefined_index, current_question_text_entry_id=new_qa_entry.id, qa_history=qa_history_list)
                    logger.info(f"Session {session.id}: Next Q (LLM Generated) {new_qa_entry.id}.")
                except Exception as e:
                    logger.error(f"Failed to generate next LLM question for project {project_id}: {e}")
                    session_is_complete = True # Mark complete if LLM fails to generate
        
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

    def start_document_qa_session(self, project_id: str, document_id: str) -> schemas.DocumentQASessionStartResponse: 
        """
        Starts an interactive Q&A session for questions generated from a specific document.
        """
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for starting document Q&A.")
            raise ValueError(f"Project with ID {project_id} not found.")
        
        document_entry = crud.get_document_knowledge_entry(self.db, document_id)
        if not document_entry or document_entry.project_id != project_id:
            logger.error(f"Document {document_id} not found or does not belong to project {project_id}.")
            raise ValueError("Document not found or invalid for this project.")

        unanswered_questions = crud.get_unanswered_questions_for_document(self.db, project_id, document_id)
        
        if not unanswered_questions:
            session = crud.create_document_qa_session(self.db, project_id, document_id, current_question_text_entry_id=None)
            crud.update_document_qa_session(self.db, session, status="no_questions")
            logger.info(f"Started document Q&A session {session.id} for document {document_id}, but no unanswered questions found.")
            return schemas.DocumentQASessionStartResponse(
                session_id=session.id,
                project_id=project_id,
                document_id=document_id,
                question=None,
                question_entry_id=None,
                is_complete=True, 
                message="No unanswered questions found for this document. Session completed."
            )
        
        first_question_entry = unanswered_questions[0]
        session = crud.create_document_qa_session(self.db, project_id, document_id, first_question_entry.id)
        
        logger.info(f"Started document Q&A session {session.id} for document {document_id} with first question {first_question_entry.id}.")
        return schemas.DocumentQASessionStartResponse(
            session_id=session.id,
            project_id=project_id,
            document_id=document_id,
            question=first_question_entry.question,
            question_entry_id=first_question_entry.id,
            is_complete=False,
            message="Document Q&A session started."
        )

    def respond_to_document_qa(self, session_id: str, project_id: str, answer: str) -> schemas.DocumentQAResponse: 
        """
        Handles old member's answers in a document-specific interactive Q&A session.
        """
        session = crud.get_document_qa_session(self.db, session_id)
        if not session or session.project_id != project_id or session.status == "completed":
            logger.error(f"Document Q&A session {session_id} not found, invalid for project {project_id}, or already completed.")
            raise HTTPException(status_code=400, detail="Document Q&A session not found, invalid, or already completed.")
        
        if session.status == "no_questions":
             return schemas.DocumentQAResponse(
                session_id=session.id,
                project_id=project_id,
                next_question=None,
                next_question_entry_id=None,
                is_complete=True,
                message="No questions available for this session. Session completed."
            )

        current_question_entry_id = session.current_question_text_entry_id
        current_question_entry = crud.get_text_knowledge_entry_by_id(self.db, current_question_entry_id)

        if not current_question_entry or current_question_entry.document_knowledge_entry_id != session.document_id:
            logger.error(f"Current question entry {current_question_entry_id} not found or mismatch with session document.")
            raise HTTPException(status_code=400, detail="Current question entry invalid or missing.")

        # Update the answer for the current question
        crud.update_text_knowledge_entry_answer(self.db, current_question_entry_id, answer)
        logger.info(f"Answered question {current_question_entry_id} for document {session.document_id}.")

        # Ingest the answered Q&A into ChromaDB
        chroma_manager = ChromaDBManager(project_id)
        # Ensure document_entry is loaded for file_name
        doc_entry_for_meta = crud.get_document_knowledge_entry(self.db, session.document_id)
        file_name_for_meta = doc_entry_for_meta.file_name if doc_entry_for_meta else "Unknown Document"

        chroma_manager.add_documents(
            documents=[answer], # Embed the answer
            metadatas=[{
                "type": "document_qa",
                "project_id": project_id,
                "document_id": session.document_id,
                "file_name": file_name_for_meta, 
                "question": current_question_entry.question,
                "answer": answer,
                "source_context": current_question_entry.source_context if current_question_entry.source_context else answer 
            }],
            ids=[f"doc_qa_{current_question_entry_id}"]
        )

        # Find the next unanswered question for the *same document*
        unanswered_questions = crud.get_unanswered_questions_for_document(self.db, project_id, session.document_id)
        
        if unanswered_questions:
            next_question_entry = unanswered_questions[0] # Get the next one
            crud.update_document_qa_session(self.db, session, current_question_text_entry_id=next_question_entry.id)
            logger.info(f"Session {session.id}: Answered question, next question {next_question_entry.id}.")
            return schemas.DocumentQAResponse(
                session_id=session.id,
                project_id=project_id,
                next_question=next_question_entry.question,
                next_question_entry_id=next_question_entry.id,
                is_complete=False,
                message="Answer recorded. Here's the next question for this document."
            )
        else:
            crud.update_document_qa_session(self.db, session, status="completed", current_question_text_entry_id=None)
            logger.info(f"Document Q&A session {session.id} completed. All questions answered.")
            return schemas.DocumentQAResponse(
                session_id=session.id,
                project_id=project_id,
                next_question=None,
                next_question_entry_id=None,
                is_complete=True,
                message="All generated questions for this document have been answered. Session completed."
            )