from sqlalchemy.orm import Session
from app import schemas, crud, models
from app.core.vector_store import ChromaDBManager
from app.core.document_loaders import load_document
from app.core.text_splitters import split_documents
from app.services.llm_service import LLMService # Assuming this is your LLM integration service
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os
import uuid
import json
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain


logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService() # Initialize your LLM service

    def _generate_question_with_llm(self, project_id: str) -> Optional[str]:
        """
        Generates a new question using an LLM based on existing project knowledge.
        This function should leverage your LLM API and potentially retrieve context
        from your ChromaDB or relational database to inform the question generation.
        Returns the generated question string or None if generation fails.
        """
        logger.info(f"Attempting to generate a new question for project {project_id} using LLM.")
        try:
            # Retrieve some existing knowledge to provide context to the LLM.
            # This is crucial for the LLM to generate relevant questions.
            existing_knowledge_entries = crud.get_recent_text_knowledge_entries(self.db, project_id, limit=5)
            context_for_llm = ""
            for entry in existing_knowledge_entries:
                if entry.question and entry.answer:
                    context_for_llm += f"Q: {entry.question}\nA: {entry.answer}\n"
                elif entry.source_context:
                    context_for_llm += f"Context: {entry.source_context}\n"

            # **IMPORTANT:** Replace this placeholder with your actual LLM API call.
            # Example using a hypothetical LLMService method:
            # prompt = f"Given the following knowledge about a project:\n{context_for_llm}\nWhat is an important question an old developer should answer about this project? Only provide the question."
            # generated_question = self.llm_service.generate_question(prompt)
               # Main QA prompt using retrieved context
            # Update prompt to request a list of questions
            qa_prompt_text = (
               f"You are an AI assistant helping an experienced developer transfer project knowledge."
               "  Your goal is to ask insightful, open-ended questions to extract critical information.  "
               "Focus on areas like project purpose, architecture, key technologies, deployment, common issues, "
                "team practices, and important contacts. Avoid asking questions that have already been covered. "
                "If you think enough information has been gathered, you can suggest concluding the session. "
               "Given the following knowledge about a project:\n{context_for_llm}\n"
               " List 5 important questions that a developer should be able to answer about this project. "
               "Return them as a numbered list (e.g., 1. ..., 2. ..., etc.)."
            )
            

            
            # Replace with your actual LLM call
            llm_response = self.llm_service.x(qa_prompt_text)

            # Extract list from numbered response (basic example, improve as needed)
            questions = [line.strip().split(". ", 1)[1] for line in llm_response.splitlines() if line.strip().startswith(tuple("123456789"))]
            logger.info(f"LLM generated questions for project {project_id}: {questions}")
            return questions
        
        except Exception as e:
            logger.error(f"Error generating question with LLM for project {project_id}: {e}")
            return None

    def ingest_static_qa(self, project_id: str, questions_answers: List[Dict[str, str]], is_interactive_qa: bool = False, document_knowledge_entry_id: Optional[str] = None):
        """
        Ingests static Q&A pairs into the system. This method is also used by the interactive Q&A flow
        to store the answered questions.
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
                source_context=answer,
                is_interactive_qa=is_interactive_qa
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
                metadata["file_name"] = crud.get_document_knowledge_entry(self.db, document_knowledge_entry_id).file_name if crud.get_document_knowledge_entry(self.db, document_knowledge_entry_id) else "Unknown Document"
            metadatas_to_add.append(metadata)
            ids_to_add.append(f"static_qa_{uuid.uuid4()}")

        if documents_to_add:
            chroma_manager.add_documents(documents_to_add, metadatas_to_add, ids_to_add)
        
        logger.info(f"Ingested {len(questions_answers)} static Q&A pairs for project {project_id}.")
        return {"message": "Static Q&A ingested successfully."}

    def ingest_document(self, project_id: str, file: bytes, file_name: str, file_type: str):
        """
        Handles the ingestion of documents, including saving them temporarily, splitting them into chunks,
        and storing them in both the relational database and ChromaDB.
        """
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
            crud.create_text_knowledge_entry(
                db=self.db,
                project_id=project_id,
                document_knowledge_entry_id=document_db_entry.id,
                answer=doc_chunk.page_content,
                source_context=doc_chunk.page_content
            )

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

    def start_interactive_qa_session(self, project_id: str) -> schemas.InteractiveQASessionStartResponse:
        """
        Initiates an interactive Q&A session for a given project. It prioritizes returning an existing
        unanswered question. If no such questions are found, it leverages an LLM to generate a new question.
        """
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for starting interactive Q&A.")
            raise ValueError(f"Project with ID {project_id} not found.")

        session = crud.create_interactive_qa_session(self.db, project_id)

        # 1. Check for any unanswered questions in TextKnowledgeEntry
        unanswered_question_entry = crud.get_unanswered_interactive_qa_question(self.db, project_id)

        question_to_ask = None
        is_complete = False

        if unanswered_question_entry:
            question_to_ask = unanswered_question_entry.question
            logger.info(f"Found unanswered question for project {project_id}: '{question_to_ask}'. Returning it.")
            crud.update_interactive_qa_session(self.db, session, current_question_index=0)
        else:
            # 2. If no unanswered questions, generate a new one using LLM
            new_questions = self._generate_question_with_llm(project_id)
            if new_questions:
                # Store the new question in TextKnowledgeEntry with a null answer
                for idx, question in enumerate(new_questions):
                    crud.create_text_knowledge_entry(
                    db=self.db,
                    project_id=project_id,
                    question=question,
                    answer=None,
                    source_context=None,
                    is_interactive_qa=True
                )
                question_to_ask = new_questions[0]  # Return the first one for the session start
                crud.update_interactive_qa_session(self.db, session, current_question_index=0)
                logger.info(f"Generated and stored {len(new_questions)} new questions for project {project_id}. First: '{question_to_ask}'")
                
            else:
                logger.warning("Failed to generate any initial question for interactive Q&A.")
                question_to_ask = "No questions available. Session complete."
                is_complete = True
                crud.update_interactive_qa_session(self.db, session, status="completed")

        logger.info(f"Started interactive Q&A session {session.id} for project {project_id}.")
        return schemas.InteractiveQASessionStartResponse(
            session_id=session.id,
            project_id=project_id,
            question=question_to_ask,
            is_complete=is_complete
        )
   
    def respond_to_interactive_qa(self, session_id: str, project_id: str, answer: str) -> schemas.InteractiveQAResponse:
        """
        Handles responses during an interactive Q&A session. It records the answer, updates the session history,
        and then determines the next question to ask (either an existing unanswered one or a newly generated one).
        """
        session = crud.get_interactive_qa_session(self.db, session_id)
        if not session or session.project_id != project_id:
            logger.error(f"Interactive Q&A session {session_id} not found or does not belong to project {project_id}.")
            raise ValueError("Interactive Q&A session not found or invalid for this project.")

        if session.status == "completed":
            logger.info(f"Interactive Q&A session {session.id} is already completed.")
            return schemas.InteractiveQAResponse(
                session_id=session.id,
                project_id=project_id,
                next_question=None,
                is_complete=True,
                message="Session already completed."
            )

        # Retrieve the *current* unanswered question that the user is responding to.
        # This assumes the client knows which question it is answering.
        # A more robust approach might be to pass the question ID from the client.
        # For now, we'll assume the most recently asked unanswered question is the one being answered.
        current_unanswered_question_entry = crud.get_unanswered_interactive_qa_question(self.db, project_id)

        if not current_unanswered_question_entry:
            logger.warning(f"Session {session.id}: No unanswered question found to respond to. This might indicate a logic error or a race condition.")
            next_questions = self._generate_question_with_llm(project_id)
            if next_questions:
                for idx, q in next_questions:
                    crud.create_text_knowledge_entry(
                      db=self.db, project_id=project_id, question=q, answer=None, is_interactive_qa=True
                    )
                message = "No active question to answer, but a new question has been generated."
                is_complete = False
            else:
                 message = "No more questions can be generated at this time."
                 is_complete = True
                 crud.update_interactive_qa_session(self.db, session, status="completed")

            return schemas.InteractiveQAResponse(
                session_id=session.id,
                project_id=project_id,
                next_question=next_questions[0],
                is_complete=is_complete,
                message=message
            )

        # Update the answer for the current question in TextKnowledgeEntry
        updated_entry = crud.update_text_knowledge_entry_answer(
            db=self.db,
            entry_id=current_unanswered_question_entry.id,
            answer=answer
        )
        if not updated_entry:
            logger.error(f"Failed to update answer for TextKnowledgeEntry {current_unanswered_question_entry.id}.")
            raise ValueError("Failed to record answer.")

        # Ingest the Q&A pair into ChromaDB for retrieval
        self.ingest_static_qa(
            project_id=project_id,
            questions_answers=[{"question": current_unanswered_question_entry.question, "answer": answer}],
            is_interactive_qa=True
        )

        # Update session history (assuming qa_history is a JSON list of dicts)
        qa_history_list = json.loads(session.qa_history) if session.qa_history else []
        qa_history_list.append({"question": current_unanswered_question_entry.question, "answer": answer})
        crud.update_interactive_qa_session(self.db, session, qa_history=json.dumps(qa_history_list))


        # Now, check for the next question:
        # 1. Look for another existing unanswered question
        next_unanswered_question_entry = crud.get_unanswered_interactive_qa_question(self.db, project_id)

        next_question_str = None
        session_is_complete = False
        response_message = "Answer recorded. Here's the next question."

        if next_unanswered_question_entry:
            next_question_str = next_unanswered_question_entry.question
            logger.info(f"Session {session.id}: Found next unanswered question: '{next_question_str}'.")
        else:
            # 2. If no more existing unanswered questions, generate a new one via LLM
            generated_new_question = self._generate_question_with_llm(project_id)
            if generated_new_question:
                # Store the new question with a null answer
                crud.create_text_knowledge_entry(
                    db=self.db,
                    project_id=project_id,
                    question=generated_new_question,
                    answer=None, # Initially null
                    source_context=None,
                    is_interactive_qa=True
                )
                next_question_str = generated_new_question
                logger.info(f"Session {session.id}: Generated and stored new question: '{generated_new_question}'.")
            else:
                # If LLM can't generate a new question, then the session is complete.
                session_is_complete = True
                crud.update_interactive_qa_session(self.db, session, status="completed")
                response_message = "All available questions answered, and no new questions could be generated. Session completed."
                logger.info(f"Interactive Q&A session {session.id} completed.")


        return schemas.InteractiveQAResponse(
            session_id=session.id,
            project_id=project_id,
            next_question=next_question_str,
            is_complete=session_is_complete,
            message=response_message
        )

    def generate_questions_from_document(self, project_id: str, document_id: str) -> schemas.DocumentQAGenerateQuestionsResponse:
        """
        Generates suggested questions based on the content of an uploaded document using an LLM.
        """
        document_entry = crud.get_document_knowledge_entry(self.db, document_id)
        if not document_entry or document_entry.project_id != project_id:
            logger.error(f"Document {document_id} not found or does not belong to project {project_id}.")
            raise ValueError("Document not found or invalid for this project.")

        text_chunks = crud.get_text_knowledge_entries_by_document_id(self.db, document_id)
        
        suggested_questions = []
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