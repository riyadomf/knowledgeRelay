from sqlalchemy.orm import Session
from app import schemas, crud
from app.core.vector_store import ChromaDBManager
from app.services.llm_service import LLMService
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()
        self.document_qa_chain, self.contextualize_q_chain = self.llm_service.get_retrieval_qa_chain()

    def _format_chat_history(self, chat_history: List[schemas.ChatMessage]) -> List[BaseMessage]:
        """Converts Pydantic ChatMessage to LangChain BaseMessage format."""
        lc_chat_history = []
        for msg in chat_history:
            if msg.role == schemas.ChatRole.HUMAN:
                lc_chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == schemas.ChatRole.AI:
                lc_chat_history.append(AIMessage(content=msg.content))
        return lc_chat_history

    def answer_query(self, project_id: str, query: str, chat_history: List[schemas.ChatMessage]) -> schemas.ChatResponse:
        project = crud.get_project(self.db, project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found for query.")
            raise ValueError(f"Project with ID {project_id} not found.")

        chroma_manager = ChromaDBManager(project_id)
        lc_chat_history = self._format_chat_history(chat_history)

        if lc_chat_history:
            logger.info(f"Contextualizing query for project {project_id} with chat history.")
            standalone_query = self.contextualize_q_chain.invoke({
                "chat_history": lc_chat_history,
                "input": query
            })
            logger.info(f"Standalone query: {standalone_query}")
        else:
            standalone_query = query
            logger.info(f"No chat history, using original query: {standalone_query}")

        retrieved_docs_content, retrieved_metadatas = chroma_manager.query_documents([standalone_query])
        
        retrieved_langchain_docs = []
        source_documents_info = []
        for i, doc_content in enumerate(retrieved_docs_content):
            metadata = retrieved_metadatas[i]
            lc_doc = Document(page_content=doc_content, metadata=metadata)
            retrieved_langchain_docs.append(lc_doc)
        

            source_documents_info.append(schemas.SourceDocument(
                file_name=metadata.get("file_name", "Static Q&A" if metadata.get("type") == "static_qa" else "Unknown"),
                file_path=metadata.get("source"),
                document_id=metadata.get("document_id"),
                page_number=metadata.get("page_number")
            ))
        
        if not retrieved_langchain_docs:
            logger.warning(f"No relevant documents found for query: '{query}' in project {project_id}.")
            return schemas.ChatResponse(
                project_id=project_id,
                answer="I couldn't find relevant information in the knowledge base for that query.",
                source_documents=[]
            )

        final_answer_response = self.document_qa_chain.invoke({
            "context": retrieved_langchain_docs, 
            "input": query,
            "chat_history": lc_chat_history
        })
        
        logger.info(f"Generated answer for query: '{query}' in project {project_id}.")
        return schemas.ChatResponse(
            project_id=project_id,
            answer=final_answer_response,
            source_documents=source_documents_info
        )
        
