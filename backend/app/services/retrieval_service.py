import pprint
from sqlalchemy.orm import Session
from app import schemas, crud
from app.core.vector_store import ChromaDBManager
from app.services.llm_service import LLMService
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from typing import List, Tuple
import logging
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()
        self.document_qa_chain, self.contextualize_q_chain = self.llm_service.get_retrieval_qa_chain()
        self.parser = PydanticOutputParser(pydantic_object=schemas.AnswerWithSources)
        self.llm = self.llm_service.get_llm()


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
        
            logger.info(metadata)
            
            source_documents_info.append(schemas.SourceDocument(
                file_name=metadata.get("file_name", "Static Q&A" if metadata.get("type") == "static_qa" else "Unknown"),
                file_path=metadata.get("source"),
                question=metadata.get("question"),
                context=metadata.get("source_context"),
                document_id=metadata.get("document_id"),
                page_number=metadata.get("page_number")
            ))
            logger.info(metadata)
            
        if not retrieved_langchain_docs:
            logger.warning(f"No relevant documents found for query: '{query}' in project {project_id}.")
            return schemas.ChatResponse(
                project_id=project_id,
                answer="I couldn't find relevant information in the knowledge base for that query.",
                source_documents=[]
            )
        
        # Define the prompt for the LLM
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI assistant tasked with answering questions based on provided documents."),
                ("system", "Generate a comprehensive and well-structured answer using Markdown. Use headings, bullet points, code snippets (if relevant), and bold text to enhance readability."),
                ("system", "Carefully read the following context documents to formulate your answer:\n\n{context}\n\n"),
                ("system", "After providing the answer, list the exact sources you used to construct your response. For each source, include the file name, and optionally its URL/path, and the specific context snippet that was most relevant. Ensure the output strictly adheres to the JSON schema provided for the `AnswerWithSources` Pydantic model."),
                ("system", "Chat History:\n{chat_history}"),
                ("human", "{input}")
            ]
        )
        
        
        # Create the structured output runnable
        # It automatically injects the Pydantic schema and handles parsing.
        # structured_output_chain = self.llm.with_structured_output(schemas.AnswerWithSources, self.llm, prompt)
        structured_output_chain = (
            {
                "context": RunnablePassthrough(), 
                "input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "format_instructions": lambda _: self.parser.get_format_instructions() 
            }
            | prompt
            | self.llm.with_structured_output(schemas.AnswerWithSources) # <--- THIS IS THE CORRECT USAGE
        )
        
        # 4. Invoke the Chain
        # The 'context' should be a list of LangChain Document objects.
        # The 'chat_history' should be in LangChain message format.
        final_structured_response: schemas.AnswerWithSources = structured_output_chain.invoke({
            "context": retrieved_langchain_docs, # Pass LangChain Document objects directly
            "input": query,
            "chat_history": lc_chat_history # Use the formatted LangChain chat history
        })

        # Format the final answer and sources for your API response
        generated_answer_markdown = final_structured_response.answer
        
        # Append sources in Markdown if you want them embedded in the answer field
        # if final_structured_response.sources:
        #     generated_answer_markdown += "\n\n---\n\n**Sources:**\n"
        #     for src in final_structured_response.sources:
        #         source_line = f"- **File:** {src.file_name}"
        #         if src.file_path:
        #             source_line += f" ([Link]({src.file_path}))"
        #         if src.page_number:
        #             source_line += f", **Page:** {src.page_number}"
        #         if src.context:
        #             # You might want to truncate context if it's too long
        #             source_line += f"\n  *Context:* \"{src.context}\""
        #         generated_answer_markdown += f"{source_line}\n"


        logger.info(f"Generated answer for query: '{query}' in project {project_id}.")


        # final_answer_response = self.document_qa_chain.invoke({
        #     "context": retrieved_langchain_docs, 
        #     "input": query,
        #     "chat_history": lc_chat_history
        # })
                
        logger.info(f"Generated answer for query: '{query}' in project {project_id}.")
        return schemas.ChatResponse(
            project_id=project_id,
            answer=generated_answer_markdown,
            source_documents=final_structured_response.sources
        )
        
