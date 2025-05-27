from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from typing import List, Tuple, Union, Dict
import logging

from app.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if settings.LLM_PROVIDER == "openai":
            logger.info(f"Initializing ChatOpenAI with model: {settings.OPENAI_MODEL_NAME}")
            return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL_NAME, temperature=0.7)
        elif settings.LLM_PROVIDER == "openrouter":
            logger.info(f"Initializing OpenRouter with model: {settings.OPENROUTER_MODEL_NAME}")
            return ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY, 
                model_name=settings.OPENROUTER_MODEL_NAME,
                temperature=0.7
            )
        elif settings.LLM_PROVIDER == "ollama":
            logger.info(f"Initializing ChatOllama with model: {settings.OLLAMA_MODEL_NAME} at {settings.OLLAMA_BASE_URL}")
            return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME, temperature=0.7)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

    def generate_static_qa_question(self, existing_qa: List[Dict[str, str]]) -> str:
        """
        Generates a question for the old member for static Q&A.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an AI assistant helping an experienced developer transfer project knowledge. "
             "Your goal is to ask insightful, open-ended questions to extract critical information. "
             "Focus on areas like project purpose, architecture, key technologies, deployment, common issues, "
             "team practices, and important contacts. Avoid asking questions that have already been covered. "
             "If you think enough information has been gathered, you can suggest concluding the session. "
             "Ask one question at a time."),
            ("human", "Here's what we've covered so far:\n{existing_qa_summary}\n\nWhat's the next important question to ask about this project?")
        ])
        
        qa_summary = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in existing_qa]) if existing_qa else "No questions asked yet."
        
        chain = prompt_template | self.llm
        response = chain.invoke({"existing_qa_summary": qa_summary})
        return response.content

    def generate_questions_from_document_chunk(self, chunk_content: str, num_questions: int = 3) -> List[str]:
        """
        Generates a specified number of questions based on a specific document chunk.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an AI assistant helping to extract knowledge from a document. "
             "Read the following text chunk and identify key pieces of information "
             f"that an incoming team member might need to know. Formulate these as clear, "
             f"concise questions (aim for {num_questions} questions). "
             "Do not answer the questions yourself. Just provide the questions, one per line."),
            ("human", "Text chunk:\n{chunk_content}\n\nWhat are important questions based on this text?")
        ])
        
        chain = prompt_template | self.llm
        response = chain.invoke({"chunk_content": chunk_content})
        # Parse response into a list of questions (assuming one per line)
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        return questions[:num_questions] # Return up to num_questions


    def get_retrieval_qa_chain(self) -> RunnablePassthrough:
        """
        Creates a retrieval-augmented generation (RAG) chain for new member Q&A.
        Includes chat history for context.
        """
        # Contextualize question if there's chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood without "
            "the chat history. Do NOT answer the question, just reformulate it "
            "if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        # This chain takes {"input": query, "chat_history": lc_chat_history} and returns a standalone question string
        contextualize_q_chain = contextualize_q_prompt | self.llm | RunnableLambda(lambda x: x.content)

        # Main QA prompt using retrieved context
        qa_system_prompt = (
            "You are an AI assistant for project knowledge transfer. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Provide the source document's file name and specific context (the relevant snippet) "
            "where the answer was found. Format sources clearly, e.g., 'Source: [File: example.pdf, Context: ...]' "
            "If multiple sources, list them all."
            "---"
            "Context: {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # This chain will combine documents and generate the final answer
        document_qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        return document_qa_chain, contextualize_q_chain