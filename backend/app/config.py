import os
from typing import List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "sqlite:///./sql_app.db"
    VECTOR_DB_PATH: str = "./chroma_db"

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Configure which LLM provider to use (openai, openrouter, ollama)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").lower() 
    
    # Model names
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
    OPENROUTER_MODEL_NAME: str = os.getenv("OPENROUTER_MODEL_NAME") # Example Openrouter model
    OLLAMA_MODEL_NAME: str = "llama3.2" # Example Ollama model
    
    # Embedding model name (used by ChromaDB if not using OpenAI's embedding function)
    # EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # Sentence-transformers model
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")  # Default to a common sentence transformer model

    # CORS settings (add this for frontend integration)
    CORS_ORIGINS: List[str] = ["http://localhost", "http://localhost:3000"] # Example origins
    
    TEMP_DIR: str = "/tmp/knowledge_relay_uploads"
    

settings = Settings()