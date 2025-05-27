import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "sqlite:///./sql_app.db"
    VECTOR_DB_PATH: str = "./chroma_db"

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Configure which LLM provider to use (openai, openrouter, ollama)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower() 
    
    # Model names
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
    OPENROUTER_MODEL_NAME: str = "mistralai/mistral-7b-instruct-v0.2" # Example Openrouter model
    OLLAMA_MODEL_NAME: str = "llama3.2" # Example Ollama model
    
    # Embedding model name (used by ChromaDB if not using OpenAI's embedding function)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # Sentence-transformers model

settings = Settings()