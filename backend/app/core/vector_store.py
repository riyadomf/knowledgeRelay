import chromadb
from chromadb.utils import embedding_functions
from app.config import settings
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ChromaDBManager:
    def __init__(self, project_id: str):
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        self.collection_name = f"project_{project_id}"
        
        self.embedding_function = self._initialize_embedding_function()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        logger.info(f"Initialized ChromaDB collection '{self.collection_name}' with embedding function: {self.embedding_function.__class__.__name__}")

    def _initialize_embedding_function(self):
        if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            logger.info("Using OpenAIEmbeddingFunction for ChromaDB.")
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
        else:
            # Fallback for hackathon simplicity: use a pre-trained Sentence Transformer model
            # This requires 'sentence-transformers' package to be installed.
            try:
                from chromadb.utils import embedding_functions
                logger.info(f"Using SentenceTransformerEmbeddingFunction ('{settings.EMBEDDING_MODEL_NAME}') for ChromaDB.")
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=settings.EMBEDDING_MODEL_NAME
                )
            except ImportError:
                logger.warning("Warning: 'sentence-transformers' not found. Using default ChromaDB embedding. "
                               "Install it for better embeddings if not using OpenAI.")
                # If sentence-transformers is not installed, ChromaDB will use its default in-memory embedding function
                # which is less robust but works for basic tests.
                return None # ChromaDB will use its default if None provided explicitly.
            except Exception as e:
                logger.error(f"Error initializing SentenceTransformerEmbeddingFunction: {e}. Falling back to default ChromaDB embedding.")
                return None

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def query_documents(self, query_texts: List[str], n_results: int = 5) -> Tuple[List[str], List[Dict]]:
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                include=['documents', 'metadatas']
            )
            return results['documents'][0], results['metadatas'][0] # Return first query's results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return [], []
    
    def delete_collection(self):
        """Deletes the entire collection for the project."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"ChromaDB collection '{self.collection_name}' deleted.")
        except Exception as e:
            logger.error(f"Error deleting ChromaDB collection '{self.collection_name}': {e}")
            raise

