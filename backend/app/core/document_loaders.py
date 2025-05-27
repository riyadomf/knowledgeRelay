from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

def load_document(file_path: str, file_type: str) -> List[Document]:
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "word":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_type == "md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_type in ["txt"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages/chunks from {file_path} (type: {file_type}).")
        return docs
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        raise

