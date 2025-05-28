from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
import logging
from typing import List
from langchain.schema import Document
import os

CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs", ".rb", ".go", ".php", ".rs"}



logger = logging.getLogger(__name__)

# def load_document(file_path: str, file_type: str) -> List[Document]:
#     try:
#         if file_type == "pdf":
#             loader = PyPDFLoader(file_path)
#         elif file_type == "word":
#             loader = UnstructuredWordDocumentLoader(file_path)
#         elif file_type == "md":
#             loader = UnstructuredMarkdownLoader(file_path)
#         elif file_type in ["txt"]:
#             loader = TextLoader(file_path)
#         else:
#             raise ValueError(f"Unsupported file type: {file_type}")
        
#         docs = loader.load()
#         logger.info(f"Loaded {len(docs)} pages/chunks from {file_path} (type: {file_type}).")
#         return docs
#     except Exception as e:
#         logger.error(f"Error loading document {file_path}: {e}")
#         raise

def load_document(file_path: str, file_type: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in {".docx", ".doc", ".ppt", ".pptx"}:
        loader = UnstructuredFileLoader(file_path)
    elif ext in {".txt", ".csv", ".md"} or ext in CODE_EXTENSIONS:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return loader.load()
