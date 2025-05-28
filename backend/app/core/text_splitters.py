
from pathlib import Path
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

from .document_loaders import CODE_EXTENSIONS


logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

    

def split_documents(documents: List[Document]) -> List[Document]:
    all_chunks = []

    for doc in documents:
        ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
        is_code = ext in CODE_EXTENSIONS

        if is_code:
            # Use special code splitter settings
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=_get_lang(ext),
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        else:
            # Use default splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n## ", "###", "##","\n\n", "\n", ".", " "]
            )

        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    pprint.pprint(all_chunks)
    
    return all_chunks

def _get_lang(ext: str):
    from langchain.text_splitter import Language

    mapping = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".java": Language.JAVA,
        ".cpp": Language.CPP,
        ".c": Language.CPP,
        ".cs": Language.CSHARP,
        ".go": Language.GO,
        ".php": Language.PHP,
        ".rb": Language.RUBY,
        ".rs": Language.RUST,
    }
    return mapping.get(ext, Language.PYTHON)  # Default fallback
