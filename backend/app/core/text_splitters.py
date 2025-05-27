
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import logging
from langchain.text_splitter import MarkdownHeaderTextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader


logger = logging.getLogger(__name__)

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
    return split_docs


def load_and_chunk_file(path: Path) -> List[Document]:
        ext = path.suffix.lower()

        if ext == ".md":
            text = path.read_text(encoding="utf-8")
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=ext[1:],
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(text)
        elif ext == ".txt":
            text = path.read_text(encoding="utf-8")
            splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(text)
        elif ext == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(text)
        elif ext in [".py", ".java", ".js", ".ts", ".cpp", ".c", ".go", ".rs"]:
            text = path.read_text(encoding="utf-8")
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=ext[1:],
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(text)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return chunks
    