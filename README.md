

## **KnowledgeRelay: AI-based Knowledge Transfer Agent**

A Retrieval-Augmented Generation (RAG) system that captures and delivers project-specific knowledge. The agent facilitates interactive knowledge transfer from outgoing members and provides project-specific answers to incoming team members with source attribution.

### Knowledge Ingestion

* Designed a multi-modal ingestion pipeline using ChromaDB for vector storage with metadata filtering.
* Built two-stage QA flow:

  * **Project QA**: LLM generates adaptive questions based on chat history to extract high-level project insights.
  * **Document QA**: LLM formulates questions from uploaded documents (code, specs, etc.), with user responses enriching the vector store.
* Preprocessing pipeline includes:

  * Context-aware text splitting with overlapping chunks.
  * Custom chunking strategies for code vs. prose.
  * Cleaning pipeline: emoji removal, non-ASCII filtering, newline normalization.


### Knowledge Retrieval

* Implemented a RAG-based query engine to retrieve relevant chunks with metadata.
* Used `contextualize_qa_chain` to reformulate user questions using prior conversation context for improved precision.
* LLM responses are grounded in retrieved sources, with file and context references included.

---

* **Technologies:** Python, LangChain, FastAPI, ChromaDB, LLMs (OpenAI, Ollama), React.
