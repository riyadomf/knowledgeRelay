**Problem:** When team members leave a project, critical, unwritten knowledge is lost. This slows down onboarding for new members, who struggle to find specific answers about deployment processes, business logic, or common pitfalls.

**Solution:** An AI agent using a Retrieval-Augmented Generation (RAG) architecture. The system will:

* Capture Knowledge: Interactively prompt outgoing members with adaptive questions to build a project-specific knowledge base from their expertise and existing documents (PDFs, Word).
* Deliver Knowledge: Allow new members to ask natural language questions (e.g., "How do we handle token expiration?") and receive concise, contextual answers with source attribution, directly from the curated knowledge base.

This addresses the loss of institutional memory, reduces onboarding time, and prevents new team members from repeating past mistakes.


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
