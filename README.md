# DocuMind — RAG Research Paper Assistant

DocuMind is a Retrieval-Augmented Generation (RAG) application that lets users upload research papers in PDF format and ask questions about their contents. It retrieves relevant passages and generates answers grounded in the document, with source-page information to help users verify responses.

## Features

- Upload research-paper PDFs through a web interface.
- Extract text and split documents into overlapping chunks.
- Generate local text embeddings using Sentence Transformers.
- Retrieve relevant passages using FAISS vector search.
- Generate context-grounded answers using a Groq-hosted language model.
- Return source document names, page numbers, and retrieved passages.
- Abstain when the retrieved context does not provide enough information.
- Expose document upload and question-answering endpoints through FastAPI.
- Provide a Streamlit interface for interacting with uploaded papers.
- Include automated unit tests for core RAG behavior.

## Architecture

```text
PDF Upload
    |
    v
PyMuPDF Text Extraction
    |
    v
Text Chunking
    |
    v
Sentence Transformers Embeddings
    |
    v
FAISS Vector Store
    |
    v
Question -> Relevant Passage Retrieval
    |
    v
Groq LLM -> Answer with Source References
    |
    v
Streamlit User Interface
```

FastAPI provides the backend endpoints used by the Streamlit frontend.

## Technology Stack

- **Language:** Python
- **LLM orchestration:** LangChain
- **Language model:** Groq API
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector search:** FAISS
- **PDF processing:** PyMuPDF
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Validation and testing:** Pydantic, pytest

## Prerequisites

- Python 3.12 recommended
- A Groq API key
- Git

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd documind
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL after creating it.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages used by the project:

```powershell
python -m pip install pymupdf sentence-transformers langchain-core langchain-community langchain-groq faiss-cpu python-dotenv fastapi uvicorn python-multipart streamlit requests pytest
```

### 4. Configure environment variables

Create a local `.env` file using `.env.example` as a template:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit `.env` or expose your API key.

### 5. Start the backend

```powershell
python -m uvicorn app:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

### 6. Start the frontend

Open a second terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Testing

Run the automated unit tests:

```powershell
python -m pytest -v
```

The current core RAG test suite has five passing tests covering input validation, source metadata, retrieval settings, and abstention when no passages are retrieved.

## Current Limitations

- Document indexes are held in server memory and must be recreated after a backend restart.
- The current retrieval pipeline uses similarity search; retrieval quality needs systematic evaluation.
- The abstention behavior has been checked with sample questions but is not yet a guarantee against hallucinations.
- Production deployment, upload limits, persistent storage, and broader security testing remain future improvements.

## Planned Improvements

- Build a reproducible RAG evaluation dataset.
- Measure retrieval quality, answer relevance, faithfulness, and abstention behavior.
- Compare retrieval configurations and document the results.
- Expand automated tests and error handling.
- Add Docker support and deployment documentation.

## Project Goal

This project demonstrates an end-to-end RAG workflow, including document ingestion, vector retrieval, LLM integration, API development, a user interface, and automated testing.
