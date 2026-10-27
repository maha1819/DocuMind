# DocuMind — AI-Powered Research & Document Intelligence

**A document-grounded question-answering system built with Retrieval-Augmented Generation (RAG).**

DocuMind lets users upload research papers in PDF format and ask questions about their contents. It retrieves relevant passages, generates answers grounded in the supplied document, provides page-level source references, and can abstain when the available evidence is insufficient.

The project focuses not only on generating answers, but also on evaluating retrieval quality, citation behavior, answer faithfulness, and abstention.

## Key Features

- **PDF ingestion:** Extracts text from uploaded research documents.
- **Semantic retrieval:** Uses sentence embeddings and FAISS vector search to find relevant passages.
- **RAG-based question answering:** Supplies retrieved document context to an LLM before generating an answer.
- **Page-level citations:** Connects answers to relevant source pages for verification.
- **Evidence-aware abstention:** Designed to avoid unsupported answers when the document does not contain sufficient evidence.
- **Retrieval benchmarking:** Compares similarity search and Maximal Marginal Relevance (MMR).
- **Answer evaluation:** Uses an LLM-based judge to assess faithfulness, relevance, correctness, and abstention behavior.
- **API and interactive interface:** Provides a FastAPI backend and a Streamlit user interface.
- **Automated tests:** Includes pytest tests for core RAG behavior.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| LLM orchestration | LangChain |
| LLM inference | Groq API |
| Embeddings | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Document processing | PyMuPDF |
| Backend API | FastAPI |
| User interface | Streamlit |
| Evaluation and testing | Custom Python evaluation scripts, LLM-as-a-judge, pytest |
| Configuration | Environment variables, python-dotenv |

## System Workflow

1. **Upload:** The user supplies a research-paper PDF.
2. **Extract:** Text and page information are extracted from the document.
3. **Chunk and embed:** Document text is split into smaller passages and converted into vector embeddings.
4. **Retrieve:** FAISS retrieves passages relevant to the user's question using the selected retrieval strategy.
5. **Generate:** The LLM receives the question and retrieved evidence to formulate a grounded response.
6. **Verify:** The response can be checked against source pages, while insufficient evidence should lead to abstention.
7. **Evaluate:** Benchmark scripts measure retrieval performance and selected answer-quality indicators.

## Evaluation Results

### 1. Retrieval benchmark

The retrieval benchmark uses a manually defined question set based on *Attention Is All You Need*. Expected source pages are used to assess whether relevant evidence appears among the retrieved results.

| Retrieval configuration | Expected-page recall | Mean reciprocal rank (MRR) |
|---|---:|---:|
| Similarity, k = 10 | 95.45% | 0.7394 |
| MMR, k = 10 | 100.00% | 0.7413 |

**Interpretation:** MMR retrieved at least one expected page for all 22 answerable benchmark questions in this evaluation. Similarity retrieval achieved 95.45% expected-page recall. MRR also considers how highly relevant pages rank in the results; the observed difference between the two configurations was small.

### 2. Answer-quality benchmark

The answer evaluation was run with similarity retrieval at k = 10 on a 30-case dataset containing 22 answerable and 8 unanswerable questions.

| Metric | Observed result |
|---|---:|
| Page hit rate | 95.45% |
| Expected-page citation accuracy | 90.91% |
| Mean expected-page recall | 95.45% |
| Mean faithfulness score | 4.90 / 5 |
| Mean answer relevance score | 4.93 / 5 |
| Mean answer correctness score | 4.86 / 5 |
| Abstention accuracy | 96.67% |
| Citation accuracy | 95.45% |

These scores are diagnostic rather than guarantees of production performance. One LLM-judge evaluation was unavailable, and the judge-generated scores should be interpreted alongside the saved answers, retrieved evidence, and citation details.

### Evaluation scope and limitations

- The benchmark is based on one research paper and a small, manually constructed question set.
- Expected-page metrics measure retrieval and citation alignment, not complete factual correctness.
- LLM-as-a-judge scores can be imperfect or overly generous and require manual review.
- The results do not establish general performance across different documents, domains, or question types.
- The system's abstention behavior and handling of partially answerable questions remain important areas for further testing.

## Project Structure

```text
documind/
├── app.py
├── streamlit_app.py
├── rag_engine.py
├── rag_pipeline.py
├── document_index.py
├── build_index.py
├── chunk_pdf.py
├── read_pdf.py
├── semantic_search.py
├── test_groq.py
├── tests/
│   └── test_rag_engine.py
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   └── eval_dataset_v2.json
├── .env.example
├── .gitignore
└── README.md
```

Generated indexes, uploaded documents, environment files, and evaluation outputs may be created locally and are not necessarily tracked in Git.

## Getting Started

### Prerequisites

- Python 3.11 or another Python version compatible with the installed dependencies
- A Groq API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/maha1819/documind.git
cd documind
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install fastapi uvicorn streamlit langchain langchain-community langchain-groq langchain-huggingface sentence-transformers faiss-cpu pymupdf python-dotenv python-multipart pytest
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure the required Groq API key and model settings using the variable names provided in `.env.example`.

**Security:** Never commit `.env` or expose your API key. Keep secrets out of source code and screenshots.

### 5. Launch the application

Start the Streamlit interface:

```powershell
streamlit run streamlit_app.py
```

To run the FastAPI backend in a separate terminal:

```powershell
uvicorn app:app --reload
```

The API's interactive documentation is available at `http://127.0.0.1:8000/docs`.

Follow the interface's upload flow to test the application with a PDF.

## Running Tests

Run the automated test suite from the project root:

```powershell
pytest -q
```

The current recorded test run passed **5 tests**. Run the suite again after making code changes to verify the current repository state.

## Reproducing the Evaluation

Run the retrieval benchmark:

```powershell
python evaluation/evaluate_retrieval.py --k 10
```

Run the answer-quality benchmark using similarity retrieval:

```powershell
python evaluation/evaluate_answers.py --dataset evaluation/eval_dataset_v2.json --k 10 --method similarity
```

Evaluation requires the configured LLM API and may consume API usage. Results are diagnostic, and exact scores can vary with model behavior, dependency versions, and code changes.

## Engineering Takeaways

This project explores practical RAG engineering beyond a basic chatbot:

- The relationship between retrieval quality and answer quality
- Similarity search versus diversity-aware MMR retrieval
- Source-page traceability and citation evaluation
- Faithfulness, answer relevance, and correctness assessment
- Abstention when evidence is missing or incomplete
- Automated testing and reproducible evaluation workflows

## Future Improvements

- Expand the evaluation dataset across multiple research papers and domains.
- Add stronger checks for partial answers and unsupported claims.
- Improve citation validation and retrieval ranking.
- Calibrate LLM-judge scores against human-reviewed examples.
- Evaluate latency, cost, and performance on larger document collections.

---

**Author:** Mahalakshmi

**Project focus:** Retrieval-Augmented Generation · LLM Applications · Semantic Search · AI Evaluation · FastAPI