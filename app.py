
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from document_index import build_vector_store
from rag_engine import answer_question

app = FastAPI(
    title="DocuMind API",
    description="Ask questions about uploaded research papers using RAG.",
    version="1.0.0",
)

# Keep each uploaded document's index separate while the server runs.
document_stores = {}


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {
        "message": "Welcome to DocuMind API",
        "docs": "/docs",
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    document_id = str(uuid4())
    pdf_path = upload_dir / f"{document_id}.pdf"

    try:
        pdf_path.write_bytes(content)

        vector_store = build_vector_store(pdf_path)
        document_stores[document_id] = vector_store

        return {
            "message": "PDF processed successfully.",
            "document_id": document_id,
            "filename": file.filename,
        }

    except (ValueError, FileNotFoundError) as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="The PDF could not be processed. Please try another file.",
        ) from exc


@app.post("/documents/{document_id}/ask")
def ask_question(document_id: str, request: QuestionRequest):
    vector_store = document_stores.get(document_id)

    if vector_store is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Please upload the PDF again.",
        )

    try:
        return answer_question(vector_store, request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
