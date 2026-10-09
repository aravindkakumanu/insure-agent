"""
main.py — FastAPI Application Entry Point
==========================================
This is where HTTP meets AI. FastAPI handles incoming web requests and
routes them to the appropriate LangChain logic.

HOW TO RUN:
    source .venv/bin/activate
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.

ENDPOINTS:
    POST /ask       — Ask a question about loaded policies
    POST /ingest    — Load a PDF into ChromaDB
    GET  /health    — Simple health check
"""

import logging
from pathlib import Path

from dotenv import load_dotenv  # loads .env file into os.environ BEFORE anything else
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load .env FIRST — so OPENAI_API_KEY is available when LangChain initialises
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="InsureAgent",
    description="Multi-agent insurance policy Q&A powered by LangChain + OpenAI",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas  (Pydantic validates these automatically)
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Your insurance policy question")


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestRequest(BaseModel):
    pdf_path: str = Field(..., description="Path to a PDF in data/policies/")


class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Utility"])
async def health_check() -> dict:
    """Quick liveness probe — returns 200 OK if the server is running."""
    return {"status": "ok", "service": "InsureAgent"}


@app.post("/ask", response_model=QuestionResponse, tags=["Q&A"])
async def ask_question(body: QuestionRequest) -> QuestionResponse:
    """
    Ask a question about any ingested insurance policy.

    The system retrieves relevant policy excerpts and uses GPT-4o to compose
    a grounded, accurate answer.
    """
    from src.agents.qa_agent import answer_question  # deferred import = faster startup

    logger.info("Received question: %r", body.question)
    try:
        result = answer_question(body.question)
        return QuestionResponse(answer=result["answer"], sources=result["sources"])
    except Exception as exc:
        logger.exception("Error answering question")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(body: IngestRequest) -> IngestResponse:
    """
    Load a PDF file from disk, chunk it, embed it, and store it in ChromaDB.

    After ingesting, you can ask questions about its content via /ask.
    """
    from src.rag.embedder import ingest_documents
    from src.rag.loader import load_pdf

    pdf_path = Path(body.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

    try:
        chunks = load_pdf(pdf_path)
        count = ingest_documents(chunks)
        return IngestResponse(
            message=f"Successfully ingested {pdf_path.name}",
            chunks_ingested=count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error ingesting PDF")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
