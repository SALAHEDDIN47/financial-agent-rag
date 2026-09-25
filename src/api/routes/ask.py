# src/api/routes/ask.py
import logging
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from src.rag.agent import ask_financial_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
router = APIRouter()


# ==============================================================================
# MODÈLES PYDANTIC
# ==============================================================================
class AskRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20, description="Nombre de chunks finaux à passer au LLM")
    company: Optional[str] = None
    period: Optional[str] = None
    document_type: Optional[str] = None
    force_simple: bool = False


class Source(BaseModel):
    rank: int
    document: str          # "AAPL - FY2025"
    chunk_id: str          # identifiant unique
    document_type: str
    page: Optional[int] = None
    is_table: bool = False
    snippet: str
    score: Optional[float] = None  # rerank_score


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    n_sources: int
    elapsed_s: float


# ==============================================================================
# ENDPOINT
# ==============================================================================
@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    t0 = time.time()
    logger.info(f"📩 Question : '{request.query}' (top_k={request.top_k})")

    try:
        # Filtres
        filters = {}
        if request.company:
            filters["company"] = request.company.upper()
        if request.period:
            filters["period"] = request.period
        if request.document_type:
            filters["document_type"] = request.document_type
        filters = filters or None

        # Appel de l'agent (retourne tuple)
        answer, docs = ask_financial_agent(
            question=request.query,
            top_k=request.top_k,
            filters=filters,
            force_simple=request.force_simple,
        )

        # ✅ FIX : on conserve TOUTES les métadonnées utiles
        sources = []
        for rank, doc in enumerate(docs, 1):
            company = doc.get("company", "Unknown")
            period = doc.get("period", "Unknown")
            sources.append(
                Source(
                    rank=rank,
                    document=f"{company} - {period}",
                    chunk_id=doc.get("chunk_id", ""),
                    document_type=doc.get("document_type", "Unknown"),
                    page=doc.get("page"),
                    is_table=bool(doc.get("metadata", {}).get("is_table", False)),
                    snippet=doc.get("text", "")[:500],   # 500 au lieu de 300
                    score=doc.get("rerank_score"),
                )
            )

        elapsed = time.time() - t0
        logger.info(f"✅ Réponse générée en {elapsed:.2f}s ({len(sources)} sources)")

        return AskResponse(
            answer=answer,
            sources=sources,
            n_sources=len(sources),
            elapsed_s=round(elapsed, 2),
        )

    except Exception as e:
        logger.error(f"❌ Erreur : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")