# src/api/routes/ask.py
import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.rag.agent import ask_financial_agent, search_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    company: Optional[str] = None
    period: Optional[str] = None
    document_type: Optional[str] = None
    force_simple: bool = False  # Pour forcer le mode simple (débogage)


class Source(BaseModel):
    document: str
    chunk_info: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        logger.info(f"📩 Requête reçue : {request.query}")

        # Construction des filtres
        filters = {}
        if request.company:
            filters["company"] = request.company
        if request.period:
            filters["period"] = request.period
        if request.document_type:
            filters["document_type"] = request.document_type
        filters = filters or None

        # Appel à l'agent (gère automatiquement simple/complexe)
        answer, docs = ask_financial_agent(request.query, top_k=request.top_k, filters=filters, force_simple=request.force_simple)
        sources = []
        for doc in docs:
            company = doc.get("company", "Inconnue")
            period = doc.get("period", "Inconnue")
            doc_type = doc.get("document_type", "Inconnu")
            text = doc.get("text", "")
            sources.append(Source(
                document=f"{company} - {period}",
                chunk_info=f"Type: {doc_type}",
                snippet=text[:300] + "...",
            ))

        return AskResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")