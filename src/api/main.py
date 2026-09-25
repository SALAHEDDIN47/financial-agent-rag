# src/api/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ask

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Agent RAG API",
    description="API pour interroger un agent d'analyse financière (RAG hybride Milvus + Elasticsearch)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ✅ FIX 1 : CORS — indispensable pour que l'UI Streamlit puisse appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En prod : ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(ask.router, prefix="/api", tags=["Agent"])


# ✅ FIX 2 : /health pour le monitoring
@app.get("/health", tags=["Monitoring"])
async def health():
    return {"status": "ok", "service": "financial-agent-rag"}


@app.get("/", tags=["Monitoring"])
async def root():
    return {
        "service": "Financial Agent RAG",
        "endpoints": {
            "ask": "POST /api/ask",
            "docs": "GET /docs",
            "health": "GET /health",
        },
    }