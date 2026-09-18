from fastapi import FastAPI
from src.api.routes import ask # Assurez-vous que ce fichier existe

app = FastAPI(title="Financial Agent RAG API")

app.include_router(ask.router, prefix="/api", tags=["Agent"])
