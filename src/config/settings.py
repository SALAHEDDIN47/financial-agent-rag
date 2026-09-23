# src/config/settings.py
"""
Configuration centralisée du projet.

Toutes les variables d'environnement sont déclarées ici.
Pydantic-Settings les charge depuis `.env` et valide les types.

Pour ajouter une variable :
  1. Ajouter le champ dans Settings (avec type + valeur par défaut si optionnel)
  2. Ajouter la clé dans `.env` (si non-default)
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -------------------------------------------------------------------------
    # MINIO (stockage objet S3-compatible)
    # -------------------------------------------------------------------------
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "admin"
    minio_secret_key: str = "password123"
    minio_bucket: str = "financial-documents"

    # -------------------------------------------------------------------------
    # MILVUS (base vectorielle)
    # -------------------------------------------------------------------------
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # -------------------------------------------------------------------------
    # ELASTICSEARCH (BM25)
    # -------------------------------------------------------------------------
    elasticsearch_host: str = "http://localhost:9200"
    elasticsearch_index: str = "financial_chunks_bm25"

    # -------------------------------------------------------------------------
    # LLM / MODELS
    # -------------------------------------------------------------------------
    gemini_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None

    # Modèles par défaut (peuvent être surchargés via .env)
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    llm_model: str = "qwen2.5:7b"      # ou "gemini-pro" si API cloud

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # -------------------------------------------------------------------------
    # RAG
    # -------------------------------------------------------------------------
    top_k_vector: int = 20       # Nombre de candidats Milvus
    top_k_bm25: int = 20         # Nombre de candidats ES
    top_k_final: int = 5         # Nombre de chunks passés au LLM

    # -------------------------------------------------------------------------
    # CONFIG PYDANTIC
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # ✅ tolère les variables du .env non déclarées
    )


# Singleton global
settings = Settings()