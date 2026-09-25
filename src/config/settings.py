# src/config/settings.py
"""
Configuration centralisée du projet.

Pydantic-Settings charge les valeurs depuis (ordre de priorité) :
  1. Variables d'environnement Docker (ex: MILVUS_HOST=milvus)
  2. Fichier .env (développement local)
  3. Valeurs par défaut ci-dessous

Le nom de la variable d'env est insensible à la casse :
  MILVUS_HOST → milvus_host
  OLLAMA_BASE_URL → ollama_base_url
  FORCE_DEVICE → force_device
"""
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
    milvus_collection: str = "financial_chunks"

    # -------------------------------------------------------------------------
    # ELASTICSEARCH (BM25)
    # -------------------------------------------------------------------------
    elasticsearch_host: str = "http://localhost:9200"
    elasticsearch_index: str = "financial_chunks_bm25"

    # -------------------------------------------------------------------------
    # LLM / OLLAMA
    # -------------------------------------------------------------------------
    # ⚠️ Dans Docker : OLLAMA_BASE_URL=http://host.docker.internal:11434
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"

    gemini_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None

    # -------------------------------------------------------------------------
    # EMBEDDING / RERANKER
    # -------------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-base"

    # ⚠️ Dans Docker GPU : FORCE_DEVICE=cuda
    #    Dans Docker CPU : FORCE_DEVICE=cpu
    #    En local : laisser vide → auto-détection
    force_device: Optional[str] = None

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # -------------------------------------------------------------------------
    # RAG
    # -------------------------------------------------------------------------
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5

    # -------------------------------------------------------------------------
    # CONFIG PYDANTIC
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # tolère les variables non déclarées
    )


# Singleton global — importé partout
settings = Settings()