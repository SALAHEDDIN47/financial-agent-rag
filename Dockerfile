# Dockerfile
FROM python:3.12-slim

# Métadonnées
LABEL maintainer="Financial Agent RAG"
LABEL description="RAG Agent — API + UI"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer uv (gestionnaire rapide)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copier UNIQUEMENT les fichiers de dépendances d'abord (cache Docker)
COPY pyproject.toml uv.lock ./

# Installer les dépendances (sans le projet lui-même)
RUN uv sync --frozen --no-install-project --no-dev

# Copier le code source
COPY src/ ./src/
COPY README.md ./

# Installer le projet
RUN uv sync --frozen --no-dev

# Exposer les ports
EXPOSE 8000 8501

# Point d'entrée par défaut (sera surchargé par docker-compose)
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]