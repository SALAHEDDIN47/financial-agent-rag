# tests/test_vector_search.py
"""
Test de recherche vectorielle sur Milvus.

Usage :
    uv run ./scripts/test_vector_search.py
    uv run ./scripts/test_vector_search.py --query "Tesla revenue 2023"
    uv run ./scripts/test_vector_search.py --query "risk factors Apple" --company AAPL --top-k 5
"""
import argparse
import logging
import sys
from pathlib import Path

from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "financial_chunks"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
# ⚠️  Préfixe OBLIGATOIRE pour BGE en mode requête (ne PAS l'utiliser côté corpus)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(
    query: str,
    top_k: int = 5,
    company: str = None,
    document_type: str = None,
):
    """Recherche vectorielle avec filtres optionnels."""
    # 1. Connexion Milvus
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 2. Encoder la requête (avec préfixe BGE)
    logger.info(f"🧠 Encodage de la requête...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    query_with_prefix = BGE_QUERY_PREFIX + query
    query_vector = model.encode(
        [query_with_prefix], normalize_embeddings=True
    )[0].tolist()

    # 3. Construire le filtre Milvus (expression booléenne)
    filters = []
    if company:
        filters.append(f'company == "{company.upper()}"')
    if document_type:
        filters.append(f'document_type == "{document_type}"')
    expr = " and ".join(filters) if filters else None

    # 4. Recherche
    logger.info(f"🔍 Recherche top-{top_k}{' | filter: ' + expr if expr else ''}")
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        expr=expr,
        output_fields=[
            "chunk_id", "company", "period", "document_type",
            "source", "page", "is_table", "text",
        ],
    )

    # 5. Affichage
    print(f"\n{'=' * 70}")
    print(f"  QUERY : {query}")
    if company:
        print(f"  FILTER: company={company.upper()}")
    print(f"{'=' * 70}\n")

    for i, hit in enumerate(results[0], 1):
        entity = hit.entity
        page = entity.get("page")
        page_str = f"p.{page}" if page and page > 0 else "N/A"
        is_table = "📊" if entity.get("is_table") else "📄"

        print(f"─── Résultat {i} ─── (score: {hit.score:.4f}) {is_table}")
        print(f"  Company       : {entity.get('company')}")
        print(f"  Period        : {entity.get('period')}")
        print(f"  Doc type      : {entity.get('document_type')}")
        print(f"  Source        : {entity.get('source')} | {page_str}")
        print(f"  Chunk ID      : {entity.get('chunk_id')}")
        print(f"  Texte (300c)  :")
        text = entity.get("text", "")[:300]
        print(f"    {text}...")
        print()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default="Tesla revenue 2023")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--company", type=str, default=None,
                    help="Filtre par ticker (AAPL, TSLA, MSFT, GOOGL, ...)")
    ap.add_argument("--document-type", type=str, default=None,
                    help="Filtre par type (10-K, 10-Q, earnings_transcript, ...)")
    args = ap.parse_args()

    search(
        query=args.query,
        top_k=args.top_k,
        company=args.company,
        document_type=args.document_type,
    )