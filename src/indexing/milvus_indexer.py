# src/indexing/milvus_indexer.py
"""
Indexation des chunks embeddés dans Milvus (S3-aware).

Lit   : embeddings/{company}/{doc_type}/{source}_chunks.jsonl
Écrit : collection Milvus 'financial_chunks'
Manifest : manifests/milvus_manifest.json
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.core.storage import compute_key_hash, storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================
DEFAULT_COLLECTION = "financial_chunks"
DEFAULT_DIM = 1024
MANIFEST_KEY = "manifests/milvus_manifest.json"
INPUT_PREFIX = "embeddings/"


# ==============================================================================
# MANIFEST (dans MinIO)
# ==============================================================================
def _load_manifest() -> dict:
    if storage.exists(MANIFEST_KEY):
        try:
            return storage.read_json(MANIFEST_KEY)
        except Exception as e:
            logger.warning(f"⚠️  Manifest corrompu ({e}), redémarrage à vide")
            return {}
    return {}


def _save_manifest(manifest: dict) -> None:
    storage.write_json(MANIFEST_KEY, manifest)


# ==============================================================================
# CONNEXION MILVUS
# ==============================================================================
def connect_milvus(retries: int = 3):
    from pymilvus import connections

    host = settings.milvus_host
    port = str(settings.milvus_port)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            connections.connect(alias="default", host=host, port=port, timeout=10)
            logger.info(f"✅ Connecté à Milvus ({host}:{port})")
            return
        except Exception as e:
            last_error = e
            logger.warning(f"⚠️  Tentative {attempt}/{retries} échouée : {e}")
            if attempt < retries:
                time.sleep(2)

    raise ConnectionError(f"Impossible de se connecter à Milvus : {last_error}")


# ==============================================================================
# SCHÉMA / INDEX (identique à la version disque)
# ==============================================================================
def setup_collection(
    collection_name: str = DEFAULT_COLLECTION,
    dim: int = DEFAULT_DIM,
    reset: bool = False,
):
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
    )

    exists = utility.has_collection(collection_name)

    if exists and reset:
        logger.warning(f"⚠️  --reset : suppression de '{collection_name}'")
        utility.drop_collection(collection_name)
        exists = False

    if exists:
        logger.info(f"ℹ️  Collection '{collection_name}' existe déjà (conservée)")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
        FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="period", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="page", dtype=DataType.INT32),
        FieldSchema(name="is_table", dtype=DataType.BOOL),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Chunks financiers RAG")
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"✅ Collection '{collection_name}' créée (dim={dim})")

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("✅ Index vectoriel HNSW créé")

    return collection


# ==============================================================================
# INGESTION
# ==============================================================================
def _chunk_to_milvus_row(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    embedding = chunk.get("embedding")
    if not embedding or not isinstance(embedding, list):
        return None

    meta = chunk.get("metadata", {}) or {}
    page = chunk.get("page")
    page_val = int(page) if page is not None else -1

    return {
        "chunk_id": str(chunk.get("chunk_id", ""))[:255],
        "company": str(chunk.get("company", "UNKNOWN"))[:50],
        "period": str(chunk.get("period", "UNKNOWN"))[:50],
        "document_type": str(chunk.get("document_type", "UNKNOWN"))[:100],
        "source": str(chunk.get("source", ""))[:255],
        "page": page_val,
        "is_table": bool(meta.get("is_table", False)),
        "text": str(chunk.get("text", ""))[:65000],
        "embedding": embedding,
    }


def ingest_jsonl_text_to_milvus(
    collection, jsonl_text: str, batch_size: int = 500
) -> Dict[str, int]:
    stats = {"inserted": 0, "skipped": 0, "invalid": 0}
    batch: List[Dict[str, Any]] = []

    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            stats["invalid"] += 1
            continue

        row = _chunk_to_milvus_row(chunk)
        if row is None:
            stats["skipped"] += 1
            continue

        batch.append(row)
        if len(batch) >= batch_size:
            collection.insert(batch)
            stats["inserted"] += len(batch)
            batch = []

    if batch:
        collection.insert(batch)
        stats["inserted"] += len(batch)

    return stats


def index_all(
    collection_name: str = DEFAULT_COLLECTION,
    dim: int = DEFAULT_DIM,
    batch_size: int = 500,
    reset: bool = False,
    force: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    # 1. Setup collection
    collection = setup_collection(collection_name, dim, reset=reset)

    # 2. Lister les JSONL dans MinIO
    keys = [k for k in storage.list(INPUT_PREFIX) if k.lower().endswith(".jsonl")]
    if limit:
        keys = keys[:limit]
    logger.info(f"🔍 {len(keys)} fichiers JSONL à ingérer")

    manifest = _load_manifest()
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_inserted": 0,
        "total_invalid": 0,
    }

    try:
        for key in keys:
            prev = manifest.get(key)

            # Idempotence
            if not force and not reset and prev:
                try:
                    current_hash = compute_key_hash(key)
                except Exception:
                    current_hash = None
                if current_hash and current_hash == prev.get("source_hash"):
                    stats["skipped"] += 1
                    continue

            try:
                t0 = time.time()
                jsonl = storage.read_text(key)
                result = ingest_jsonl_text_to_milvus(collection, jsonl, batch_size)
                elapsed = time.time() - t0

                manifest[key] = {
                    "source_hash": compute_key_hash(key),
                    "indexed_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "count": result["inserted"],
                }

                stats["processed"] += 1
                stats["total_inserted"] += result["inserted"]
                stats["total_invalid"] += result["invalid"]

                # Log tous les 50 pour ne pas spammer
                if stats["processed"] % 50 == 0:
                    logger.info(
                        f"  → {stats['processed']}/{len(keys)} fichiers, "
                        f"{stats['total_inserted']} chunks insérés"
                    )

                # Checkpoint manifest toutes les 50
                if stats["processed"] % 50 == 0:
                    _save_manifest(manifest)

            except Exception as e:
                logger.error(f"❌ Échec : {key} → {e}")
                stats["failed"] += 1
    finally:
        _save_manifest(manifest)

    # 3. Flush + Load
    logger.info("⏳ Flush des données sur disque...")
    collection.flush()
    logger.info("⏳ Chargement de la collection en mémoire...")
    collection.load()
    logger.info("✅ Collection chargée et prête pour la recherche")

    return stats


def print_collection_stats(collection_name: str = DEFAULT_COLLECTION):
    from pymilvus import Collection, utility

    if not utility.has_collection(collection_name):
        logger.error(f"❌ Collection '{collection_name}' n'existe pas")
        return

    collection = Collection(collection_name)
    collection.load()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 Collection '{collection_name}'")
    logger.info(f"   Entities : {collection.num_entities}")
    logger.info(f"   Schema   : {len(collection.schema.fields)} champs")

    try:
        results = collection.query(
            expr="chunk_id != ''",
            output_fields=["company"],
            limit=16384,
        )
        from collections import Counter
        counts = Counter(r["company"] for r in results)
        logger.info(f"   Top companies (sur {len(results)} échantillons) :")
        for company, count in counts.most_common(10):
            logger.info(f"      - {company}: {count}")
    except Exception as e:
        logger.warning(f"   ⚠️  Répartition indisponible : {e}")

    logger.info(f"{'=' * 60}")


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--reset", action="store_true", help="⚠️  DESTRUCTIF")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Pour test")
    args = ap.parse_args()

    connect_milvus()

    if args.stats:
        print_collection_stats(args.collection)
        sys.exit(0)

    t0 = time.time()
    stats = index_all(
        collection_name=args.collection,
        dim=args.dim,
        batch_size=args.batch_size,
        reset=args.reset,
        force=args.force,
        limit=args.limit,
    )
    elapsed = time.time() - t0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Ingestion Milvus terminée !")
    logger.info(f"   ✅ Fichiers traités   : {stats['processed']}")
    logger.info(f"   ⏭️  Fichiers ignorés  : {stats['skipped']}")
    logger.info(f"   ❌ Échecs             : {stats['failed']}")
    logger.info(f"   📊 Chunks insérés     : {stats['total_inserted']}")
    logger.info(f"   ⚠️  Chunks invalides  : {stats['total_invalid']}")
    logger.info(f"   ⏱️  Temps total       : {elapsed:.1f}s")
    if stats["total_inserted"] > 0 and elapsed > 0:
        logger.info(f"   🚀 Débit              : {stats['total_inserted'] / elapsed:.0f} chunks/s")
    logger.info(f"{'=' * 60}")