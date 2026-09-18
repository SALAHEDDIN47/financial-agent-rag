# src/indexing/milvus_indexer.py
"""
Indexation des chunks embeddés dans Milvus.

Fonctionnalités :
  - Non-destructif : ne supprime PAS la collection sauf --reset explicite
  - Idempotence via manifest (data/processed/.milvus_manifest.json)
  - Retry sur erreurs transitoires (connexion, timeout)
  - Flush + load après ingestion
  - Vérification de la connexion avant tout
  - Rapport détaillé (inserts, doublons, échecs)

Usage :
    uv run ./src/indexing/milvus_indexer.py                  # ingère les nouveaux
    uv run ./src/indexing/milvus_indexer.py --reset          # DESTRUCTIF : reset tout
    uv run ./src/indexing/milvus_indexer.py --recreate       # recrée sans ingérer
    uv run ./src/indexing/milvus_indexer.py --stats          # affiche les stats
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ==============================================================================
# MANIFEST D'IDEMPOTENCE
# ==============================================================================
MANIFEST_PATH = Path("data/processed/.milvus_manifest.json")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _compute_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# SETUP MILVUS
# ==============================================================================
DEFAULT_COLLECTION = "financial_chunks"
DEFAULT_DIM = 1024  # BGE-large-en-v1.5
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"


def connect_milvus(host: str = MILVUS_HOST, port: str = MILVUS_PORT, retries: int = 3):
    """Établit la connexion à Milvus avec retry."""
    from pymilvus import connections

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            connections.connect(
                alias="default", host=host, port=port, timeout=10
            )
            logger.info(f"✅ Connecté à Milvus ({host}:{port})")
            return
        except Exception as e:
            last_error = e
            logger.warning(
                f"⚠️  Tentative {attempt}/{retries} échouée : {e}"
            )
            if attempt < retries:
                time.sleep(2)
    raise ConnectionError(f"Impossible de se connecter à Milvus : {last_error}")


def setup_collection(
    collection_name: str = DEFAULT_COLLECTION,
    dim: int = DEFAULT_DIM,
    reset: bool = False,
):
    """
    Crée la collection si elle n'existe pas.
    ⚠️ Ne supprime PAS une collection existante, sauf si `reset=True`.
    """
    from pymilvus import (
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )

    exists = utility.has_collection(collection_name)

    if exists and reset:
        logger.warning(f"⚠️  --reset demandé : suppression de '{collection_name}'")
        utility.drop_collection(collection_name)
        exists = False

    if exists:
        logger.info(f"ℹ️  Collection '{collection_name}' existe déjà (conservée)")
        return Collection(collection_name)

    # Création
    fields = [
        FieldSchema(
            name="chunk_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True
        ),
        FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="period", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(
            name="page",
            dtype=DataType.INT32,
        ),  # -1 si non applicable
        FieldSchema(
            name="is_table", dtype=DataType.BOOL
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(
        fields, description="Chunks financiers RAG (multi-entreprises)"
    )
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"✅ Collection '{collection_name}' créée (dim={dim})")

    # Index HNSW pour recherche cosinus
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
    """
    Convertit un chunk JSON en ligne Milvus.
    Retourne None si le chunk est invalide.
    """
    embedding = chunk.get("embedding")
    if not embedding or not isinstance(embedding, list):
        return None

    meta = chunk.get("metadata", {}) or {}

    # Page : -1 si None (Milvus INT32 n'accepte pas None)
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
        "text": str(chunk.get("text", ""))[:65000],  # sécurité VARCHAR
        "embedding": embedding,
    }


def ingest_file_to_milvus(
    collection,
    input_file: Path,
    batch_size: int = 500,
) -> Dict[str, int]:
    """
    Ingère un fichier JSONL dans Milvus par lots.
    Retourne {"inserted": N, "skipped": K, "invalid": M}
    """
    stats = {"inserted": 0, "skipped": 0, "invalid": 0}

    batch: List[Dict[str, Any]] = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
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

    # Dernier batch partiel
    if batch:
        collection.insert(batch)
        stats["inserted"] += len(batch)

    return stats


def index_directory(
    input_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
    dim: int = DEFAULT_DIM,
    batch_size: int = 500,
    reset: bool = False,
    force: bool = False,
) -> Dict[str, int]:
    """
    Ingère tous les JSONL embeddés dans Milvus.
    Idempotent : utilise un manifest pour skip les fichiers déjà ingérés.
    """
    input_dir = Path(input_dir)

    # 1. Setup collection
    collection = setup_collection(collection_name, dim, reset=reset)

    # 2. Récupérer tous les JSONL
    jsonl_files = sorted(input_dir.rglob("*.jsonl"))
    logger.info(f"🔍 {len(jsonl_files)} fichiers JSONL à ingérer")

    manifest = _load_manifest()
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_inserted": 0,
        "total_invalid": 0,
    }

    try:
        for input_file in jsonl_files:
            key = str(input_file)
            prev = manifest.get(key)

            # Vérification idempotente
            if not force and not reset and prev:
                try:
                    current_hash = _compute_hash(input_file)
                except (FileNotFoundError, PermissionError):
                    current_hash = None
                if current_hash and current_hash == prev.get("source_hash"):
                    stats["skipped"] += 1
                    continue

            # Ingestion
            try:
                t0 = time.time()
                result = ingest_file_to_milvus(
                    collection, input_file, batch_size=batch_size
                )
                elapsed = time.time() - t0

                manifest[key] = {
                    "source_hash": _compute_hash(input_file),
                    "indexed_at": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    "count": result["inserted"],
                }

                stats["processed"] += 1
                stats["total_inserted"] += result["inserted"]
                stats["total_invalid"] += result["invalid"]

                logger.info(
                    f"✅ {input_file.name} : {result['inserted']} insérés "
                    f"(skip={result['skipped']}, invalid={result['invalid']}) "
                    f"en {elapsed:.1f}s"
                )

            except Exception as e:
                logger.error(f"❌ Échec : {input_file.name} → {e}")
                stats["failed"] += 1
    finally:
        _save_manifest(manifest)

    # 3. Flush + Load (crucial pour rendre les données disponibles)
    logger.info("⏳ Flush des données sur disque...")
    collection.flush()
    logger.info("⏳ Chargement de la collection en mémoire...")
    collection.load()
    logger.info("✅ Collection chargée et prête pour la recherche")

    return stats


def print_collection_stats(collection_name: str = DEFAULT_COLLECTION):
    """Affiche les statistiques de la collection."""
    from pymilvus import Collection, utility

    if not utility.has_collection(collection_name):
        logger.error(f"❌ Collection '{collection_name}' n'existe pas")
        return

    collection = Collection(collection_name)
    collection.load()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 Statistiques de la collection '{collection_name}'")
    logger.info(f"   Num entities : {collection.num_entities}")
    logger.info(f"   Schema       : {len(collection.schema.fields)} champs")

    # Répartition par company (via query)
    try:
        results = collection.query(
            expr="chunk_id != ''",
            output_fields=["company"],
            limit=16384,  # Milvus limite les queries sans pagination
        )
        from collections import Counter
        counts = Counter(r["company"] for r in results)
        logger.info(f"   Top companies (sur {len(results)} entités échantillonnées) :")
        for company, count in counts.most_common(10):
            logger.info(f"      - {company}: {count}")
    except Exception as e:
        logger.warning(f"   ⚠️  Impossible de calculer la répartition : {e}")

    logger.info(f"{'=' * 60}")


# ==============================================================================
# EXÉCUTION DIRECTE
# ==============================================================================
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/embedded_chunks"),
        help="Dossier des chunks embeddés",
    )
    ap.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument(
        "--reset",
        action="store_true",
        help="⚠️  DESTRUCTIF : supprime la collection et réindexe tout",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-ingère même les fichiers déjà indexés",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Affiche les statistiques et quitte",
    )
    args = ap.parse_args()

    connect_milvus()

    if args.stats:
        print_collection_stats(args.collection)
        sys.exit(0)

    t_global = time.time()
    stats = index_directory(
        input_dir=args.input,
        collection_name=args.collection,
        dim=args.dim,
        batch_size=args.batch_size,
        reset=args.reset,
        force=args.force,
    )
    elapsed = time.time() - t_global

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Ingestion Milvus terminée !")
    logger.info(f"   ✅ Fichiers traités    : {stats['processed']}")
    logger.info(f"   ⏭️  Fichiers ignorés   : {stats['skipped']}")
    logger.info(f"   ❌ Échecs              : {stats['failed']}")
    logger.info(f"   📊 Chunks insérés      : {stats['total_inserted']}")
    logger.info(f"   ⚠️  Chunks invalides   : {stats['total_invalid']}")
    logger.info(f"   ⏱️  Temps total        : {elapsed:.1f}s")
    if stats["total_inserted"] > 0 and elapsed > 0:
        logger.info(f"   🚀 Débit moyen         : {stats['total_inserted'] / elapsed:.0f} chunks/s")
    logger.info(f"{'=' * 60}")