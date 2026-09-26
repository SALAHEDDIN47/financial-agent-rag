# src/core/bm25.py
"""
Indexation BM25 dans Elasticsearch (S3-aware).

Lit   : processed-chunks/{company}/{doc_type}/{source}_chunks.jsonl
Écrit : index ES 'financial_chunks_bm25'
Manifest : manifests/bm25_manifest.json

Usage :
    python -m src.core.bm25                 # ingère les nouveaux
    python -m src.core.bm25 --reset         # ⚠️ DESTRUCTIF : drop + réindexe
    python -m src.core.bm25 --force         # réindexe même les fichiers inchangés
    python -m src.core.bm25 --stats         # affiche le count
    python -m src.core.bm25 --search "risk factors"
    python -m src.core.bm25 --limit 100     # test rapide
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from elasticsearch import Elasticsearch, helpers

from src.config.settings import settings
from src.core.storage import compute_key_hash, storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================
DEFAULT_INDEX = "financial_chunks_bm25"
MANIFEST_KEY = "manifests/bm25_manifest.json"
INPUT_PREFIX = "processed-chunks/"


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
# INDEXER
# ==============================================================================
class ElasticsearchIndexer:
    """Indexe les chunks dans Elasticsearch pour la recherche BM25."""

    def __init__(self, es_url: Optional[str] = None, index_name: str = DEFAULT_INDEX):
        if es_url is None:
            es_url = settings.elasticsearch_host
        self.es_url = es_url
        self.index_name = index_name
        self.es: Optional[Elasticsearch] = None

    # -------------------------------------------------------------------------
    # CONNEXION
    # -------------------------------------------------------------------------
    def connect(self) -> None:
        try:
            self.es = Elasticsearch(
                [self.es_url],
                verify_certs=False,
                ssl_show_warn=False,
                request_timeout=120,
            )
            if not self.es.ping():
                raise ConnectionError("ping échoué")
            info = self.es.info()
            logger.info(
                f"✅ Connecté à Elasticsearch {self.es_url} "
                f"(v{info['version']['number']})"
            )
        except Exception as e:
            logger.error(f"❌ Erreur connexion ES : {e}")
            raise

    # -------------------------------------------------------------------------
    # CRÉATION D'INDEX
    # -------------------------------------------------------------------------
    def create_index(self, reset: bool = False) -> None:
        exists = self.es.indices.exists(index=self.index_name)

        if exists and reset:
            logger.warning(f"⚠️  --reset : suppression de l'index '{self.index_name}'")
            self.es.indices.delete(index=self.index_name)
            exists = False

        if exists:
            logger.info(f"ℹ️  Index '{self.index_name}' existe déjà (conservé)")
            return

        mappings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "financial_analyzer": {
                            "type": "standard",
                            "stopwords": ["_english_"],
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "company": {"type": "keyword"},
                    "period": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "is_table": {"type": "boolean"},
                    "text": {
                        "type": "text",
                        "analyzer": "financial_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        },
                    },
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "token_count": {"type": "integer"},
                }
            },
        }

        self.es.indices.create(index=self.index_name, body=mappings)
        logger.info(f"✅ Index '{self.index_name}' créé")

    # -------------------------------------------------------------------------
    # GÉNÉRATION BULK (streaming)
    # -------------------------------------------------------------------------
    def _generate_actions(
        self,
        keys_and_texts: list[tuple[str, str]],
        batch_stats: dict,
    ) -> Iterator[dict]:
        for _key, jsonl_text in keys_and_texts:
            for line in jsonl_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text", "")
                if not chunk_id or not text:
                    continue

                meta = chunk.get("metadata", {}) or {}
                page = chunk.get("page")
                page_val = int(page) if page is not None else -1

                yield {
                    "_index": self.index_name,
                    "_id": chunk_id,
                    "_source": {
                        "chunk_id": chunk_id,
                        "company": chunk.get("company", "UNKNOWN"),
                        "period": chunk.get("period", "UNKNOWN"),
                        "document_type": chunk.get("document_type", "UNKNOWN"),
                        "source": chunk.get("source", ""),
                        "page": page_val,
                        "is_table": bool(meta.get("is_table", False)),
                        "text": text,
                        "chunk_index": int(meta.get("chunk_index", 0)),
                        "total_chunks": int(meta.get("total_chunks", 0)),
                        "token_count": int(meta.get("token_count", 0)),
                    },
                }
                batch_stats["generated"] += 1

    # -------------------------------------------------------------------------
    # INDEXATION IDEMPOTENTE
    # -------------------------------------------------------------------------
    def index_chunks(
        self,
        batch_size: int = 1000,
        force: bool = False,
        reset: bool = False,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        # 1. Lister tous les JSONL dans MinIO
        keys = [
            k for k in storage.list(INPUT_PREFIX)
            if k.lower().endswith(".jsonl")
        ]
        if limit:
            keys = keys[:limit]
        logger.info(f"📂 {len(keys)} fichiers JSONL à indexer")

        if not keys:
            return {"processed": 0, "skipped": 0, "failed": 0, "total_indexed": 0}

        manifest = _load_manifest()
        stats = {"processed": 0, "skipped": 0, "failed": 0, "total_indexed": 0}

        # 2. Idempotence
        to_index: list[tuple[str, str]] = []
        for k in keys:
            prev = manifest.get(k)
            if not force and not reset and prev:
                try:
                    current_hash = compute_key_hash(k)
                except Exception:
                    current_hash = None
                if current_hash and current_hash == prev.get("source_hash"):
                    stats["skipped"] += 1
                    continue
            to_index.append((k, storage.read_text(k)))

        logger.info(
            f"📊 {len(to_index)} à indexer, {stats['skipped']} ignorés (déjà indexés)"
        )

        if not to_index:
            logger.info("✅ Rien à faire, tout est déjà indexé")
            return stats

        # 3. Bulk (streaming)
        batch_stats = {"generated": 0}
        try:
            logger.info("⏳ Indexation bulk en cours...")
            client = self.es.options(request_timeout=300, max_retries=3)

            success, failed_list = helpers.bulk(
                client,
                self._generate_actions(to_index, batch_stats),
                chunk_size=batch_size,
                raise_on_error=False,
                stats_only=False,
            )

            if isinstance(failed_list, list) and failed_list:
                logger.warning(f"⚠️  {len(failed_list)} documents en échec")
                stats["failed"] = len(failed_list)

            stats["total_indexed"] = success
            stats["processed"] = len(to_index)

            # Manifest
            ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            for k, _ in to_index:
                manifest[k] = {"source_hash": compute_key_hash(k), "indexed_at": ts}
            _save_manifest(manifest)
            logger.info(f"💾 Manifest BM25 sauvegardé : {len(manifest)} entrées")

        except KeyboardInterrupt:
            logger.warning("\n⏸️  Ctrl+C — sauvegarde manifest...")
            _save_manifest(manifest)

        return stats

    # -------------------------------------------------------------------------
    # STATS + SEARCH
    # -------------------------------------------------------------------------
    def refresh(self) -> None:
        self.es.indices.refresh(index=self.index_name)

    def get_stats(self) -> int:
        try:
            self.refresh()
            count = self.es.count(index=self.index_name)["count"]
            logger.info(f"📊 Index '{self.index_name}' : {count} documents")
            return count
        except Exception as e:
            logger.error(f"❌ Erreur stats : {e}")
            return 0

    def search(self, query: str, top_k: int = 5, company: str = None,
               document_type: str = None) -> list:
        must = [{"multi_match": {"query": query, "fields": ["text^2"]}}]
        filters = []
        if company:
            filters.append({"term": {"company": company.upper()}})
        if document_type:
            filters.append({"term": {"document_type": document_type}})

        body = {
            "query": {"bool": {"must": must, "filter": filters}},
            "size": top_k,
        }
        result = self.es.search(index=self.index_name, body=body)
        return result["hits"]["hits"]


# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Indexation BM25 S3-aware")
    ap.add_argument("--index", type=str, default=DEFAULT_INDEX)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--reset", action="store_true", help="⚠️  DESTRUCTIF")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--search", type=str)
    ap.add_argument("--limit", type=int, default=None, help="Test sur N fichiers")
    args = ap.parse_args()

    indexer = ElasticsearchIndexer(index_name=args.index)
    indexer.connect()

    if args.stats:
        indexer.get_stats()
        return

    if args.search:
        hits = indexer.search(args.search, top_k=5)
        print(f"\n{'=' * 70}\n  QUERY : {args.search}\n{'=' * 70}\n")
        for i, hit in enumerate(hits, 1):
            src = hit["_source"]
            print(f"─── {i} ─── (score: {hit['_score']:.3f})")
            print(f"  {src.get('company')} | {src.get('period')} | {src.get('document_type')}")
            print(f"  {src.get('text', '')[:250]}...\n")
        return

    t0 = time.time()
    indexer.create_index(reset=args.reset)
    stats = indexer.index_chunks(
        batch_size=args.batch_size,
        force=args.force,
        reset=args.reset,
        limit=args.limit,
    )
    elapsed = time.time() - t0
    doc_count = indexer.get_stats()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Indexation BM25 terminée !")
    logger.info(f"   ✅ Fichiers traités  : {stats['processed']}")
    logger.info(f"   ⏭️  Fichiers ignorés : {stats['skipped']}")
    logger.info(f"   ❌ Échecs            : {stats['failed']}")
    logger.info(f"   📊 Docs indexés      : {stats['total_indexed']}")
    logger.info(f"   📦 Total dans ES     : {doc_count}")
    logger.info(f"   ⏱️  Temps total      : {elapsed:.1f}s")
    if elapsed > 0:
        logger.info(f"   🚀 Débit             : {stats['total_indexed'] / elapsed:.0f} docs/s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()