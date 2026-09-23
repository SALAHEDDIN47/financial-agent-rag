# src/core/bm25.py
"""
Indexation BM25 dans Elasticsearch pour la recherche hybride.

Améliorations vs version initiale :
  - ✅ rglob (récursif) au lieu de glob → trouve tous les chunks
  - ✅ Analyzer corrigé (stopwords en tableau)
  - ✅ Idempotence via manifest (data/processed/.bm25_manifest.json)
  - ✅ Non-destructif par défaut (drop index seulement avec --reset)
  - ✅ Checkpoint périodique (tous les 100 fichiers OU 60s)
  - ✅ Métadonnées enrichies : page, is_table, token_count, chunk_index, total_chunks
  - ✅ Ctrl+C géré proprement
  - ✅ Bulk indexing via streaming generator (mémoire O(1))
  - ✅ Utilise settings.py au lieu de hardcoder l'URL
  - ✅ --stats pour afficher l'état sans réindexer

Usage :
    uv run ./src/core/bm25.py                # ingère les nouveaux
    uv run ./src/core/bm25.py --reset        # ⚠️ DESTRUCTIF : drop + réindexe tout
    uv run ./src/core/bm25.py --force        # re-indexe même les fichiers inchangés
    uv run ./src/core/bm25.py --stats        # affiche les stats
    uv run ./src/core/bm25.py --search "risk factors"
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

from elasticsearch import Elasticsearch, helpers

# Import des settings centralisés
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from src.config.settings import settings
except ImportError:
    settings = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEFAULT_INDEX = "financial_chunks_bm25"
MANIFEST_PATH = Path("data/processed/.bm25_manifest.json")

# Mapping document_type → priorité BM25 (optionnel, pour tuning)
# On peut booster certaines sections (risk factors, financial statements)
BOOST_FIELDS = {
    "document_type": 2.0,  # si la requête cible un type, on booste
    "company": 1.5,
}


# ==============================================================================
# MANIFEST D'IDEMPOTENCE
# ==============================================================================
def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("⚠️  Manifest BM25 corrompu, redémarrage à vide")
            return {}
    return {}


def _save_manifest(manifest: dict) -> None:
    """Sauvegarde atomique du manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(MANIFEST_PATH)


def _compute_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# CLASSE PRINCIPALE
# ==============================================================================
class ElasticsearchIndexer:
    """Indexe les chunks dans Elasticsearch pour la recherche lexicale BM25."""

    def __init__(
        self,
        es_url: str = None,
        index_name: str = DEFAULT_INDEX,
    ):
        if es_url is None:
            es_url = getattr(settings, "elasticsearch_host", None) or "http://localhost:9200"
        self.es_url = es_url
        self.index_name = index_name
        self.es: Optional[Elasticsearch] = None

    # -------------------------------------------------------------------------
    # CONNEXION
    # -------------------------------------------------------------------------
    def connect(self) -> None:
        """Se connecte à Elasticsearch avec retry."""
        try:
            self.es = Elasticsearch(
                [self.es_url],
                verify_certs=False,
                ssl_show_warn=False,
                request_timeout=60,
            )
            if not self.es.ping():
                raise ConnectionError("ping échoué")

            info = self.es.info()
            logger.info(
                f"✅ Connecté à Elasticsearch {self.es_url} "
                f"(v{info['version']['number']})"
            )
        except Exception as e:
            logger.error(f"❌ Erreur de connexion à Elasticsearch : {e}")
            raise

    # -------------------------------------------------------------------------
    # CRÉATION D'INDEX (non-destructif par défaut)
    # -------------------------------------------------------------------------
    def create_index(self, reset: bool = False) -> None:
        """
        Crée l'index si nécessaire.
        ⚠️ Ne le supprime PAS sauf si `reset=True`.
        """
        exists = self.es.indices.exists(index=self.index_name)

        if exists and reset:
            logger.warning(f"⚠️  --reset : suppression de l'index '{self.index_name}'")
            self.es.indices.delete(index=self.index_name)
            exists = False

        if exists:
            logger.info(f"ℹ️  Index '{self.index_name}' existe déjà (conservé)")
            return

        # ✅ FIX : stopwords en TABLEAU, plus de valeur scalaire invalide
        mappings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "financial_analyzer": {
                            "type": "standard",
                            "stopwords": ["_english_"],   # ← crochets !
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
                    # ✅ NOUVEAU : page et is_table pour la recherche hybride
                    "page": {"type": "integer"},
                    "is_table": {"type": "boolean"},
                    "text": {
                        "type": "text",
                        "analyzer": "financial_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        },
                    },
                    # ✅ NOUVEAU : chunk_index, total_chunks, token_count
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "token_count": {"type": "integer"},
                }
            },
        }

        self.es.indices.create(index=self.index_name, body=mappings)
        logger.info(f"✅ Index '{self.index_name}' créé")

    # -------------------------------------------------------------------------
    # GÉNÉRATION DES ACTIONS BULK
    # -------------------------------------------------------------------------
    def _generate_actions(
        self, jsonl_files: list, batch_stats: dict
    ) -> Iterator[dict]:
        """
        Génère les actions pour helpers.bulk en streaming.
        Avantage : ne charge JAMAIS tout en mémoire (O(1)).
        """
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
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
                            # ✅ Métadonnées enrichies
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
        chunks_dir: str = "data/processed/chunks",
        batch_size: int = 1000,
        force: bool = False,
        reset: bool = False,
        save_every: int = 100,
        save_interval_s: int = 60,
    ) -> Dict[str, Any]:
        """
        Indexe tous les chunks avec idempotence.

        Skip un fichier si :
          - Déjà dans le manifest ET le hash est identique ET --force n'est pas activé.
        """
        chunks_path = Path(chunks_dir)

        # ✅ FIX : rglob au lieu de glob
        jsonl_files = sorted(chunks_path.rglob("*.jsonl"))
        if not jsonl_files:
            logger.error(f"❌ Aucun fichier JSONL dans {chunks_dir}")
            return {"processed": 0, "skipped": 0, "failed": 0, "total": 0}

        logger.info(f"📂 {len(jsonl_files)} fichiers JSONL trouvés")

        manifest = _load_manifest()
        stats = {"processed": 0, "skipped": 0, "failed": 0, "total_indexed": 0}

        # --- Filtrer les fichiers à traiter (idempotence) ---
        files_to_index = []
        for f in jsonl_files:
            key = str(f)
            prev = manifest.get(key)

            if not force and not reset and prev:
                try:
                    current_hash = _compute_hash(f)
                except (FileNotFoundError, PermissionError):
                    current_hash = None
                if current_hash and current_hash == prev.get("source_hash"):
                    stats["skipped"] += 1
                    continue
            files_to_index.append(f)

        logger.info(
            f"📊 {len(files_to_index)} à indexer, {stats['skipped']} ignorés (déjà indexés)"
        )

        if not files_to_index:
            logger.info("✅ Rien à faire, tout est déjà indexé")
            return stats

        # --- Indexation par bulk ---
        batch_stats = {"generated": 0}
        processed_since_save = 0
        last_save_time = time.time()

        def _checkpoint(force_save: bool = False):
            nonlocal processed_since_save, last_save_time
            now = time.time()
            if (
                force_save
                or processed_since_save >= save_every
                or (now - last_save_time) >= save_interval_s
            ):
                _save_manifest(manifest)
                processed_since_save = 0
                last_save_time = now
                logger.debug(f"💾 Manifest BM25 sauvegardé ({len(manifest)} entrées)")

        try:
            # --- Appel bulk unique (le plus efficace) ---
            logger.info("⏳ Indexation bulk en cours...")
            client = self.es.options(request_timeout=180, max_retries=3)

            success, failed_list = helpers.bulk(
                client,
                self._generate_actions(files_to_index, batch_stats),
                chunk_size=batch_size,
                raise_on_error=False,
                stats_only=False,
            )

            if isinstance(failed_list, list) and failed_list:
                logger.warning(f"⚠️  {len(failed_list)} documents en échec")
                stats["failed"] = len(failed_list)

            stats["total_indexed"] = success
            stats["processed"] = len(files_to_index)

            # Mettre à jour le manifest pour tous les fichiers traités
            parsed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            for f in files_to_index:
                manifest[str(f)] = {
                    "source_hash": _compute_hash(f),
                    "indexed_at": parsed_at,
                }
                processed_since_save += 1
                _checkpoint()

        except KeyboardInterrupt:
            logger.warning("\n⏸️  Ctrl+C détecté — sauvegarde du manifest...")
        finally:
            _save_manifest(manifest)
            logger.info(f"💾 Manifest BM25 sauvegardé : {len(manifest)} entrées")

        return stats

    # -------------------------------------------------------------------------
    # RAFRAÎCHISSEMENT + STATS
    # -------------------------------------------------------------------------
    def refresh(self) -> None:
        """Force ES à rendre les documents visibles (normalement auto toutes les 1s)."""
        self.es.indices.refresh(index=self.index_name)

    def get_stats(self) -> int:
        """Retourne le nombre de documents dans l'index."""
        try:
            self.refresh()
            count = self.es.count(index=self.index_name)["count"]
            logger.info(f"📊 Index '{self.index_name}' : {count} documents")
            return count
        except Exception as e:
            logger.error(f"❌ Erreur stats : {e}")
            return 0

    # -------------------------------------------------------------------------
    # RECHERCHE (test rapide)
    # -------------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str = None,
        document_type: str = None,
    ) -> list:
        """Recherche BM25 multi-champs."""
        must = [{"multi_match": {"query": query, "fields": ["text^2"]}}]

        filters = []
        if company:
            filters.append({"term": {"company": company.upper()}})
        if document_type:
            filters.append({"term": {"document_type": document_type}})

        body = {
            "query": {
                "bool": {
                    "must": must,
                    "filter": filters,
                }
            },
            "size": top_k,
        }

        result = self.es.search(index=self.index_name, body=body)
        return result["hits"]["hits"]


# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Indexation BM25 Elasticsearch")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/chunks"),
        help="Dossier des chunks JSONL",
    )
    ap.add_argument("--index", type=str, default=DEFAULT_INDEX)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument(
        "--reset",
        action="store_true",
        help="⚠️  DESTRUCTIF : supprime l'index et réindexe tout",
    )
    ap.add_argument(
        "--force", action="store_true", help="Réindexe même les fichiers inchangés"
    )
    ap.add_argument("--stats", action="store_true", help="Affiche les stats et quitte")
    ap.add_argument("--search", type=str, help="Test rapide de recherche")
    args = ap.parse_args()

    indexer = ElasticsearchIndexer(index_name=args.index)
    indexer.connect()

    # Mode stats
    if args.stats:
        indexer.get_stats()
        return

    # Mode recherche rapide
    if args.search:
        hits = indexer.search(args.search, top_k=5)
        print(f"\n{'=' * 70}")
        print(f"  QUERY : {args.search}")
        print(f"{'=' * 70}\n")
        for i, hit in enumerate(hits, 1):
            src = hit["_source"]
            print(f"─── {i} ─── (score: {hit['_score']:.3f})")
            print(f"  {src.get('company')} | {src.get('period')} | {src.get('document_type')}")
            print(f"  {src.get('text', '')[:250]}...\n")
        return

    # Mode indexation
    t0 = time.time()
    indexer.create_index(reset=args.reset)
    stats = indexer.index_chunks(
        chunks_dir=str(args.input),
        batch_size=args.batch_size,
        force=args.force,
        reset=args.reset,
    )
    elapsed = time.time() - t0

    doc_count = indexer.get_stats()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Indexation BM25 terminée !")
    logger.info(f"   ✅ Fichiers traités   : {stats['processed']}")
    logger.info(f"   ⏭️  Fichiers ignorés  : {stats['skipped']}")
    logger.info(f"   ❌ Échecs             : {stats['failed']}")
    logger.info(f"   📊 Documents indexés  : {stats['total_indexed']}")
    logger.info(f"   📦 Total dans ES      : {doc_count}")
    logger.info(f"   ⏱️  Temps total       : {elapsed:.1f}s")
    if elapsed > 0:
        logger.info(f"   🚀 Débit              : {stats['total_indexed'] / elapsed:.0f} docs/s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()