# src/core/reset_databases.py
import logging
from pymilvus import connections, utility
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def reset_milvus():
    logger.info("🔄 Connexion à Milvus...")
    try:
        connections.connect("default", host="localhost", port="19530")
        collection_name = "financial_chunks"
        if utility.has_collection(collection_name):
            logger.info(f"🗑️ Suppression de la collection Milvus : {collection_name}")
            utility.drop_collection(collection_name)
            logger.info("✅ Collection Milvus supprimée.")
        else:
            logger.info("ℹ️ La collection Milvus n'existe pas encore.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réinitialisation de Milvus : {e}")

def reset_elasticsearch():
    logger.info("🔄 Connexion à Elasticsearch...")
    try:
        es = Elasticsearch("http://localhost:9200")
        index_name = "financial_chunks_bm25"
        if es.indices.exists(index=index_name):
            logger.info(f"🗑️ Suppression de l'index Elasticsearch : {index_name}")
            es.indices.delete(index=index_name)
            logger.info("✅ Index Elasticsearch supprimé.")
        else:
            logger.info("ℹ️ L'index Elasticsearch n'existe pas encore.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réinitialisation d'Elasticsearch : {e}")

def reset_manifests():
    """Supprime les manifests pour forcer le retraitement complet."""
    manifests = [
        Path("data/interim/.parsing_manifest.json"),
        Path("data/processed/.chunking_manifest.json"),
        Path("data/processed/.embedding_manifest.json"),
        Path("data/processed/.milvus_manifest.json"),
    ]
    for m in manifests:
        if m.exists():
            logger.info(f"🗑️  Suppression du manifest : {m}")
            m.unlink()

if __name__ == "__main__":
    logger.info("🚀 Démarrage de la réinitialisation des bases de données...")
    reset_milvus()
    reset_elasticsearch()
    logger.info("🎉 Réinitialisation terminée ! Vous pouvez relancer votre pipeline d'ingestion.")