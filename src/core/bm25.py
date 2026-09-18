# src/core/bm25.py
import json
import logging
from pathlib import Path
from elasticsearch import Elasticsearch, helpers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class ElasticsearchIndexer:
    """Indexe les chunks dans Elasticsearch pour la recherche lexicale BM25."""
    
    def __init__(self, es_url: str = "http://localhost:9200", index_name: str = "financial_chunks_bm25"):
        self.es_url = es_url
        self.index_name = index_name
        self.es = None
    
    def connect(self):
        """Se connecte à Elasticsearch."""
        try:
            self.es = Elasticsearch(
                [self.es_url],
                verify_certs=False,      # Désactivé pour le développement local
                ssl_show_warn=False      # Évite les warnings SSL en local
            )
            if self.es.ping():
                logger.info(f"✅ Connecté à Elasticsearch : {self.es_url}")
                info = self.es.info()
                logger.info(f"📦 Version du serveur Elasticsearch : {info['version']['number']}")
            else:
                raise ConnectionError("Le ping vers Elasticsearch a échoué.")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion à Elasticsearch : {e}")
            raise
    
    def create_index(self):
        """Crée l'index avec les mappings optimisés pour BM25."""
        if self.es.indices.exists(index=self.index_name):
            logger.info(f"⚠️ L'index '{self.index_name}' existe déjà. Suppression...")
            self.es.indices.delete(index=self.index_name)
        
        mappings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "financial_analyzer": {
                            "type": "standard",
                            "stopwords": ["_english_"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "company": {"type": "keyword"},
                    "period": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "financial_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"}
                }
            }
        }
        
        self.es.indices.create(index=self.index_name, body=mappings)
        logger.info(f"✅ Index '{self.index_name}' créé avec les mappings BM25")
    
    def index_chunks(self, chunks_dir: str, batch_size: int = 1000):
        """Indexe tous les chunks depuis les fichiers JSONL."""
        chunks_path = Path(chunks_dir)
        jsonl_files = list(chunks_path.rglob("*.jsonl"))
        
        if not jsonl_files:
            logger.error(f"❌ Aucun fichier JSONL trouvé dans {chunks_dir}")
            return
        
        logger.info(f"📂 {len(jsonl_files)} fichiers JSONL à indexer")
        
        def generate_actions():
            """Génère les actions d'indexation pour elasticsearch.helpers.bulk."""
            for jsonl_file in jsonl_files:
                logger.info(f"📥 Traitement de {jsonl_file.name}...")
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        chunk = json.loads(line)
                        yield {
                            "_index": self.index_name,
                            "_id": chunk.get("chunk_id"),
                            "_source": {
                                "chunk_id": chunk.get("chunk_id"),
                                "source": chunk.get("source"),
                                "company": chunk.get("company"),
                                "period": chunk.get("period"),
                                "document_type": chunk.get("document_type"),
                                "text": chunk.get("text"),
                                "chunk_index": chunk.get("metadata", {}).get("chunk_index", 0),
                                "total_chunks": chunk.get("metadata", {}).get("total_chunks", 0)
                            }
                        }
        
        try:
            # ✅ FIX pour Elasticsearch v8 : on utilise .options() pour les paramètres de transport
            client_with_options = self.es.options(request_timeout=120, max_retries=3)
            
            success, failed = helpers.bulk(
                client_with_options,
                generate_actions(),
                chunk_size=batch_size
            )
            logger.info(f"✅ Indexation terminée ! {success} chunks indexés, {failed} échecs")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation : {e}")
            raise
    
    def get_stats(self):
        """Récupère les statistiques de l'index."""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            doc_count = stats["indices"][self.index_name]["total"]["docs"]["count"]
            logger.info(f"📊 Statistiques de l'index '{self.index_name}' : {doc_count} documents")
            return doc_count
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des statistiques : {e}")
            return 0

def main():
    """Fonction principale pour l'indexation Elasticsearch."""
    indexer = ElasticsearchIndexer(
        es_url="http://localhost:9200",
        index_name="financial_chunks_bm25"
    )
    
    try:
        indexer.connect()
        indexer.create_index()
        indexer.index_chunks(chunks_dir="data/processed/chunks", batch_size=1000)
        indexer.get_stats()
        logger.info("🎉 Indexation Elasticsearch terminée avec succès !")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'indexation Elasticsearch : {e}")
        raise

if __name__ == "__main__":
    main()