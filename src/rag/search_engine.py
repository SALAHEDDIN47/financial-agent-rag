# src/rag/search_engine.py
import json
import logging
from pathlib import Path
from fastembed import TextEmbedding  # Remplace sentence_transformers
from pymilvus import connections, Collection
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        es_host: str = "http://localhost:9200",
        collection_name: str = "financial_chunks",
        es_index_name: str = "financial_chunks_bm25",
        model_name: str = "BAAI/bge-large-en-v1.5",
    ):
        self.collection_name = collection_name
        self.es_index_name = es_index_name
        
        # ✅ Utiliser fastembed au lieu de sentence-transformers
        logger.info(f" Chargement du modèle d'embedding : {model_name}")
        self.embedding_model = TextEmbedding(model_name=model_name)
        
        # Connexion à Milvus
        logger.info("🔌 Connexion à Milvus...")
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.collection = Collection(self.collection_name)
        logger.info(f"✅ Collection '{self.collection_name}' chargée")
        
        # Connexion à Elasticsearch
        logger.info("🔌 Connexion à Elasticsearch...")
        self.es = Elasticsearch(es_host)
        if self.es.ping():
            logger.info("✅ Elasticsearch connecté")
        else:
            logger.warning("⚠️ Elasticsearch non disponible")
    
    def _get_embedding(self, text: str) -> list[float]:
        """Génère un embedding pour un texte"""
        # fastembed retourne un générateur, on prend le premier résultat
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()
    
    def search_semantic(self, query: str, top_k: int = 10):
        """Recherche sémantique dans Milvus"""
        query_embedding = self._get_embedding(query)
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "text", "company", "period", "document_type"]
        )
        
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "text": hit.entity.get("text"),
                    "company": hit.entity.get("company"),
                    "period": hit.entity.get("period"),
                    "document_type": hit.entity.get("document_type"),
                    "score": hit.score
                })
        
        return search_results
    
    def search_lexical(self, query: str, top_k: int = 10):
        """Recherche lexicale dans Elasticsearch"""
        query_body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }
        
        response = self.es.search(index=self.es_index_name, body=query_body)
        
        search_results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            search_results.append({
                "chunk_id": source.get("chunk_id"),
                "text": source.get("text"),
                "company": source.get("company"),
                "period": source.get("period"),
                "document_type": source.get("document_type"),
                "score": hit["_score"]
            })
        
        return search_results
    
    def search_hybrid(self, query: str, top_k: int = 10):
        """Recherche hybride combinant sémantique et lexicale"""
        semantic_results = self.search_semantic(query, top_k=top_k * 2)
        lexical_results = self.search_lexical(query, top_k=top_k * 2)
        
        # Fusion simple (on peut améliorer avec RRF plus tard)
        combined = semantic_results + lexical_results
        
        # Déduplication par chunk_id
        seen = set()
        unique_results = []
        for result in combined:
            if result["chunk_id"] not in seen:
                seen.add(result["chunk_id"])
                unique_results.append(result)
        
        return unique_results[:top_k]


if __name__ == "__main__":
    engine = SearchEngine()
    
    query = "What was Tesla's revenue in 2024?"
    print(f"\n🔍 Recherche : {query}\n")
    
    results = engine.search_hybrid(query, top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['company']} - {result['period']}] (Score: {result['score']:.4f})")
        print(f"   Type: {result['document_type']}")
        print(f"   Texte: {result['text'][:200]}...\n")