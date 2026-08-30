# src/rag/search_engine.py
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from pymilvus import Collection, connections

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Résultat de recherche unifié."""
    chunk_id: str
    text: str
    score: float
    company: str
    period: str
    document_type: str
    source: str


class SearchEngine:
    """Moteur de recherche hybride combinant Milvus (sémantique) et Elasticsearch (lexical)."""
    
    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        es_host: str = "http://localhost:9200",
        collection_name: str = "financial_chunks",
        es_index_name: str = "financial_chunks_bm25",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        top_k: int = 10,
    ):
        self.collection_name = collection_name
        self.es_index_name = es_index_name
        self.top_k = top_k
        
        # Initialisation du modèle d'embedding
        logger.info(f"🧠 Chargement du modèle d'embedding : {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Connexion à Milvus
        logger.info("🔌 Connexion à Milvus...")
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info(f"✅ Collection '{self.collection_name}' chargée.")
        
        # Connexion à Elasticsearch
        logger.info(" Connexion à Elasticsearch...")
        self.es = Elasticsearch(es_host)
        if self.es.ping():
            logger.info("✅ Elasticsearch connecté.")
        else:
            logger.warning("⚠️ Elasticsearch non disponible. La recherche lexicale sera désactivée.")
            self.es = None
    
    def search_semantic(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, str]] = None) -> List[SearchResult]:
        """Recherche sémantique via Milvus (similarité cosinus)."""
        top_k = top_k or self.top_k
        
        # Génération de l'embedding de la requête
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        
        # Construction de la requête Milvus
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }
        
        # Filtres optionnels (ex: company="TSLA")
        expr = None
        if filters:
            filter_parts = []
            for key, value in filters.items():
                filter_parts.append(f'{key} == "{value}"')
            expr = " and ".join(filter_parts)
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "text", "company", "period", "document_type", "source"]
        )
        
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    chunk_id=hit.entity.get("chunk_id", ""),
                    text=hit.entity.get("text", ""),
                    score=hit.score,
                    company=hit.entity.get("company", ""),
                    period=hit.entity.get("period", ""),
                    document_type=hit.entity.get("document_type", ""),
                    source=hit.entity.get("source", "")
                ))
        
        return search_results
    
    def search_lexical(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, str]] = None) -> List[SearchResult]:
        """Recherche lexicale via Elasticsearch (BM25)."""
        if not self.es:
            logger.warning("️ Elasticsearch non disponible.")
            return []
        
        top_k = top_k or self.top_k
        
        # Construction de la requête Elasticsearch
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": query}}
                    ]
                }
            },
            "size": top_k
        }
        
        # Ajout des filtres
        if filters:
            filter_clauses = [{"term": {key: value}} for key, value in filters.items()]
            query_body["query"]["bool"]["filter"] = filter_clauses
        
        response = self.es.search(index=self.es_index_name, body=query_body)
        
        search_results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            search_results.append(SearchResult(
                chunk_id=source.get("chunk_id", ""),
                text=source.get("text", ""),
                score=hit["_score"],
                company=source.get("company", ""),
                period=source.get("period", ""),
                document_type=source.get("document_type", ""),
                source=source.get("source", "")
            ))
        
        return search_results
    
    def search_hybrid(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, str]] = None, k_param: int = 60) -> List[SearchResult]:
        """
        Recherche hybride combinant résultats sémantiques et lexicaux via RRF (Reciprocal Rank Fusion).
        
        Formule RRF : score = Σ 1 / (k + rank_i)
        où k est une constante (généralement 60) et rank_i est le rang du document dans la liste i.
        """
        top_k = top_k or self.top_k
        
        # Récupération des résultats des deux sources
        semantic_results = self.search_semantic(query, top_k=top_k * 2, filters=filters)
        lexical_results = self.search_lexical(query, top_k=top_k * 2, filters=filters)
        
        logger.info(f" Résultats : {len(semantic_results)} sémantiques, {len(lexical_results)} lexicaux")
        
        # Fusion RRF
        return self._rrf_fusion(semantic_results, lexical_results, top_k=top_k, k=k_param)
    
    def _rrf_fusion(self, semantic_results: List[SearchResult], lexical_results: List[SearchResult], top_k: int, k: int = 60) -> List[SearchResult]:
        """Fusion des résultats par Reciprocal Rank Fusion."""
        # Dictionnaire pour accumuler les scores RRF par chunk_id
        rrf_scores = {}
        chunk_map = {}
        
        # Traitement des résultats sémantiques
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result.chunk_id
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                chunk_map[chunk_id] = result
            rrf_scores[chunk_id] += 1.0 / (k + rank)
        
        # Traitement des résultats lexicaux
        for rank, result in enumerate(lexical_results, start=1):
            chunk_id = result.chunk_id
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                chunk_map[chunk_id] = result
            rrf_scores[chunk_id] += 1.0 / (k + rank)
        
        # Tri par score RRF décroissant
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Construction des résultats finaux
        final_results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            result = chunk_map[chunk_id]
            result.score = rrf_scores[chunk_id]
            final_results.append(result)
        
        return final_results


# ==============================================================================
# TEST DIRECT
# ==============================================================================
if __name__ == "__main__":
    engine = SearchEngine()
    
    # Exemple de requête
    query = "What was Tesla's revenue in Q1 2024?"
    
    print(f"\n{'='*60}")
    print(f" Requête : {query}")
    print(f"{'='*60}\n")
    
    # Recherche hybride
    results = engine.search_hybrid(query, top_k=5)
    
    print(f"📊 Top {len(results)} résultats :\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.company} - {result.period}] (Score RRF: {result.score:.4f})")
        print(f"   Type: {result.document_type}")
        print(f"   Source: {result.source}")
        print(f"   Texte: {result.text[:200]}...")
        print()