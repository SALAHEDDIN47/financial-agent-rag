# src/rag/search_engine.py
import logging
from typing import List, Dict, Optional
from fastembed import TextEmbedding
from pymilvus import connections, Collection
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder
import re

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
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.collection_name = collection_name
        self.es_index_name = es_index_name

        # 1. Modèle d'embedding (fastembed)
        logger.info(f"🧠 Chargement du modèle d'embedding : {model_name}")
        self.embedding_model = TextEmbedding(model_name=model_name)

        # 2. Reranker (cross-encoder)
        logger.info(f"🎯 Chargement du reranker : {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)

        # 3. Connexion Milvus
        logger.info("🔌 Connexion à Milvus...")
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.collection = Collection(self.collection_name)
        logger.info(f"✅ Collection '{self.collection_name}' chargée")

        # 4. Connexion Elasticsearch
        logger.info("🔌 Connexion à Elasticsearch...")
        self.es = Elasticsearch(es_host)
        if self.es.ping():
            logger.info("✅ Elasticsearch connecté")
        else:
            logger.warning("⚠️ Elasticsearch non disponible")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _get_embedding(self, text: str) -> List[float]:
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()

    # ------------------------------------------------------------------
    # Recherche sémantique (Milvus)
    # ------------------------------------------------------------------
    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        query_embedding = self._get_embedding(query)

        # Construction du filtre Milvus (expression booléenne)
        milvus_expr = None
        if filters:
            conditions = []
            if "company" in filters:
                conditions.append(f'company == "{filters["company"]}"')
            if "period" in filters:
                conditions.append(f'period == "{filters["period"]}"')
            if "document_type" in filters:
                conditions.append(f'document_type == "{filters["document_type"]}"')
            if conditions:
                milvus_expr = " and ".join(conditions)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=milvus_expr,
            output_fields=["chunk_id", "text", "company", "period", "document_type"],
        )

        search_results = []
        for hits in results:
            for rank, hit in enumerate(hits):
                search_results.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "text": hit.entity.get("text"),
                    "company": hit.entity.get("company"),
                    "period": hit.entity.get("period"),
                    "document_type": hit.entity.get("document_type"),
                    "score": hit.score,
                    "rank_dense": rank,
                })
        search_results = [r for r in search_results if re.search(r'\d{2,}', r["text"])]
        return search_results

    # ------------------------------------------------------------------
    # Recherche lexicale (Elasticsearch)
    # ------------------------------------------------------------------
    def search_lexical(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        # Construction de la requête ES avec filtres
        must_clause = {"match": {"text": query}}
        filter_clauses = []
        if filters:
            if "company" in filters:
                filter_clauses.append({"term": {"company.keyword": filters["company"]}})
            if "period" in filters:
                filter_clauses.append({"term": {"period.keyword": filters["period"]}})
            if "document_type" in filters:
                filter_clauses.append({"term": {"document_type.keyword": filters["document_type"]}})

        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "match": {"text": query}
                    },
                    "should": [
                        # Booste les chunks contenant des indicateurs de tableaux
                        {"match_phrase": {"text": "Total revenue"}},
                        {"match_phrase": {"text": "Total revenues"}},
                        {"match_phrase": {"text": "Total net sales"}},
                        {"match_phrase": {"text": "In millions"}},
                        {"match_phrase": {"text": "Year Ended"}},
                        {"match_phrase": {"text": "Fiscal Year"}},
                        {"match_phrase": {"text": "CONSOLIDATED STATEMENTS OF OPERATIONS"}},
                        {"match_phrase": {"text": "INCOME STATEMENTS"}},
                    ],
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
        }

        response = self.es.search(index=self.es_index_name, body=query_body)

        search_results = []
        for rank, hit in enumerate(response["hits"]["hits"]):
            source = hit["_source"]
            search_results.append({
                "chunk_id": source.get("chunk_id"),
                "text": source.get("text"),
                "company": source.get("company"),
                "period": source.get("period"),
                "document_type": source.get("document_type"),
                "score": hit["_score"],
                "rank_sparse": rank,
            })
        return search_results

    # ------------------------------------------------------------------
    # Fusion RRF (Reciprocal Rank Fusion)
    # ------------------------------------------------------------------
    @staticmethod
    def _rrf_fusion(
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Fusionne deux listes de résultats avec RRF."""
        scores = {}
        docs_by_id = {}

        for rank, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            docs_by_id[cid] = doc

        for rank, doc in enumerate(sparse_results):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            if cid not in docs_by_id:
                docs_by_id[cid] = doc

        # Trier par score RRF décroissant
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        merged = []
        for cid in sorted_ids:
            doc = docs_by_id[cid]
            doc["rrf_score"] = scores[cid]
            merged.append(doc)
        return merged

    # ------------------------------------------------------------------
    # Reranking (cross-encoder)
    # ------------------------------------------------------------------
    def _rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        if not docs:
            return []
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # ✅ FILTRE : on ignore les chunks non pertinents (score < 0)
        filtered = [d for d in docs if d["rerank_score"] > 0]
        
        # Si tous les chunks sont filtrés, on retourne au moins les 2 meilleurs
        if not filtered:
            logger.warning("⚠️ Aucun chunk avec score positif, on garde les 2 meilleurs")
            return docs[:2]
        
        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Recherche hybride complète
    # ------------------------------------------------------------------
    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        use_reranker: bool = True,
    ) -> List[Dict]:
        """
        Pipeline complet :
        1. Recherche dense (Milvus)
        2. Recherche lexicale (Elasticsearch)
        3. Fusion RRF
        4. Reranking (cross-encoder)
        """
        # On récupère plus de candidats pour le reranker
        candidates_k = top_k * 4

        logger.info(f"🔍 Recherche hybride pour : '{query}'")
        dense = self.search_semantic(query, top_k=candidates_k, filters=filters)
        sparse = self.search_lexical(query, top_k=candidates_k, filters=filters)
        logger.info(f"   → {len(dense)} résultats denses, {len(sparse)} résultats lexicaux")

        # Fusion RRF
        merged = self._rrf_fusion(dense, sparse)
        logger.info(f"   → {len(merged)} résultats après fusion RRF")

        # Reranking
        if use_reranker:
            final = self._rerank(query, merged, top_k=top_k)
            logger.info(f"   → {len(final)} résultats après reranking")
        else:
            final = merged[:top_k]

        return final


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    engine = SearchEngine()
    query = "What was Tesla's revenue in 2024?"
    print(f"\n🔍 Recherche : {query}\n")
    results = engine.search_hybrid(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['company']} - {r['period']}] "
              f"(RRF: {r.get('rrf_score', 0):.4f} | Rerank: {r.get('rerank_score', 0):.4f})")
        print(f"   Type: {r['document_type']}")
        print(f"   Texte: {r['text'][:200]}...\n")