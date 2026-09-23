# src/rag/search_engine.py
import logging
import re
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
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
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",   
        device: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.es_index_name = es_index_name
        self.device = device

        # 1. Embedding model
        logger.info(f"🧠 Chargement embedding : {model_name}")
        self.embedding_model = SentenceTransformer(
            model_name, device=self.device
        )

        # 2. Reranker (cross-encoder)
        logger.info(f"🎯 Chargement reranker : {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, device=self.device)

        # 3. Milvus
        logger.info("🔌 Connexion à Milvus...")
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.collection = Collection(self.collection_name)
        logger.info(f"✅ Collection '{self.collection_name}' chargée")

        # 4. Elasticsearch
        logger.info("🔌 Connexion à Elasticsearch...")
        self.es = Elasticsearch(es_host)
        if self.es.ping():
            logger.info("✅ Elasticsearch connecté")
        else:
            logger.warning("⚠️ Elasticsearch non disponible")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """
        Encode un texte avec BGE-large.

        ⚠️ Préfixe OBLIGATOIRE pour les requêtes, INTERDIT pour les passages.
        """
        if is_query:
            text = (
                "Represent this sentence for searching relevant passages: " + text
            )
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    # ------------------------------------------------------------------
    # Recherche sémantique (Milvus)
    # ------------------------------------------------------------------
    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        query_embedding = self._get_embedding(query, is_query=True)

        # Filtre Milvus
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

        # ✅ Filtre amélioré : garde le filtre uniquement si la query cible des chiffres
        if re.search(
            r"(revenue|income|cash|earnings|profit|assets|liabilities|revenues)",
            query,
            re.IGNORECASE,
        ):
            search_results = [
                r for r in search_results if re.search(r"\d{2,}", r["text"])
            ]
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
        cleaned_query = re.sub(r"[\u2018\u2019']s\b", "", query)   # Alphabet's → Alphabet
        cleaned_query = re.sub(r"[\u2018\u2019']", " ", cleaned_query)
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
        filter_clauses = []
        if filters:
            if "company" in filters:
                filter_clauses.append({"term": {"company": filters["company"]}})          # sans .keyword
            if "period" in filters:
                filter_clauses.append({"term": {"period": filters["period"]}})            # sans .keyword
            if "document_type" in filters:
                filter_clauses.append({"term": {"document_type": filters["document_type"]}})  # sans .keyword

        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": {"query": cleaned_query, "operator": "or"}}}
                    ],
                    "should": [
                        # Boost sur les indicateurs de tableaux financiers
                        {"match_phrase": {"text": {"query": "Total revenue", "boost": 3.0}}},
                        {"match_phrase": {"text": {"query": "Total revenues", "boost": 3.0}}},
                        {"match_phrase": {"text": {"query": "Total net sales", "boost": 3.0}}},
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
    # Fusion RRF
    # ------------------------------------------------------------------
    @staticmethod
    def _rrf_fusion(
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
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

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        merged = []
        for cid in sorted_ids:
            doc = docs_by_id[cid]
            doc["rrf_score"] = scores[cid]
            merged.append(doc)
        return merged

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    def _rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        if not docs:
            return []

        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        if docs:
            best = docs[0]["rerank_score"]
            threshold = best * 0.3   # ✅ seuil relatif (50% du meilleur)
            filtered = [d for d in docs if d["rerank_score"] >= threshold]
        else:
            filtered = []

        if len(filtered) < len(docs):
            logger.info(
                f"   → Rerank : {len(docs) - len(filtered)} chunks filtrés "
                f"(threshold = {threshold:.3f})"
            )

        if not filtered:
            logger.warning("⚠️ Fallback : tous les chunks filtrés, top 2 conservés")
            return docs[:2]

        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Pipeline hybride complet
    # ------------------------------------------------------------------
    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        use_reranker: bool = True,
    ) -> List[Dict]:
        candidates_k = top_k * 4

        logger.info(f"🔍 Recherche hybride : '{query}'")
        dense = self.search_semantic(query, top_k=candidates_k, filters=filters)
        sparse = self.search_lexical(query, top_k=candidates_k, filters=filters)
        logger.info(f"   → {len(dense)} dense, {len(sparse)} lexical")

        merged = self._rrf_fusion(dense, sparse)
        logger.info(f"   → {len(merged)} après RRF")

        if use_reranker:
            final = self._rerank(query, merged, top_k=top_k)
            logger.info(f"   → {len(final)} après reranking")
        else:
            final = merged[:top_k]

        return final


# ==============================================================================
# TEST DIAGNOSTIC (sans LLM)
# ==============================================================================
if __name__ == "__main__":
    engine = SearchEngine()

    query = "Tesla revenue 2024"
    print(f"\n{'=' * 70}")
    print(f"  QUERY : {query}")
    print(f"{'=' * 70}\n")

    print("--- DENSE (Milvus) ---")
    for d in engine.search_semantic(query, top_k=5):
        print(f"  {d['score']:.4f} | {d['period']:<10} | {d['text'][:90]}")

    print("\n--- SPARSE (BM25) ---")
    for d in engine.search_lexical(query, top_k=5):
        print(f"  {d['score']:.2f} | {d['period']:<10} | {d['text'][:90]}")

    print("\n--- HYBRID (final) ---")
    for d in engine.search_hybrid(query, top_k=5):
        print(
            f"  rerank={d.get('rerank_score', 0):.3f} | {d['period']:<10} | "
            f"{d['text'][:90]}"
        )