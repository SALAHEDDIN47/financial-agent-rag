"""
spark_jobs/chunk_distributed.py

Job Spark distribué : lit tous les parsed-documents/*.json,
chunk chaque document, écrit dans processed-chunks/*.jsonl.

Réutilise FinancialChunker (src/ingestion/chunking/financial_chunker.py)
via une UDF Spark.

Usage :
    spark-submit \
        --master spark://spark-master:7077 \
        /app/spark_jobs/chunk_distributed.py \
        --prefix parsed-documents/ \
        --partitions 8
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, StructField, StructType

sys.path.insert(0, "/app")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# UDF : exécutée sur chaque worker Spark
# =============================================================================
def _chunk_one(source_key: str) -> dict:
    """Lit un parsed JSON, le chunk, écrit le JSONL dans MinIO."""
    from src.core.storage import storage, compute_key_hash
    from src.ingestion.chunking.financial_chunker import FinancialChunker

    result = {
        "source_key": source_key,
        "status": "unknown",
        "output_key": "",
        "n_chunks": 0,
        "error": "",
    }

    try:
        # 1. Lire le document parsé
        doc = storage.read_json(source_key)

        # 2. Chunker
        chunker = FinancialChunker(max_tokens=500, overlap_tokens=80)
        chunks = chunker.chunk_document(doc)

        if not chunks:
            result["status"] = "empty"
            return result

        # 3. Calculer la clé de sortie
        # source_key = "parsed-documents/tsla/10q/TSLA-Q4-2025-Update.json"
        # output_key = "processed-chunks/tsla/10q/TSLA-Q4-2025-Update_chunks.jsonl"
        rel = source_key.replace("parsed-documents/", "")
        stem = Path(rel).with_suffix("")
        output_key = f"processed-chunks/{stem}_chunks.jsonl"

        # 4. Écrire en JSONL
        jsonl = "\n".join(
            json.dumps(c, ensure_ascii=False) for c in chunks
        )
        storage.write_text(output_key, jsonl)

        result["status"] = "chunked"
        result["output_key"] = output_key
        result["n_chunks"] = len(chunks)
        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:200]
        return result


_chunk_udf = udf(
    _chunk_one,
    StructType([
        StructField("source_key", StringType(), False),
        StructField("status", StringType(), False),
        StructField("output_key", StringType(), True),
        StructField("n_chunks", StringType(), True),
        StructField("error", StringType(), True),
    ]),
)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="parsed-documents/")
    ap.add_argument("--partitions", type=int, default=8)
    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("financial-rag-chunk-distributed")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Lister tous les JSON parsés
    from src.core.storage import storage
    source_keys = [
        k for k in storage.list(args.prefix)
        if k.lower().endswith(".json") and "_chunks" not in k
    ]
    logger.info(f"📂 {len(source_keys)} documents à chunker")

    # 2. Créer le DataFrame distribué
    df = spark.createDataFrame(
        [(k,) for k in source_keys], ["source_key"]
    ).repartition(args.partitions)

    # 3. Lancer le chunking distribué
    df_result = df.withColumn("result", _chunk_udf(col("source_key")))
    df_flat = df_result.select(
        col("result.source_key").alias("source_key"),
        col("result.status").alias("status"),
        col("result.output_key").alias("output_key"),
        col("result.n_chunks").alias("n_chunks"),
        col("result.error").alias("error"),
    )

    # 4. Résumé
    df_flat.groupBy("status").count().show()

    # 5. Total chunks
    total_chunks = df_flat.agg({"n_chunks": "sum"}).collect()[0][0]
    logger.info(f"📊 Total chunks générés : {total_chunks}")

    # 6. Erreurs
    errors = df_flat.filter(col("status") == "failed").limit(10).collect()
    if errors:
        logger.warning(f"⚠️  {len(errors)} erreurs (sur les 10 premières) :")
        for e in errors:
            logger.warning(f"   {e['source_key']} : {e['error']}")

    spark.stop()
    logger.info("✅ Job chunking terminé")


if __name__ == "__main__":
    main()