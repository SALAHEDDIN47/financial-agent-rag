"""
spark_jobs/parse_distributed.py

Job Spark distribué : parse TOUS les fichiers de raw-documents/
et écrit dans parsed-documents/. Réutilise les parsers séquentiels
via des UDFs.

Usage :
    spark-submit \
        --master spark://spark-master:7077 \
        --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
        --conf spark.hadoop.fs.s3a.access.key=admin \
        --conf spark.hadoop.fs.s3a.secret.key=password123 \
        --conf spark.hadoop.fs.s3a.path.style.access=true \
        --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
        /app/spark_jobs/parse_distributed.py \
        --prefix raw-documents/ \
        --partitions 4
"""
import argparse
import logging
import sys
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, StructField, StructType

sys.path.insert(0, "/app")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# UDF : exécutée sur chaque worker Spark
# =============================================================================
def _parse_one(source_key: str) -> dict:
    """Parse un seul fichier, écrit dans MinIO, retourne un statut."""
    from src.core.storage import storage, compute_key_hash
    from src.ingestion.parsers.pdf_parser import PDFParser
    from src.ingestion.parsers.docx_parser import DOCXParser
    from src.ingestion.parsers.sec_parser import SECParser
    from src.ingestion.parsers.json_parser import JSONParser

    result = {
        "source_key": source_key,
        "status": "unknown",
        "output_key": "",
        "error": "",
    }

    try:
        # Dispatch selon l'extension / nom
        lower = source_key.lower()
        if lower.endswith(".pdf"):
            parser = PDFParser(preserve_tables=True, extract_layout=False)
            is_multi = False
        elif lower.endswith(".docx"):
            parser = DOCXParser()
            is_multi = False
        elif lower.endswith("full-submission.txt"):
            parser = SECParser()
            is_multi = False
        elif lower.endswith(".json"):
            parser = JSONParser()
            is_multi = True
        else:
            result["status"] = "skipped_unknown_ext"
            return result

        # Parse
        if is_multi:
            # JSONParser : multi-docs, utilise son propre wrapper
            manifest = {}  # pas de manifest partagé entre workers (S3 race)
            docs = parser.parse_file_and_save_to_storage(source_key, manifest)
            result["status"] = "parsed"
            result["output_key"] = f"{len(docs)} docs"
            return result
        else:
            # PDF / DOCX / SEC : single doc
            with storage.open_local_temp(source_key) as local_path:
                doc = parser.parse(local_path)

            output_key = parser._compute_output_key(source_key)
            source_hash = compute_key_hash(source_key)
            parsed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            doc.metadata["source_hash"] = source_hash
            doc.metadata["source_key"] = source_key
            doc.metadata["parsed_at"] = parsed_at

            storage.write_json(output_key, parser._doc_to_dict(doc))

            result["status"] = "parsed"
            result["output_key"] = output_key
            return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:200]
        return result


_parse_udf = udf(
    _parse_one,
    StructType([
        StructField("source_key", StringType(), False),
        StructField("status", StringType(), False),
        StructField("output_key", StringType(), True),
        StructField("error", StringType(), True),
    ]),
)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="raw-documents/")
    ap.add_argument("--partitions", type=int, default=4)
    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("financial-rag-parse-distributed")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Lister tous les fichiers à parser (driver-side, rapide)
    from src.core.storage import storage
    source_keys = [
        k for k in storage.list(args.prefix)
        if k.lower().endswith((".pdf", ".docx", ".json", "full-submission.txt"))
    ]
    logger.info(f"📂 {len(source_keys)} fichiers à parser")

    # 2. Créer le DataFrame distribué
    df = spark.createDataFrame(
        [(k,) for k in source_keys], ["source_key"]
    ).repartition(args.partitions)

    # 3. Lancer le parsing distribué
    df_result = df.withColumn("result", _parse_udf(col("source_key")))
    df_flat = df_result.select(
        col("result.source_key").alias("source_key"),
        col("result.status").alias("status"),
        col("result.output_key").alias("output_key"),
        col("result.error").alias("error"),
    )

    # 4. Résumé
    df_flat.groupBy("status").count().show()

    # 5. Afficher 5 échantillons
    df_flat.filter(col("status") == "parsed").show(5, truncate=False)

    # 6. Compter erreurs
    errors = df_flat.filter(col("status") == "failed").collect()
    if errors:
        logger.warning(f"⚠️  {len(errors)} erreurs :")
        for e in errors[:5]:
            logger.warning(f"   {e['source_key']} : {e['error']}")

    spark.stop()


if __name__ == "__main__":
    main()