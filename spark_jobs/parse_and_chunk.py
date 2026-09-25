"""
spark_jobs/parse_and_chunk.py

Job Spark distribué : parse + chunk TOUS les formats supportés.

Supporte :
  - .pdf                    → PDFParser
  - .docx                   → DOCXParser
  - .json                   → JSONParser (multi-documents)
  - full-submission.txt     → SECParser

Usage :
    spark-submit \
        --master spark://spark-master:7077 \
        /app/spark_jobs/parse_and_chunk.py \
        --input /app/data/raw/ \
        --output /app/data/processed/chunks/

Sortie : JSONL (1 ligne = 1 chunk), compatible avec ton pipeline d'embedding.
"""
import argparse
import io
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, input_file_name
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

sys.path.insert(0, "/app")


# =============================================================================
# DISPATCH : choisit le bon parser selon l'extension / le nom
# =============================================================================
def _pick_parser(path: Path):
    """Retourne (parser, is_multi_doc) selon le fichier."""
    from src.ingestion.parsers.pdf_parser import PDFParser
    from src.ingestion.parsers.docx_parser import DOCXParser
    from src.ingestion.parsers.json_parser import JSONParser
    from src.ingestion.parsers.sec_parser import SECParser

    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".pdf":
        return PDFParser(preserve_tables=True), False
    if suffix == ".docx":
        return DOCXParser(), False
    if suffix == ".json":
        return JSONParser(), True          # multi-documents
    if suffix == ".txt" and "full-submission" in name:
        return SECParser(), False
    return None, False


# =============================================================================
# UDF : parse + chunk un fichier binaire, exécutée sur chaque worker
# =============================================================================
def _process_file(file_bytes: bytes, file_path: str) -> list[dict]:
    """Retourne une liste de chunks sérialisés (dicts)."""
    if not file_bytes:
        return []

    original_path = Path(file_path)
    parser, is_multi = _pick_parser(original_path)
    if parser is None:
        return []

    # On écrit les bytes dans un fichier temporaire car les parsers
    # existants lisent depuis le disque (Path). On garde le même suffixe
    # pour que `_validate_file` passe.
    tmp_dir = tempfile.mkdtemp(prefix="spark_parse_")
    tmp_path = Path(tmp_dir) / original_path.name
    try:
        tmp_path.write_bytes(file_bytes)

        # ---- 1) Parsing ----
        try:
            if is_multi:
                docs = parser.parse_file(tmp_path)
            else:
                docs = [parser.parse(tmp_path)]
        except Exception as e:  # noqa: BLE001
            return [{
                "chunk_id": str(uuid.uuid4()),
                "text": "",
                "metadata": f'{{"source":"{file_path}","error":"parse:{e}"}}',
            }]

        # ---- 2) Correction des métadonnées (company / period / file_path) ----
        # Les parsers ont lu le chemin temporaire → on réinjecte les bonnes valeurs
        parent_name = original_path.parent.name
        if parser.__class__.__name__ == "SECParser":
            parent_name = original_path.parent.parent.parent.name

        for doc in docs:
            try:
                meta = parser._extract_metadata_from_filename(  # noqa: SLF001
                    original_path.name, parent_name
                )
                doc.company = meta.get("company", doc.company)
                if doc.period in ("UNKNOWN", "", None):
                    doc.period = meta.get("period", doc.period)
            except Exception:  # noqa: BLE001
                pass
            doc.metadata["file_path"] = str(original_path)

        # ---- 3) Chunking ----
        from src.ingestion.chunking.financial_chunker import FinancialChunker
        chunker = FinancialChunker()

        out: list[dict] = []
        for doc in docs:
            try:
                chunks = chunker.chunk_document(doc)
            except Exception as e:  # noqa: BLE001
                continue
            for c in chunks:
                # On sérialise metadata en JSON string (Spark ne supporte pas
                # les dicts dans les types primitifs)
                import json as _json
                out.append({
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "metadata": _json.dumps({
                        "company": c.get("company"),
                        "period": c.get("period"),
                        "document_type": c.get("document_type"),
                        "source": c.get("source"),
                        "page": c.get("page"),
                        **c.get("metadata", {}),
                    }, ensure_ascii=False),
                })
        return out

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            os.rmdir(tmp_dir)
        except OSError:
            pass


_process_file_udf = udf(
    _process_file,
    ArrayType(
        StructType([
            StructField("chunk_id", StringType(), False),
            StructField("text", StringType(), True),
            StructField("metadata", StringType(), True),
        ])
    ),
)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Dossier source (local, s3a://, ou file://)")
    ap.add_argument("--output", required=True,
                    help="Dossier sortie JSONL (local ou s3a://)")
    ap.add_argument("--glob", default="*.{pdf,docx,json,txt}",
                    help="Filtre de fichiers (glob)")
    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("financial-rag-parse-chunk")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", "32m")
        .config("spark.sql.files.openCostInBytes", "4m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1) Lister tous les fichiers binaires récursivement
    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", args.glob)
        .option("recursiveFileLookup", "true")
        .load(args.input)
    )

    print(f"📂 Fichiers trouvés : {df.count()}")

    # 2) Parser + chunker (distribué)
    df_chunks = df.withColumn(
        "chunks", _process_file_udf(col("content"), input_file_name())
    ).selectExpr("explode(chunks) as chunk")

    # 3) Aplatir
    df_flat = df_chunks.select(
        col("chunk.chunk_id").alias("chunk_id"),
        col("chunk.text").alias("text"),
        col("chunk.metadata").alias("metadata"),
    ).filter(col("text") != "")

    # 4) Écrire en JSONL
    (
        df_flat
        .repartition(4)          # ajuster selon la taille
        .write
        .mode("overwrite")
        .json(args.output)
    )

    n = df_flat.count()
    print(f"✅ {n} chunks écrits vers {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()