# src/ingestion/collectors/collect_finqa.py
"""
Collector FinQA — télécharge train/dev/test.json et les upload vers MinIO.

Destination MinIO : raw-documents/finqa/{filename}
Idempotent : skip si le fichier existe déjà dans MinIO.
"""
import logging

import requests

from src.core.storage import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FINQA_URLS = {
    "train.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}

BUCKET_PREFIX = "raw-documents/finqa"


def download_finqa_dataset() -> None:
    logger.info(f"📥 Collector FinQA → {BUCKET_PREFIX}/")

    for filename, url in FINQA_URLS.items():
        key = f"{BUCKET_PREFIX}/{filename}"

        if storage.exists(key):
            logger.info(f"  ℹ️  {filename} existe déjà dans MinIO")
            continue

        logger.info(f"  ⏳ Téléchargement {filename}...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            storage.write_bytes(key, response.content)
            size_kb = len(response.content) / 1024
            logger.info(f"  ✅ {filename} uploadé ({size_kb:.1f} KB)")
        except requests.RequestException as e:
            logger.error(f"  ❌ Échec {filename} : {e}")


if __name__ == "__main__":
    download_finqa_dataset()