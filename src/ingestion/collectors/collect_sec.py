# src/ingestion/collectors/collect_sec.py
"""
Collector SEC (10-K, 10-Q, etc.) — télécharge et upload vers MinIO.

Destination MinIO : raw-documents/sec/{TICKER}/{TYPE}/{ACCESSION}/...
Idempotent : skip si le fichier existe déjà dans MinIO (mode --force pour forcer).
"""
import argparse
import logging
import os
import tempfile
from pathlib import Path

from sec_edgar_downloader import Downloader

from src.core.storage import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET_PREFIX = "raw-documents/sec"

# La SEC exige un User-Agent propre (company + email).
# ⚠️  Change l'email pour un VRAI email, sinon blocage IP.
COMPANY = os.getenv("SEC_COMPANY", "Financial RAG Project")
EMAIL = os.getenv("SEC_USER_AGENT", "salah@example.com")


def _upload_tree(local_root: Path, s3_prefix: str, skip_existing: bool = True) -> int:
    n = 0
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root).as_posix()
        key = f"{s3_prefix}/{rel}"
        if skip_existing and storage.exists(key):
            continue
        storage.write_bytes(key, path.read_bytes())
        n += 1
    return n


def download_sec_reports(
    tickers: list[str],
    doc_type: str = "10-K",
    limit: int = 1,
    force: bool = False,
) -> None:
    logger.info(f"📥 Collector SEC ({doc_type}) → {BUCKET_PREFIX}/")
    logger.info(f"   Tickers : {tickers} | limit={limit}")
    logger.info(f"   User-Agent SEC : {COMPANY} <{EMAIL}>")

    with tempfile.TemporaryDirectory(prefix=f"sec_{doc_type}_") as tmp:
        tmp_root = Path(tmp)
        dl = Downloader(COMPANY, EMAIL, tmp_root)

        for ticker in tickers:
            logger.info(f"  ⏳ Téléchargement {doc_type} pour {ticker}...")
            try:
                dl.get(doc_type, ticker, limit=limit)
                logger.info(f"  ✅ {ticker} téléchargé en local")
            except Exception as e:
                logger.error(f"  ❌ Échec {ticker} : {e}")

        n = _upload_tree(tmp_root, BUCKET_PREFIX, skip_existing=not force)
        logger.info(f"✅ {n} nouveaux fichiers uploadés vers MinIO")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "GOOGL", "MSFT"])
    ap.add_argument("--type", dest="doc_type", default="10-K",
                    choices=["10-K", "10-Q", "8-K", "20-F"])
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    download_sec_reports(
        tickers=args.tickers,
        doc_type=args.doc_type,
        limit=args.limit,
        force=args.force,
    )