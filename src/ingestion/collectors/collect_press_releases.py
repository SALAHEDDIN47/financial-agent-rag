# src/ingestion/collectors/collect_press_releases.py
"""
Collector 8-K (communiqués SEC) — télécharge et upload vers MinIO.

Destination MinIO : raw-documents/sec/{TICKER}/8-K/{ACCESSION}/...
Idempotent : skip si le dossier du ticker existe déjà (mode --force pour forcer).
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

# La SEC exige un User-Agent propre (company + email)
COMPANY = "Financial RAG Project"
EMAIL = os.getenv("SEC_USER_AGENT", "salah@example.com")


def _upload_tree(local_root: Path, s3_prefix: str, skip_existing: bool = True) -> int:
    """Upload récursif local_root/** → s3_prefix/**. Retourne le nb de fichiers."""
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


def download_8k(tickers: list[str], limit: int = 3, force: bool = False) -> None:
    logger.info(f"📥 Collector 8-K → {BUCKET_PREFIX}/")
    logger.info(f"   Tickers : {tickers} | limit={limit}")

    with tempfile.TemporaryDirectory(prefix="sec_8k_") as tmp:
        tmp_root = Path(tmp)
        dl = Downloader(COMPANY, EMAIL, tmp_root)

        for ticker in tickers:
            logger.info(f"  ⏳ Téléchargement {limit} × 8-K pour {ticker}...")
            try:
                dl.get("8-K", ticker, limit=limit)
            except Exception as e:
                logger.error(f"  ❌ Échec {ticker} : {e}")
                continue

        # Upload vers MinIO
        n = _upload_tree(tmp_root, BUCKET_PREFIX, skip_existing=not force)
        logger.info(f"✅ {n} nouveaux fichiers uploadés vers MinIO")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA"])
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    download_8k(args.tickers, limit=args.limit, force=args.force)