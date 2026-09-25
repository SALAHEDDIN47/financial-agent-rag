# src/ingestion/collectors/collect_transcripts.py
"""
Collector Transcripts — parcourt le dataset HuggingFace en streaming
et upload les transcripts vers MinIO.

Destination MinIO : raw-documents/transcripts/{TICKER}_Q{N}_{YEAR}_transcript.json
Idempotent : skip si le fichier existe déjà dans MinIO.
"""
import argparse
import json
import logging

from datasets import load_dataset

from src.core.storage import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET_PREFIX = "raw-documents/transcripts"
DATASET_NAME = "Rogersurf/earnings-call-transcripts"


def download_transcripts(
    tickers: list[str],
    limit: int = 2,
    force: bool = False,
) -> None:
    logger.info(f"📥 Collector Transcripts → {BUCKET_PREFIX}/")
    logger.info(f"   Tickers : {tickers} | limit={limit}/ticker")

    logger.info(f"  ⏳ Chargement dataset {DATASET_NAME} (streaming)...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    counts = {t: 0 for t in tickers}
    saved = 0

    for row in dataset:
        ticker = row.get("ticker")
        if ticker not in tickers or counts[ticker] >= limit:
            # On saute si le ticker n'est pas demandé ou qu'on a atteint le quota
            if all(c >= limit for c in counts.values()):
                break
            continue

        quarter = row.get("quarter")
        year = row.get("earnings_year")
        q = str(row.get("quarter", "")).lstrip("Q") or "?"
        filename = f"{ticker}_Q{q}_{year}_transcript.json"
        key = f"{BUCKET_PREFIX}/{filename}"

        if not force and storage.exists(key):
            logger.info(f"  ℹ️  {filename} déjà dans MinIO")
            counts[ticker] += 1
            continue

        payload = {
            "ticker": ticker,
            "quarter": quarter,
            "year": year,
            "transcript": row.get("transcript"),
        }
        storage.write_bytes(
            key,
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        logger.info(f"  ✅ {filename} uploadé")
        counts[ticker] += 1
        saved += 1

        if all(c >= limit for c in counts.values()):
            break

    logger.info(f"✅ {saved} nouveaux transcripts uploadés")
    logger.info(f"   Compteurs : {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA"])
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    download_transcripts(args.tickers, limit=args.limit, force=args.force)