# src/ingestion/collectors/collect_news.py
"""
Collector News — récupère les actualités Yahoo Finance et les upload vers MinIO.

Destination MinIO : raw-documents/news/{TICKER}_news.json
Idempotent : skip si le fichier existe déjà (ajouter --force pour écraser).
"""
import argparse
import json
import logging
import time

import yfinance as yf

from src.core.storage import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET_PREFIX = "raw-documents/news"


def download_news(tickers: list[str], force: bool = False) -> None:
    logger.info(f"📥 Collector News → {BUCKET_PREFIX}/")
    logger.info(f"   Tickers : {tickers}")

    for ticker in tickers:
        key = f"{BUCKET_PREFIX}/{ticker}_news.json"

        if not force and storage.exists(key):
            logger.info(f"  ℹ️  {ticker} déjà présent dans MinIO")
            continue

        logger.info(f"  ⏳ Récupération actualités {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
        except Exception as e:
            logger.error(f"  ❌ Échec API Yahoo pour {ticker} : {e}")
            continue

        if not news:
            logger.warning(f"  ⚠️  Aucune actualité pour {ticker}")
            continue

        payload = json.dumps(news, ensure_ascii=False, indent=2).encode("utf-8")
        storage.write_bytes(key, payload)
        logger.info(f"  ✅ {ticker} uploadé ({len(news)} articles)")

        # Rate limit pour éviter le ban Yahoo
        time.sleep(1.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "MSFT", "GOOGL"])
    ap.add_argument("--force", action="store_true", help="Réécrase si déjà présent")
    args = ap.parse_args()

    download_news(args.tickers, force=args.force)