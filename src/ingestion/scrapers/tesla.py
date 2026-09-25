# src/ingestion/scrapers/tesla.py
"""
Scraper Tesla (ir.tesla.com) — télécharge les quarterly update PDFs
et uploade vers MinIO.

Destination MinIO : raw-documents/presentations/tesla/{filename}.pdf
Idempotent : skip si la clé existe déjà dans MinIO.
"""
import logging
import time

from curl_cffi import requests

from src.core.storage import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET_PREFIX = "raw-documents/presentations/tesla"
QUARTERS = [
    "Q2-2026", "Q1-2026", "Q4-2025", "Q3-2025", "Q2-2025", "Q1-2025",
    "Q4-2024", "Q3-2024", "Q2-2024", "Q1-2024",
]
DOMAINS = [
    "https://assets-ir.tesla.com/tesla-contents/IR/",
    "https://digitalassets.tesla.com/tesla-contents/image/upload/IR/",
]


def scrape_tesla_presentations() -> None:
    logger.info(f"🔍 Scraper Tesla → {BUCKET_PREFIX}/")
    logger.info(f"   Quarters : {len(QUARTERS)}")

    session = requests.Session()
    downloaded = skipped = failed = 0

    for quarter in QUARTERS:
        filename = f"TSLA-{quarter}-Update.pdf"
        key = f"{BUCKET_PREFIX}/{filename}"

        if storage.exists(key):
            logger.info(f"  ⏭️  Déjà dans MinIO : {filename}")
            skipped += 1
            continue

        logger.info(f"  ⬇️  {filename}...")
        success = False
        for base_url in DOMAINS:
            url = f"{base_url}{filename}"
            try:
                r = session.get(
                    url,
                    impersonate="chrome120",
                    headers={"Referer": "https://ir.tesla.com/"},
                    timeout=30,
                )
                if r.status_code == 200 and r.content.startswith(b"%PDF"):
                    storage.write_bytes(key, r.content)
                    logger.info(f"  ✅ {filename} → MinIO")
                    downloaded += 1
                    success = True
                    break
            except Exception:
                continue

        if not success:
            logger.warning(f"  ⚠️  {filename} indisponible")
            failed += 1

        time.sleep(1)

    logger.info(
        f"\n🎉 Téléchargés={downloaded} | Ignorés={skipped} | Échecs={failed}"
    )


if __name__ == "__main__":
    scrape_tesla_presentations()