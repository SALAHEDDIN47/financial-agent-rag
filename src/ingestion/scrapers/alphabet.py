# src/ingestion/scrapers/alphabet.py
"""
Scraper Alphabet (abc.xyz) — télécharge les PDFs investor et uploade vers MinIO.

Destination MinIO : raw-documents/presentations/alphabet/{filename}.pdf
Idempotent : skip si la clé existe déjà dans MinIO.
Playwright en headless=True (obligatoire dans Docker).
"""
import logging
from pathlib import Path

from playwright.sync_api import sync_playwright

from src.core.storage import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET_PREFIX = "raw-documents/presentations/alphabet"
URL = "https://abc.xyz/investor/earnings/"


def scrape_alphabet_pdfs() -> None:
    logger.info(f"🔍 Scraper Alphabet → {BUCKET_PREFIX}/")
    logger.info(f"   URL : {URL}")

    downloaded = skipped = failed = 0

    with sync_playwright() as p:
        # ⚠️ headless=True OBLIGATOIRE en Docker (pas d'écran)
        browser = p.chromium.launch(headless=True, slow_mo=200)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            accept_downloads=True,
        )
        page = context.new_page()

        try:
            logger.info("⏳ Chargement de la page...")
            page.goto(URL, wait_until="networkidle", timeout=60000)

            logger.info("🔍 Analyse des liens...")
            all_links = page.locator("a[href]").evaluate_all(
                "elements => elements.map(e => e.href)"
            )
            pdf_links = list({link for link in all_links if "pdf" in link.lower()})
            logger.info(f"✅ {len(pdf_links)} PDFs potentiels trouvés")

            for idx, pdf_url in enumerate(pdf_links):
                try:
                    filename = pdf_url.split("/")[-1].split("?")[0]
                    if not filename.lower().endswith(".pdf"):
                        filename = f"alphabet_report_{idx}.pdf"

                    key = f"{BUCKET_PREFIX}/{filename}"

                    if storage.exists(key):
                        logger.info(f"  ⏭️  Déjà dans MinIO : {filename}")
                        skipped += 1
                        continue

                    logger.info(f"  ⬇️  {filename}...")
                    response = context.request.get(pdf_url)

                    if response.ok:
                        storage.write_bytes(key, response.body())
                        logger.info(f"  ✅ {filename} → MinIO")
                        downloaded += 1
                    else:
                        logger.warning(
                            f"  ⚠️  HTTP {response.status} pour {pdf_url}"
                        )
                        failed += 1

                except Exception as e:
                    logger.error(f"  ❌ Erreur sur {pdf_url} : {e}")
                    failed += 1

        except Exception as e:
            logger.error(f"❌ Erreur scraping : {e}")
        finally:
            browser.close()

    logger.info(
        f"\n🎉 Téléchargés={downloaded} | Ignorés={skipped} | Échecs={failed}"
    )


if __name__ == "__main__":
    scrape_alphabet_pdfs()