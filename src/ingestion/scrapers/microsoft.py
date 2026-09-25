# src/ingestion/scrapers/microsoft.py
"""
Scraper Microsoft (microsoft.com/investor) — télécharge les annual reports
(PDF ou DOCX) et uploade vers MinIO.

Destination MinIO : raw-documents/presentations/microsoft/{filename}
Idempotent : skip si la clé existe déjà dans MinIO.
Playwright en headless=True (obligatoire dans Docker).
"""
import logging

from playwright.sync_api import sync_playwright

from src.core.storage import storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET_PREFIX = "raw-documents/presentations/microsoft"
YEARS = [21, 22, 23, 24, 25]


def scrape_microsoft_annual_reports() -> None:
    logger.info(f"🔍 Scraper Microsoft → {BUCKET_PREFIX}/")
    logger.info(f"   Années : {['20' + str(y) for y in YEARS]}")

    downloaded = skipped = failed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=200)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            accept_downloads=True,
        )
        page = context.new_page()

        for year in YEARS:
            url = f"https://www.microsoft.com/investor/reports/ar{year}/download-center/index.html"
            logger.info(f"\n🔍 AR20{year} : {url}")

            try:
                response = page.goto(url, wait_until="networkidle", timeout=30000)
                if response and response.status == 404:
                    logger.warning(f"  ⚠️  Page 404 pour AR20{year}")
                    continue

                all_links = page.locator("a[href]").evaluate_all(
                    "elements => elements.map(e => e.href)"
                )
                target_links = list({
                    link for link in all_links
                    if "cdn-dynmedia" in link.lower()
                    or "microsoftcorp" in link.lower()
                })
                logger.info(f"  ✅ {len(target_links)} documents potentiels")

                for doc_url in target_links:
                    try:
                        # HEAD pour identifier le content-type sans télécharger
                        head_resp = context.request.head(doc_url)
                        if not head_resp.ok:
                            continue

                        content_type = head_resp.headers.get(
                            "content-type", ""
                        ).lower()
                        is_pdf = "application/pdf" in content_type
                        is_docx = "wordprocessingml.document" in content_type

                        if not (is_pdf or is_docx):
                            continue

                        ext = ".docx" if is_docx else ".pdf"
                        base_name = doc_url.split("/")[-1].split("?")[0]
                        if base_name.lower().endswith(".pdf") and is_docx:
                            base_name = base_name[:-4]
                        filename = f"{base_name}{ext}"
                        key = f"{BUCKET_PREFIX}/{filename}"

                        if storage.exists(key):
                            logger.info(f"  ⏭️  Déjà dans MinIO : {filename}")
                            skipped += 1
                            continue

                        logger.info(f"  ⬇️  {filename}...")
                        get_resp = context.request.get(doc_url)
                        if get_resp.ok:
                            storage.write_bytes(key, get_resp.body())
                            logger.info(f"  ✅ {filename} → MinIO")
                            downloaded += 1
                        else:
                            logger.warning(
                                f"  ❌ HTTP {get_resp.status} pour {doc_url}"
                            )
                            failed += 1

                    except Exception as e:
                        logger.error(f"  ⚠️  Erreur sur {doc_url} : {e}")
                        failed += 1

            except Exception as e:
                logger.error(f"  ⚠️  Erreur AR20{year} : {e}")

        browser.close()

    logger.info(
        f"\n🎉 Téléchargés={downloaded} | Ignorés={skipped} | Échecs={failed}"
    )


if __name__ == "__main__":
    scrape_microsoft_annual_reports()