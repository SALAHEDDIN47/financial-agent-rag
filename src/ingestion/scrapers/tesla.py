# Ajouts en haut
import tempfile
from pathlib import Path
from src.core.storage import storage

BUCKET_PREFIX = "raw-documents/presentations/tesla"

def scrape_tesla_presentations():
    quarters = ["Q2-2026", "Q1-2026", "Q4-2025", "Q3-2025", "Q2-2025", "Q1-2025",
                "Q4-2024", "Q3-2024", "Q2-2024", "Q1-2024"]
    domains = [
        "https://assets-ir.tesla.com/tesla-contents/IR/",
        "https://digitalassets.tesla.com/tesla-contents/image/upload/IR/",
    ]
    session = requests.Session()

    downloaded = skipped = failed = 0

    for quarter in quarters:
        filename = f"TSLA-{quarter}-Update.pdf"
        key = f"{BUCKET_PREFIX}/{filename}"

        if storage.exists(key):
            print(f"⏭️  Déjà dans MinIO : {filename}")
            skipped += 1
            continue

        print(f"⬇️  {filename}...")
        for base_url in domains:
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
                    print(f"✅ {filename} → MinIO")
                    downloaded += 1
                    break
            except Exception:
                continue
        else:
            print(f"⚠️  {filename} indisponible")
            failed += 1

        time.sleep(1)

    print(f"\n🎉 Téléchargés={downloaded} | Ignorés={skipped} | Échecs={failed}")