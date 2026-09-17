from pathlib import Path
import time
from curl_cffi import requests

def scrape_tesla_presentations(output_dir: str = "data/raw/presentations/tesla"):
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Téléchargement des présentations Tesla...")
    
    quarters = ["Q2-2026", "Q1-2026", "Q4-2025", "Q3-2025", "Q2-2025", "Q1-2025", "Q4-2024", "Q3-2024", "Q2-2024", "Q1-2024"]
    
    # Domaines possibles à tester en cas d'échec
    domains = [
        "https://assets-ir.tesla.com/tesla-contents/IR/",
        "https://digitalassets.tesla.com/tesla-contents/image/upload/IR/"
    ]
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    session = requests.Session()

    for quarter in quarters:
        file_name = f"TSLA-{quarter}-Update.pdf"
        file_path = target_path / file_name
        
        if file_path.exists():
            print(f"⏭️ Déjà présent : {file_name}")
            skipped_count += 1
            continue
        
        success = False
        print(f"⬇️ Téléchargement : {file_name}...")
        
        for base_url in domains:
            pdf_url = f"{base_url}{file_name}"
            try:
                response = session.get(
                    pdf_url, 
                    impersonate="chrome120", 
                    headers={"Referer": "https://ir.tesla.com/"},
                    timeout=20
                )
                
                # Vérification de l'en-tête binaire du PDF (%PDF)
                if response.status_code == 200 and response.content.startswith(b'%PDF'):
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Sauvegardé depuis {pdf_url.split('/')[2]} : {file_name}")
                    downloaded_count += 1
                    success = True
                    break
            except Exception:
                continue
        
        if not success:
            print(f"⚠️ Échec sur tous les domaines : {file_name} (fichier non valide ou URL modifiée)")
            failed_count += 1

        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"🎉 Bilan final :")
    print(f"   ✅ Téléchargés : {downloaded_count}")
    print(f"   ⏭️ Ignorés : {skipped_count}")
    print(f"   ⚠️ Indisponibles : {failed_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    scrape_tesla_presentations()
    