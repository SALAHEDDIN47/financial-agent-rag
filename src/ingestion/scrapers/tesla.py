from pathlib import Path
import requests

def scrape_tesla_presentations(output_dir: str = "data/raw/presentations/tesla"):
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Téléchargement direct des présentations Tesla...")
    
    # Mapping précis des URLs en fonction du changement de domaine de Tesla
    # Les rapports 2024 et antérieurs sont sur digitalassets.tesla.com
    # Les rapports 2025+ sont sur assets-ir.tesla.com
    quarters_config = [
        ("Q2-2026", "assets-ir.tesla.com"),
        ("Q1-2026", "assets-ir.tesla.com"),
        ("Q4-2025", "assets-ir.tesla.com"),
        ("Q3-2025", "assets-ir.tesla.com"),
        ("Q2-2025", "assets-ir.tesla.com"),
        ("Q1-2025", "assets-ir.tesla.com"),
        ("Q4-2024", "digitalassets.tesla.com"),  # ⚠️ Ancien domaine
        ("Q3-2024", "digitalassets.tesla.com"),  # ⚠️ Ancien domaine
        ("Q2-2024", "digitalassets.tesla.com"),  # ⚠️ Ancien domaine
        ("Q1-2024", "digitalassets.tesla.com"),  # ⚠️ Ancien domaine
    ]
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Headers réalistes pour éviter tout blocage CDN (Akamai/Cloudflare)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Accept": "application/pdf,*/*",
        "Referer": "https://ir.tesla.com/",
        "Origin": "https://ir.tesla.com",
    }

    for quarter, domain in quarters_config:
        file_name = f"TSLA-{quarter}-Update.pdf"
        file_path = target_path / file_name
        
        if file_path.exists():
            print(f"⏭️ Déjà présent : {file_name}")
            skipped_count += 1
            continue
        
        # Construction de l'URL en fonction du domaine correct
        if domain == "digitalassets.tesla.com":
            pdf_url = f"https://{domain}/tesla-contents/image/upload/IR/{file_name}"
        else:
            pdf_url = f"https://{domain}/tesla-contents/IR/{file_name}"
            
        try:
            print(f"⬇️ Téléchargement : {file_name}...")
            
            response = requests.get(pdf_url, headers=headers, timeout=30)
            
            # Vérification stricte : code 200 ET contenu de type PDF
            if response.status_code == 200 and 'application/pdf' in response.headers.get('content-type', ''):
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Sauvegardé : {file_name}")
                downloaded_count += 1
            elif response.status_code == 404:
                print(f"⚠️ Non trouvé (404) : {file_name} (n'existe peut-être pas sous ce format)")
                failed_count += 1
            else:
                print(f"⚠️ Échec (HTTP {response.status_code}) : {file_name}")
                failed_count += 1
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur réseau sur {file_name}: {str(e)[:80]}")
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"🎉 Mission accomplie !")
    print(f"   ✅ Nouveaux fichiers téléchargés : {downloaded_count}")
    print(f"   ⏭️ Fichiers déjà présents (ignorés) : {skipped_count}")
    print(f"   ⚠️ Échecs / Indisponibles : {failed_count}")
    print(f"   📁 Dossier de destination : {target_path.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    scrape_tesla_presentations()