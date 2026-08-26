# src/ingestion/scrapers/microsoft.py
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def scrape_microsoft_annual_reports(output_dir: str = "data/raw/presentations/microsoft"):
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    years_to_scrape = [21, 22, 23, 24, 25]
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=500)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            accept_downloads=True
        )
        page = context.new_page()
        
        for year in years_to_scrape:
            url = f"https://www.microsoft.com/investor/reports/ar{year}/download-center/index.html"
            print(f"\n🔍 Exploration de l'année 20{year} : {url}")
            
            try:
                response = page.goto(url, wait_until="networkidle", timeout=30000)
                if response and response.status == 404:
                    print(f"⚠️ Page introuvable pour 20{year}. Passage à la suivante.")
                    continue
                
                all_links = page.locator('a[href]').evaluate_all('elements => elements.map(e => e.href)')
                
                # On cible le CDN de Microsoft
                target_links = list(set([
                    link for link in all_links 
                    if 'cdn-dynmedia' in link.lower() or 'microsoftcorp' in link.lower()
                ]))
                
                print(f"✅ {len(target_links)} documents potentiels trouvés pour AR{year}.")
                
                for idx, doc_url in enumerate(target_links):
                    # Vérifier le type de contenu AVANT de télécharger
                    dl_resp = page.context.request.head(doc_url) # HEAD request pour être rapide
                    
                    if not dl_resp.ok:
                        continue
                        
                    content_type = dl_resp.headers.get('content-type', '').lower()
                    
                    # ✅ NOUVEAU : Accepter PDF ou DOCX
                    is_pdf = 'application/pdf' in content_type
                    is_docx = 'wordprocessingml.document' in content_type
                    
                    if not (is_pdf or is_docx):
                        continue # Ignorer les pages HTML (office viewer) ou autres
                    
                    # Déterminer la bonne extension
                    ext = '.docx' if is_docx else '.pdf'
                    
                    # Nettoyer le nom de fichier (parfois Microsoft ajoute .pdf à un lien .docx)
                    base_name = doc_url.split("/")[-1].split("?")[0]
                    if base_name.lower().endswith('.pdf') and is_docx:
                        base_name = base_name[:-4] # Retirer le .pdf trompeur
                        
                    file_name = f"{base_name}{ext}"
                    file_path = target_path / file_name
                    
                    if file_path.exists():
                        print(f"⏭️ Déjà présent : {file_name}")
                        continue
                    
                    print(f"⬇️ Téléchargement ({'DOCX' if is_docx else 'PDF'}) : {file_name}...")
                    
                    # Téléchargement réel (GET)
                    dl_resp_get = page.context.request.get(doc_url)
                    if dl_resp_get.ok:
                        with open(file_path, 'wb') as f:
                            f.write(dl_resp_get.body())
                        print(f"✅ Sauvegardé : {file_path}")
                    else:
                        print(f"❌ Échec (HTTP {dl_resp_get.status}) pour {doc_url}")
                        
            except Exception as e:
                print(f"⚠️ Erreur lors du scraping de l'année 20{year} : {e}")
        
        print("\n🛑 Fermeture du navigateur...")
        browser.close()

if __name__ == "__main__":
    scrape_microsoft_annual_reports()