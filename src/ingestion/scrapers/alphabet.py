from pathlib import Path
from playwright.sync_api import sync_playwright
# from playwright_stealth import stealth_sync  # Décommentez si vous avez installé setuptools

def scrape_alphabet_pdfs(output_dir: str = "data/raw/presentations/alphabet"):
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # On cible directement la page des résultats
    url = "https://abc.xyz/investor/earnings/"
    print(f"🔍 Démarrage de Playwright pour : {url}")
    
    with sync_playwright() as p:
        # headless=False pour voir le navigateur (mettez True pour aller plus vite en prod)
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            accept_downloads=True
        )
        page = context.new_page()

        # stealth_sync(page)  # Décommentez si vous avez installé setuptools
        
        try:
            print("⏳ Chargement de la page...")
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            print("🔍 Analyse des liens...")
            # ✅ FIX 1: Utilisation de l'API moderne (evaluate_all) au lieu de eval_on_selector_all
            all_links = page.locator('a[href]').evaluate_all('elements => elements.map(e => e.href)')
            
            # Filtrer uniquement les liens contenant "pdf" et supprimer les doublons
            pdf_links = list(set([link for link in all_links if 'pdf' in link.lower()]))
            print(f"✅ {len(pdf_links)} fichiers PDF potentiels trouvés.")

            for idx, pdf_url in enumerate(pdf_links):
                try:
                    file_name = pdf_url.split("/")[-1].split("?")[0]
                    if not file_name.lower().endswith('.pdf'):
                        file_name = f"alphabet_report_{idx}.pdf"
                        
                    file_path = target_path / file_name
                    
                    if file_path.exists():
                        print(f"⏭️ Déjà présent : {file_name}")
                        continue
                        
                    print(f"⬇️ Tentative de téléchargement de : {file_name}...")
                    
                    # ✅ FIX 2: Téléchargement direct via l'API Request.
                    # Cliquer sur un PDF ouvre souvent le viewer interne du navigateur, 
                    # ce qui fait planter expect_download. Cette méthode est infaillible.
                    response = page.context.request.get(pdf_url)
                    
                    if response.ok:
                        with open(file_path, 'wb') as f:
                            f.write(response.body())
                        print(f"✅ Sauvegardé avec succès : {file_path}")
                    else:
                        print(f"⚠️ Échec du téléchargement (HTTP {response.status}) pour {pdf_url}")
                        
                except Exception as e:
                    print(f"⚠️ Erreur sur le lien {pdf_url}: {e}")

        except Exception as e:
            print(f"❌ Erreur lors de l'exécution : {e}")
        finally:
            print("🛑 Fermeture du navigateur...")
            browser.close()

if __name__ == "__main__":
    scrape_alphabet_pdfs()