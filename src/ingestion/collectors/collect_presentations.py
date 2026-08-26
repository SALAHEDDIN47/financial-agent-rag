import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

def download_investor_presentations(output_dir: str = "data/raw/presentations"):
    """Scrape les pages web 'Investor Relations' pour trouver et télécharger des PDF."""
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Dictionnaire des pages IR à cibler
    sources = {
        "TSLA": "https://ir.tesla.com/", 
        "AAPL": "https://investor.apple.com/investor-relations/default.aspx"
    }

    # Le User-Agent est crucial pour ne pas être bloqué par les sécurités anti-bots
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for ticker, url in sources.items():
        print(f"\n🔍 Analyse de la page investisseur de {ticker} : {url}")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            pdf_links = []
            
            # Recherche de tous les liens (balises <a>)
            for link in soup.find_all("a", href=True):
                href = link["href"]
                
                # Vérifie si le lien pointe vers un PDF
                if ".pdf" in href.lower():
                    full_url = urljoin(url, href)
                    link_text = link.get_text(strip=True).lower()
                    
                    # Filtre intelligent : on cherche des mots clés spécifiques aux présentations
                    keywords = ["deck", "presentation", "update", "slide", "q1", "q2", "q3", "q4"]
                    if any(keyword in href.lower() or keyword in link_text for keyword in keywords):
                        if full_url not in [p['url'] for p in pdf_links]: # Évite les doublons
                            pdf_links.append({"url": full_url, "text": link_text or "Document PDF"})

            if not pdf_links:
                print(f"⚠️ Aucun PDF pertinent trouvé pour {ticker} (la structure du site a peut-être changé).")
                continue

            # Téléchargement des 2 premiers PDF trouvés pour ne pas surcharger
            for i, pdf in enumerate(pdf_links[:2]):
                print(f"⏳ Téléchargement : {pdf['text']}...")
                pdf_response = requests.get(pdf["url"], headers=headers, timeout=20)
                pdf_response.raise_for_status()

                file_name = f"{ticker}_presentation_{i+1}.pdf"
                file_path = target_path / file_name

                with open(file_path, "wb") as f:
                    f.write(pdf_response.content)
                print(f"✅ Sauvegardé : {file_name}")

        except Exception as e:
            print(f"❌ Erreur lors du scraping de {ticker}: {e}")

if __name__ == "__main__":
    download_investor_presentations()